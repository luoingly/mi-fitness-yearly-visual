import argparse
import json
from datetime import timedelta, timezone
from math import atan2
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


# Fix dash pattern issue
_original_line2d_draw = mlines.Line2D.draw


def _safe_line2d_draw(self, renderer):
    offset, pattern = self._dash_pattern
    if pattern is None or not any(value > 0 for value in pattern):
        self._dash_pattern = (0.0, [1.0])
    return _original_line2d_draw(self, renderer)


mlines.Line2D.draw = _safe_line2d_draw


def parse_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        return None


def circular_mean(hours: pd.Series) -> float:
    values = hours.dropna()
    if values.empty:
        return float("nan")
    angles = np.array(values) / 24 * 2 * np.pi
    mean_angle = atan2(np.sin(angles).sum(), np.cos(angles).sum())
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    return mean_angle / (2 * np.pi) * 24


def hour_to_label(hour_float: float) -> str:
    if pd.isna(hour_float):
        return "--:--"
    minutes = int(round((hour_float % 1) * 60))
    hour = int(hour_float) % 24
    return f"{hour:02d}:{minutes:02d}"


def load_data(data_file: Path, start_date: pd.Timestamp, end_date: pd.Timestamp, local_tz: timezone) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    df["dt_utc"] = pd.to_datetime(df["Time"], unit="s", utc=True)
    df["dt_local"] = df["dt_utc"].dt.tz_convert(local_tz)
    df["date_local"] = df["dt_local"].dt.date

    mask = (df["dt_local"] >= start_date) & (df["dt_local"] <= end_date)
    df = df.loc[mask].copy()

    tag_rank = {"daily_report": 2, "daily_mark": 1, "daily_fitness": 1}
    df["tag_rank"] = df["Tag"].map(tag_rank).fillna(0)
    df.sort_values(["date_local", "tag_rank", "dt_local"],
                   ascending=[True, False, True], inplace=True)
    df = df.drop_duplicates(subset=["date_local", "Key"], keep="first")

    df["month"] = pd.to_datetime(df["date_local"]).dt.to_period("M")
    return df


def build_sleep_data(df: pd.DataFrame) -> pd.DataFrame:
    sleep_rows = []
    for _, row in df[df["Key"] == "sleep"].iterrows():
        payload = parse_json(row["Value"])
        if not payload:
            continue

        segs = payload.get("segment_details") or []
        tz_code = payload.get("timezone")
        bedtime_ts = None
        wake_ts = None

        for seg in segs:
            tz_code = seg.get("timezone", tz_code)
            if "bedtime" in seg:
                bedtime_ts = seg["bedtime"] if bedtime_ts is None else min(
                    bedtime_ts, seg["bedtime"])
            if "wake_up_time" in seg:
                wake_ts = seg["wake_up_time"] if wake_ts is None else max(
                    wake_ts, seg["wake_up_time"])

        tz_code = tz_code or 32
        tz_offset = timedelta(seconds=int(tz_code) * 900)
        tzinfo = timezone(tz_offset)

        def to_hour(ts: Optional[int]) -> float:
            if ts is None:
                return float("nan")
            dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tzinfo)
            return dt.hour + dt.minute / 60 + dt.second / 3600

        local_date = pd.to_datetime(row["date_local"])
        sleep_rows.append({
            "date_local": row["date_local"],
            "month": row["month"],
            "week": local_date.to_period("W-MON"),
            "weekday": local_date.day_name(),
            "bed_hour": to_hour(bedtime_ts),
            "wake_hour": to_hour(wake_ts),
            "total_min": payload.get("total_duration"),
            "deep_min": payload.get("sleep_deep_duration"),
            "rem_min": payload.get("sleep_rem_duration"),
            "light_min": payload.get("sleep_light_duration"),
            "awake_min": payload.get("sleep_awake_duration"),
        })

    sleep_df = pd.DataFrame(sleep_rows)
    if not sleep_df.empty:
        weekday_order = ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"]
        sleep_df["weekday"] = pd.Categorical(
            sleep_df["weekday"], categories=weekday_order, ordered=True)
    return sleep_df


def build_steps_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df[df["Key"] == "steps"].iterrows():
        payload = parse_json(row["Value"])
        if not payload:
            continue

        local_date = pd.to_datetime(row["date_local"])
        rows.append({
            "date_local": row["date_local"],
            "month": row["month"],
            "week": local_date.to_period("W-MON"),
            "weekday": local_date.day_name(),
            "steps": payload.get("steps"),
            "distance": payload.get("distance"),
            "calories": payload.get("calories"),
        })

    steps_df = pd.DataFrame(rows)
    if not steps_df.empty:
        weekday_order = ["Monday", "Tuesday", "Wednesday",
                         "Thursday", "Friday", "Saturday", "Sunday"]
        steps_df["weekday"] = pd.Categorical(
            steps_df["weekday"], categories=weekday_order, ordered=True)
    return steps_df


def cute_axes(ax: plt.Axes, facecolor: str = "#fff6f2") -> None:
    ax.set_facecolor(facecolor)
    for spine in ax.spines.values():
        spine.set_color("#ffb6a1")
        spine.set_linewidth(1.1)
    ax.tick_params(colors="#7a5a4f")
    ax.grid(True, axis="y", linestyle="--", color="#fcd9c7", alpha=0.7)


def save_figure(fig: plt.Figure, path: Path, format: str) -> None:
    fig.tight_layout(pad=1.2)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick_line in axis.get_ticklines():
                tick_line.set_dashes([1.0])

    if format == "png":
        fig.savefig(path.with_suffix(".png"), format="png", dpi=150)
    elif format == "svg":
        fig.savefig(path.with_suffix(".svg"), format="svg")
    else:
        raise ValueError(f"Unsupported format: {format}")


def plot_sleep_clock_monthly(monthly_data: pd.DataFrame, out_path: Path, format: str, font_family: str) -> None:
    if monthly_data.empty:
        return

    months = monthly_data["month"].dt.month.tolist()
    bed_hours = monthly_data["bed_hour"].tolist()
    wake_hours = monthly_data["wake_hour"].tolist()

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(10, 5))
        cute_axes(ax)

        ax.plot(months, bed_hours, marker="P", markersize=10,
                linewidth=2, color="#f26a7e", label="Bed time")
        ax.plot(months, wake_hours, marker="X", markersize=8,
                linewidth=2, color="#43a5ff", label="Wake time")

        for x, y in zip(months, bed_hours):
            if not pd.isna(y):
                ax.annotate(hour_to_label(y), (x, y), xytext=(0, 8), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, color="#f26a7e")

        for x, y in zip(months, wake_hours):
            if not pd.isna(y):
                ax.annotate(hour_to_label(y), (x, y), xytext=(0, -12), textcoords='offset points',
                            ha='center', va='top', fontsize=8, color="#43a5ff")

        ax.set_title("Average sleep clock by month")
        ax.set_ylabel("Hour of day")
        ax.set_ylim(-1, 23)
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([str(h) for h in range(0, 24, 2)])
        ax.set_xlabel("Month")
        ax.set_xticks(months)
        ax.set_xticklabels([str(m) for m in months])
        ax.legend(frameon=False)
        save_figure(fig, out_path, format)


def plot_steps_monthly(monthly_data: pd.DataFrame, out_path: Path, format: str, font_family: str) -> None:
    if monthly_data.empty:
        return

    months = monthly_data["month"].dt.month.tolist()
    steps_values = monthly_data["steps"].tolist()

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(10, 5))
        cute_axes(ax)

        ax.bar(months, steps_values, color="#9c6ef0",
               alpha=0.85, edgecolor="#f6d2ff", linewidth=0.8)

        for x, y in zip(months, steps_values):
            if not pd.isna(y):
                ax.text(x, y + max(steps_values) * 0.01, f"{int(round(y))}",
                        ha='center', va='bottom', fontsize=9, color="#5a3f4c")

        ax.set_title("Monthly average steps")
        ax.set_ylabel("Steps per day")
        ax.set_xlabel("Month")
        ax.set_xticks(months)
        ax.set_xticklabels([str(m) for m in months])
        ax.set_ylim(0, max(steps_values) * 1.2)
        save_figure(fig, out_path, format)


def plot_steps_weekday(weekday_data: pd.DataFrame, out_path: Path, format: str, font_family: str) -> None:
    steps = weekday_data.dropna(subset=["steps"]).copy()
    if steps.empty:
        return

    labels = steps["weekday"].astype(str).tolist()
    values = steps["steps"].tolist()

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(9, 5))
        cute_axes(ax)

        ax.bar(labels, values, color="#f26a7e", alpha=0.85,
               edgecolor="#fbd7e6", linewidth=0.9)

        for i, (label, value) in enumerate(zip(labels, values)):
            if not pd.isna(value):
                ax.text(i, value + max(values) * 0.01, f"{int(round(value))}",
                        ha='center', va='bottom', fontsize=9, color="#5a3f4c")

        ax.set_title("Weekday average steps")
        ax.set_ylabel("Steps per day")
        ax.set_xlabel("Weekday")
        ax.set_ylim(0, max(values) * 1.2)
        save_figure(fig, out_path, format)


def plot_sleep_clock_weekday(weekday_data: pd.DataFrame, out_path: Path, format: str, font_family: str) -> None:
    entries = weekday_data.dropna(subset=["bed_hour", "wake_hour"]).copy()
    if entries.empty:
        return

    labels = entries["weekday"].astype(str).tolist()
    bed_hours = entries["bed_hour"].tolist()
    wake_hours = entries["wake_hour"].tolist()

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(10, 5))
        cute_axes(ax)

        ax.plot(labels, bed_hours, marker="P", markersize=10,
                linewidth=2, color="#f26a7e", label="Bed time")
        ax.plot(labels, wake_hours, marker="X", markersize=8,
                linewidth=2, color="#43a5ff", label="Wake time")

        for i, (label, y) in enumerate(zip(labels, bed_hours)):
            if not pd.isna(y):
                ax.annotate(hour_to_label(y), (i, y), xytext=(0, 8), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, color="#f26a7e")

        for i, (label, y) in enumerate(zip(labels, wake_hours)):
            if not pd.isna(y):
                ax.annotate(hour_to_label(y), (i, y), xytext=(0, -12), textcoords='offset points',
                            ha='center', va='top', fontsize=8, color="#43a5ff")

        ax.set_title("Average sleep clock by weekday")
        ax.set_ylabel("Hour of day")
        ax.set_ylim(-1, 23)
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([str(h) for h in range(0, 24, 2)])
        ax.set_xlabel("Weekday")
        ax.legend(frameon=False)
        save_figure(fig, out_path, format)


def plot_sleep_stage_share(stage_share: dict[str, float], out_path: Path, format: str, font_family: str) -> None:
    if not stage_share:
        return

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(6, 6))
        cute_axes(ax)

        labels = list(stage_share.keys())
        values = list(stage_share.values())
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#40c4ff", "#ffb347", "#9c6ef0", "#ffc1c1"],
            wedgeprops=dict(edgecolor="#fffdf8", width=0.4),
        )

        for text in texts + autotexts:
            text.set_color("#5a3f4c")

        ax.set_title("Average sleep stage share")
        save_figure(fig, out_path, format)


def plot_sleep_stage_monthly(monthly_data: pd.DataFrame, out_path: Path, format: str, font_family: str) -> None:
    if monthly_data.empty:
        return

    months = monthly_data["month"].dt.month.tolist()
    data = [str(m) for m in months]
    deep = monthly_data["deep_min"] / 60
    rem = monthly_data["rem_min"] / 60
    light = monthly_data["light_min"] / 60

    with plt.xkcd():
        plt.rcParams["font.family"] = font_family
        fig, ax = plt.subplots(figsize=(10, 5))
        cute_axes(ax)

        ax.bar(data, deep, label="Deep", color="#2e8b57")
        ax.bar(data, rem, bottom=deep, label="REM", color="#ff9f68")
        ax.bar(data, light, bottom=deep + rem, label="Light", color="#fdd835")

        for i in range(len(data)):
            if not pd.isna(deep.iloc[i]):
                ax.text(i, deep.iloc[i]/2, f"{deep.iloc[i]:.1f}",
                        ha='center', va='center', fontsize=8, color="#5a3f4c")

            if not pd.isna(rem.iloc[i]):
                ax.text(i, deep.iloc[i] + rem.iloc[i]/2, f"{rem.iloc[i]:.1f}",
                        ha='center', va='center', fontsize=8, color="#5a3f4c")

            if not pd.isna(light.iloc[i]):
                total_height = deep.iloc[i] + rem.iloc[i] + light.iloc[i]
                ax.text(i, deep.iloc[i] + rem.iloc[i] + light.iloc[i]/2, f"{light.iloc[i]:.1f}",
                        ha='center', va='center', fontsize=8, color="#5a3f4c")

        ax.set_ylabel("Hours per night")
        ax.set_title("Average sleep stages by month")
        ax.set_xlabel("Month")
        ax.legend()
        save_figure(fig, out_path, format)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fitness data visualizations")
    parser.add_argument("input", type=Path,
                        help="Input CSV data file")
    parser.add_argument("--year", type=int, default=2025,
                        help="Year of the data to analyze")
    parser.add_argument("--timezone", type=int, default=8,
                        help="Timezone offset in hours (e.g., 8 for UTC+8)")
    parser.add_argument("--format", choices=["png", "svg"], default="png",
                        help="Output format for plots")
    parser.add_argument("--output", type=Path, default="output",
                        help="Output directory for plots")
    parser.add_argument("--font", type=str, default="xkcd Script",
                        help="Font family for plots")

    args = parser.parse_args()

    # Setup paths and settings
    args.output.mkdir(exist_ok=True)

    LOCAL_TZ = timezone(timedelta(hours=args.timezone))
    START_DATE = pd.Timestamp(year=args.year, month=1, day=1, tz=LOCAL_TZ)
    END_DATE = pd.Timestamp(year=args.year, month=12,
                            day=31, hour=23, minute=59, second=59, tz=LOCAL_TZ)

    # Load and process data
    try:
        df = load_data(args.input, START_DATE, END_DATE, LOCAL_TZ)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure the input file is 'hlth_center_aggregated_fitness_data.csv' exported from Mi Fit.")
        return
    print(f"Records in range: {len(df)}")

    sleep_df = build_sleep_data(df)
    steps_df = build_steps_data(df)

    # Monthly sleep stats
    monthly_sleep = (sleep_df.groupby("month")
                     .agg(bed_hour=("bed_hour", circular_mean),
                          wake_hour=("wake_hour", circular_mean),
                          total_min=("total_min", "mean"),
                          rem_min=("rem_min", "mean"),
                          deep_min=("deep_min", "mean"),
                          light_min=("light_min", "mean"))
                     .reset_index()
                     .sort_values("month"))

    # Annual sleep stage share
    stage_totals = {
        "Deep": sleep_df["deep_min"].sum(),
        "REM": sleep_df["rem_min"].sum(),
        "Light": sleep_df["light_min"].sum(),
        "Awake": sleep_df["awake_min"].sum(),
    }
    total_stage = sum(v for v in stage_totals.values() if pd.notna(v))
    stage_share = {k: (v / total_stage * 100 if total_stage else 0)
                   for k, v in stage_totals.items()}

    # Steps data
    steps_monthly = steps_df.groupby(
        "month")["steps"].mean().reset_index().sort_values("month")
    weekday_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
    steps_weekday = (steps_df.groupby("weekday", observed=True)["steps"]
                     .mean()
                     .reindex(weekday_order)
                     .rename_axis("weekday")
                     .reset_index())
    sleep_weekday = (sleep_df.groupby("weekday", observed=True)
                     .agg(bed_hour=("bed_hour", circular_mean),
                          wake_hour=("wake_hour", circular_mean))
                     .reindex(weekday_order)
                     .rename_axis("weekday")
                     .reset_index())

    # Generate plots
    plot_steps_monthly(
        steps_monthly, args.output / "steps_monthly", args.format, args.font)
    plot_steps_weekday(
        steps_weekday, args.output / "steps_weekday", args.format, args.font)
    plot_sleep_clock_monthly(
        monthly_sleep, args.output / "sleep_clock_monthly", args.format, args.font)
    plot_sleep_clock_weekday(
        sleep_weekday, args.output / "sleep_clock_weekday", args.format, args.font)
    plot_sleep_stage_monthly(
        monthly_sleep, args.output / "sleep_stage_monthly", args.format, args.font)
    plot_sleep_stage_share(
        stage_share, args.output / "sleep_stage_share", args.format, args.font)

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
