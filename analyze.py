"""
7 train weekday morning commute disruption analysis.

Defines a "disrupted weekday" as: at least one NYCT Subway service alert
affecting the 7 (or 7X) line was active during any minute of 6:00-9:59am ET
on that weekday. Weekly probability = disrupted_weekdays / weekdays_in_week,
restricted to weekdays after the dataset start (2022-05-09) and on/before today.

Data: data.ny.gov MTA Service Alerts (dataset 7kct-peq7), April 2020-present.
"""

import io
import sys
from datetime import date, datetime, time, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import requests

DATASET_URL = "https://data.ny.gov/resource/7kct-peq7.csv"
DATASET_JSON = "https://data.ny.gov/resource/7kct-peq7.json"
START = date(2022, 5, 9)          # Monday, ~4 years before 2026-05-13

MORNING_START = time(6, 0)
MORNING_END = time(10, 0)         # exclusive

# Status labels that don't represent an active morning-commute disruption
EXCLUDED_STATUS = {"station-notice", "planned-work"}


def fetch_data_end() -> date:
    """Latest NYCT Subway alert date in the dataset (data has a publishing lag)."""
    r = requests.get(
        DATASET_JSON,
        params={"$select": "max(date) as m", "agency": "NYCT Subway"},
        timeout=60,
    )
    r.raise_for_status()
    return pd.to_datetime(r.json()[0]["m"]).date()


def last_full_friday(through: date) -> date:
    """Friday of the last Mon-Fri week fully covered by `through`."""
    # Need data through Friday of the target week
    wd = through.weekday()  # Mon=0 ... Sun=6
    if wd >= 4:  # Fri (4), Sat (5), Sun (6) — this week's Friday is covered
        return through - timedelta(days=wd - 4)
    # Mon (0)..Thu (3): only previous week is fully covered
    return through - timedelta(days=wd + 3)


def fetch_alerts(end: date) -> pd.DataFrame:
    """Fetch all NYCT Subway alerts mentioning the 7 line in [START, end]."""
    where = (
        f"affected like '%7%' "
        f"AND date >= '{START.isoformat()}T00:00:00' "
        f"AND date < '{(end + timedelta(days=1)).isoformat()}T00:00:00'"
    )
    params = {
        "$select": "alert_id,event_id,update_number,date,status_label,affected",
        "$where": where,
        "agency": "NYCT Subway",
        "$order": "date",
        "$limit": 50000,
    }
    r = requests.get(DATASET_URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # Keep only rows where '7' or '7X' is an actual token in the affected list
    def has_7(s: str) -> bool:
        tokens = [t.strip() for t in str(s).split("|")]
        return "7" in tokens or "7X" in tokens
    df = df[df["affected"].apply(has_7)].copy()
    # Drop excluded status labels (incl. compound labels containing them)
    def is_excluded(s: str) -> bool:
        tokens = {t.strip() for t in str(s).split("|")}
        return bool(tokens & EXCLUDED_STATUS)
    excluded_mask = df["status_label"].apply(is_excluded)
    print(f"  dropping {excluded_mask.sum()} alert rows with excluded status labels", flush=True)
    df = df[~excluded_mask].copy()
    df["date"] = pd.to_datetime(df["date"])  # naive, treat as ET
    return df


def event_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse each event_id to a [start, end] window across its updates."""
    grp = df.groupby("event_id")["date"].agg(["min", "max"]).reset_index()
    grp.columns = ["event_id", "start", "end"]
    # For point-in-time alerts (single update), give a tiny buffer so overlap math works
    same = grp["start"] == grp["end"]
    grp.loc[same, "end"] = grp.loc[same, "start"] + pd.Timedelta(minutes=1)
    return grp


def disrupted_weekdays(events: pd.DataFrame, end: date) -> pd.DataFrame:
    """For every weekday in [START, end], compute whether any event overlapped 6-10am ET."""
    weekdays = pd.bdate_range(START, end).date  # Mon-Fri only, federal holidays NOT excluded here
    rows = []
    # Pre-build morning intervals as tuples of pd.Timestamp
    morning = []
    for d in weekdays:
        s = pd.Timestamp.combine(d, MORNING_START)
        e = pd.Timestamp.combine(d, MORNING_END)
        morning.append((d, s, e))
    morning_df = pd.DataFrame(morning, columns=["weekday", "m_start", "m_end"])

    # Vectorized overlap: for each weekday, count events where start < m_end and end > m_start
    # Sort events by start once; then per weekday filter via binary search.
    ev = events.sort_values("start").reset_index(drop=True)
    starts = ev["start"].values
    ends = ev["end"].values

    import numpy as np
    starts_ns = pd.to_datetime(starts).astype("datetime64[ns]").astype("int64")
    ends_ns = pd.to_datetime(ends).astype("datetime64[ns]").astype("int64")

    disrupted = []
    for _, row in morning_df.iterrows():
        m_start_ns = row["m_start"].to_datetime64().astype("datetime64[ns]").astype("int64")
        m_end_ns = row["m_end"].to_datetime64().astype("datetime64[ns]").astype("int64")
        # event overlaps morning if start < m_end and end > m_start
        overlap = (starts_ns < m_end_ns) & (ends_ns > m_start_ns)
        disrupted.append(bool(overlap.any()))
    morning_df["disrupted"] = disrupted
    return morning_df[["weekday", "disrupted"]]


def weekly_probability(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily disruption flags to ISO weeks (week starting Monday)."""
    d = daily.copy()
    d["weekday"] = pd.to_datetime(d["weekday"])
    d["week_start"] = d["weekday"] - pd.to_timedelta(d["weekday"].dt.weekday, unit="D")
    weekly = d.groupby("week_start").agg(
        disrupted_days=("disrupted", "sum"),
        total_days=("disrupted", "count"),
    ).reset_index()
    weekly["probability"] = weekly["disrupted_days"] / weekly["total_days"]
    # Drop partial trailing week if it has < 5 weekdays
    weekly = weekly[weekly["total_days"] == 5].reset_index(drop=True)
    return weekly


def plot(weekly: pd.DataFrame, data_end: date, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(
        weekly["week_start"], weekly["probability"],
        color="#1f77b4", linewidth=1.1, marker="o", markersize=2.5,
        label="Weekly probability",
    )
    weekly["rolling_4w"] = weekly["probability"].rolling(4, min_periods=1).mean()
    ax.plot(
        weekly["week_start"], weekly["rolling_4w"],
        color="#d62728", linewidth=2.2, label="4-week rolling average",
    )
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(≥1 alert active 6–10am, weekday)")
    ax.set_xlabel("Week (starting Monday)")
    ax.set_title(
        "NYC 7 Train — Weekday Morning Commute Disruption Probability\n"
        f"Any NYCT Subway service alert affecting 7/7X active during 6–10am ET "
        f"(data through {data_end:%Y-%m-%d})",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> int:
    data_end = fetch_data_end()
    end = last_full_friday(data_end)
    print(f"dataset latest NYCT Subway alert: {data_end}")
    print(f"trimming to last full Mon-Fri week ending: {end}")
    df = fetch_alerts(end)
    print(f"  {len(df):,} alert rows after 7/7X token filter", flush=True)
    events = event_windows(df)
    print(f"  {len(events):,} distinct events", flush=True)
    daily = disrupted_weekdays(events, end)
    print(f"  {len(daily):,} weekdays scanned; {daily['disrupted'].sum():,} disrupted", flush=True)
    weekly = weekly_probability(daily)
    weekly_out = weekly[["week_start", "disrupted_days", "total_days", "probability"]].copy()
    weekly_out["week_start"] = weekly_out["week_start"].dt.date
    weekly_out.to_csv("weekly_disruption.csv", index=False)
    print(f"wrote weekly_disruption.csv ({len(weekly_out)} weeks)")
    plot(weekly, data_end, "weekly_disruption.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
