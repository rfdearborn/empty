"""Investigate what's driving the 2026 Q1 uptick in 7-train morning disruptions."""

import io
import re
from collections import Counter

import pandas as pd
import requests

DATASET_URL = "https://data.ny.gov/resource/7kct-peq7.csv"

# Fetch all 6-10am weekday 7-line alerts since 2022-05-09
params = {
    "$select": "event_id,date,status_label,affected,header",
    "$where": (
        "affected like '%7%' "
        "AND date >= '2022-05-09T00:00:00' "
        "AND date_extract_hh(date) >= 6 AND date_extract_hh(date) < 10 "
        "AND date_extract_dow(date) BETWEEN 1 AND 5"
    ),
    "agency": "NYCT Subway",
    "$order": "date",
    "$limit": 50000,
}
r = requests.get(DATASET_URL, params=params, timeout=120)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))
# token filter
df = df[df["affected"].apply(lambda s: any(t.strip() in ("7", "7X") for t in str(s).split("|")))].copy()
# drop excluded labels
EX = {"station-notice", "planned-work"}
df = df[~df["status_label"].apply(lambda s: bool({t.strip() for t in str(s).split("|")} & EX))].copy()
df["date"] = pd.to_datetime(df["date"])
df["year_q"] = df["date"].dt.to_period("Q").astype(str)

print(f"=== Total 6-10am weekday 7-line alert rows by quarter ===")
by_q = df.groupby("year_q").agg(
    rows=("event_id", "count"),
    events=("event_id", "nunique"),
).reset_index()
print(by_q.to_string(index=False))

print()
print("=== Status label mix, 2026-Q1 vs prior 12 months ===")
recent = df[df["date"] >= "2026-01-01"]
prior = df[(df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")]
def status_share(d):
    s = d["status_label"].value_counts(normalize=True).head(8)
    return s.round(3)
print("2026-Q1:")
print(status_share(recent))
print()
print("2025 full year:")
print(status_share(prior))

# Extract reason keywords from header text
print()
print("=== Reason keywords in 6-10am alert headers ===")
def reasons(d, label):
    text = " ".join(d["header"].dropna().astype(str).str.lower())
    patterns = {
        "signal problem": r"\bsignal (problem|malfunction|issue)",
        "track work/maintenance": r"\btrack (work|maintenance|repair|replacement)|\brail (replacement|repair)|broken rail",
        "scheduled maintenance": r"\bscheduled maintenance",
        "sick passenger/EMS": r"\bsomeone in need\b|\bems\b|\bmedical\b|\bsick passenger\b",
        "police/NYPD": r"\bnypd\b|police|investigat",
        "person on tracks": r"\bperson on (the )?track|trespass|unauthorized person",
        "mechanical": r"\bmechanical (problem|issue)|\btrain with mechanical",
        "brakes activated": r"\bbrake",
        "switch problem": r"\bswitch (problem|malfunction)",
        "debris/object": r"\bdebris|\bobject on (the )?track",
        "power/third rail": r"\bpower (problem|outage)|\bloss of power|\bthird rail",
        "smoke/fire/FDNY": r"\bsmoke\b|\bfire\b|\bfdny\b",
        "weather": r"\bweather\b|\bsnow\b|\bice\b|\bflood\b|\brain\b|\bheavy rain\b",
    }
    counts = {name: len(re.findall(p, text)) for name, p in patterns.items()}
    total = max(sum(counts.values()), 1)
    print(f"  {label}: total mentions={total}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        if v:
            print(f"    {k:24s} {v:4d}  ({100*v/total:.0f}%)")

reasons(recent, "2026-Q1 (Jan-Mar)")
print()
reasons(prior, "2025 full year")

print()
print("=== Q1 same-quarter comparison ===")
for year in (2023, 2024, 2025, 2026):
    q1 = df[(df["date"] >= f"{year}-01-01") & (df["date"] < f"{year}-04-01")]
    print(f"  {year} Q1: {q1['event_id'].nunique()} distinct morning events, {len(q1)} alert rows")

# Stations mentioned
print()
print("=== Top stations in 2026-Q1 6-10am headers ===")
stations_recent = re.findall(
    r"at ([A-Z][A-Za-z0-9 \-]+?(?:St|Av|Sq|Blvd|Plaza|Yards|Pl))",
    " ".join(recent["header"].dropna().astype(str)),
)
print(Counter(stations_recent).most_common(10))

print()
print("=== Sample 2026-Q1 morning alert headers (first updates of distinct events) ===")
first_updates = recent.sort_values(["event_id", "date"]).drop_duplicates("event_id")
for _, r in first_updates.iterrows():
    print(f"  {r['date'].strftime('%Y-%m-%d %H:%M')}  [{r['status_label']}]  {r['header'][:160]}")
