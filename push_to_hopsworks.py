import os
import csv
import pandas as pd
from dotenv import load_dotenv
import hopsworks

from feature_pipeline import build_feature_record

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HISTORY_FILE = "recent_history.csv"
ALERTS_LOG_FILE = "alerts_log.csv"
ALERT_THRESHOLD = 151   # "Unhealthy" and worse
ALERT_RETENTION_DAYS = 30

def log_alert_if_hazardous(record):
    if record["aqi"] < ALERT_THRESHOLD:
        return

    alert_row = {"timestamp": record["timestamp"], "aqi": record["aqi"],
                 "dominant_pollutant": record["dominant_pollutant"]}
    file_exists = os.path.exists(ALERTS_LOG_FILE)
    with open(ALERTS_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=alert_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(alert_row)

    cutoff = record["timestamp"] - (ALERT_RETENTION_DAYS * 24 * 3600)
    with open(ALERTS_LOG_FILE, "r") as f:
        all_rows = list(csv.DictReader(f))
    rows = []
    for row in all_rows:
        if int(row["timestamp"]) >= cutoff:
            rows.append(row)
    with open(ALERTS_LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=alert_row.keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"ALERT: AQI {record['aqi']} >= {ALERT_THRESHOLD} - logged to {ALERTS_LOG_FILE}")

def append_to_local_history(record):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    # Keep only the last 14 days, so this file never grows unbounded
    cutoff = record["timestamp"] - (14 * 24 * 3600)
    with open(HISTORY_FILE, "r") as f:
        all_rows = list(csv.DictReader(f))
    rows = []
    for row in all_rows:
        if int(row["timestamp"]) >= cutoff:
            rows.append(row)
    with open(HISTORY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        writer.writeheader()
        writer.writerows(rows)

record = build_feature_record()
df = pd.DataFrame([record])

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=1,
    description="Weather, pollution, calculated AQI, and time features for Karachi",
    primary_key=["timestamp"],
    event_time="timestamp",
    time_travel_format="HUDI"
)

fg.insert(df)
append_to_local_history(record)
log_alert_if_hazardous(record)

print("Inserted record into Hopsworks feature group.")