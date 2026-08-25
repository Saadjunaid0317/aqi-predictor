import os
import csv
import pandas as pd
from dotenv import load_dotenv
import hopsworks

from feature_pipeline import build_feature_record

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HISTORY_FILE = "recent_history.csv"

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
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["timestamp"]) >= cutoff]
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

print("Inserted record into Hopsworks feature group.")