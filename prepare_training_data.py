import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)

def read_feature_group_in_chunks(fg, start_date, end_date, chunk_days=60):
    start_ts = int(datetime.fromisoformat(start_date).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).timestamp())
    chunk_seconds = chunk_days * 86400

    chunks = []
    current_start = start_ts

    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        print(f"Reading {datetime.fromtimestamp(current_start).date()} to {datetime.fromtimestamp(current_end).date()}...")

        query = fg.filter((fg.timestamp >= current_start) & (fg.timestamp < current_end))
        chunk_df = query.read()
        print(f"  -> {len(chunk_df)} rows")
        chunks.append(chunk_df)

        current_start = current_end

    return pd.concat(chunks, ignore_index=True)

df = read_feature_group_in_chunks(fg, "2023-08-11", "2026-08-16", chunk_days=60)
print(f"\nTotal rows read: {len(df)}")

df = df.sort_values("timestamp").reset_index(drop=True)

HOURS_AHEAD = 3 * 24
SECONDS_AHEAD = HOURS_AHEAD * 3600
timestamp_to_aqi = dict(zip(df["timestamp"], df["aqi"]))

def get_future_aqi(row):
    target_timestamp = row["timestamp"] + SECONDS_AHEAD
    return timestamp_to_aqi.get(target_timestamp)

df["target_aqi"] = df.apply(get_future_aqi, axis=1)

before = len(df)
df = df.dropna(subset=["target_aqi"])
after = len(df)

print(f"Started with {before} rows")
print(f"Dropped {before - after} rows with no 3-day-ahead target available")
print(f"Training dataset: {after} rows")

feature_columns = [
    "hour", "day", "month", "day_of_week",
    "temp", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "aqi"
]

X = df[feature_columns]
y = df["target_aqi"]

split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training on {len(X_train)} rows (older data)")
print(f"Testing on {len(X_test)} rows (most recent data)")