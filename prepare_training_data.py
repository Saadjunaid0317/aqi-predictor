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

def load_training_data():
    df = read_feature_group_in_chunks(fg, "2023-08-11", "2026-08-16", chunk_days=60)
    print(f"\nTotal rows read: {len(df)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamp_to_aqi = dict(zip(df["timestamp"], df["aqi"]))

    def get_window_average(row, start_hours, end_hours, min_coverage=0.5):
        # Average AQI over a future window (e.g. hours 1-24 for "Day 1").
        # min_coverage=0.5 means: if more than half the hours in this window
        # are missing (a data gap), skip it rather than average a handful of
        # values and call it representative of a full day.
        values = []
        for h in range(start_hours, end_hours + 1):
            ts = row["timestamp"] + h * 3600
            if ts in timestamp_to_aqi:
                values.append(timestamp_to_aqi[ts])
        expected = end_hours - start_hours + 1
        if len(values) < expected * min_coverage:
            return None
        return sum(values) / len(values)

    # Per mentor guidance: Day 1 = hours 1-24, Day 2 = hours 25-48, Day 3 = hours 49-72,
    # each target being the AVERAGE AQI over that day, not one instantaneous point.
    df["target_aqi_24h"] = df.apply(lambda row: get_window_average(row, 1, 24), axis=1)
    df["target_aqi_48h"] = df.apply(lambda row: get_window_average(row, 25, 48), axis=1)
    df["target_aqi_72h"] = df.apply(lambda row: get_window_average(row, 49, 72), axis=1)

    # --- lag/trend features (kept - cheap and still theoretically useful) ---
    def make_lag_feature(hours_ago):
        seconds_ago = hours_ago * 3600
        return df.apply(lambda row: timestamp_to_aqi.get(row["timestamp"] - seconds_ago), axis=1)

    df["aqi_24h_ago"] = make_lag_feature(24)
    df["aqi_48h_ago"] = make_lag_feature(48)
    df["aqi_72h_ago"] = make_lag_feature(72)
    df["aqi_change_24h"] = df["aqi"] - df["aqi_24h_ago"]

    df["_datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("_datetime")
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling("24h").mean()
    df["aqi_rolling_std_24h"] = df["aqi"].rolling("24h").std()
    df = df.reset_index(drop=True)

    required_columns = [
        "target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
        "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
        "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h"
    ]
    before = len(df)
    df = df.dropna(subset=required_columns)
    after = len(df)
    print(f"Started with {before} rows, dropped {before - after}, training dataset: {after} rows")

    feature_columns = [
        "hour", "day", "month", "day_of_week",
        "temp", "humidity", "wind_speed",
        "pm2_5", "pm10", "co", "no2", "so2", "o3",
        "aqi", "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
        "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h"
    ]

    X = df[feature_columns]
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]

    targets = {}
    for horizon in ["24h", "48h", "72h"]:
        y = df[f"target_aqi_{horizon}"]
        targets[horizon] = (y.iloc[:split_index], y.iloc[split_index:])

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows")
    return X_train, X_test, targets

if __name__ == "__main__":
    load_training_data()