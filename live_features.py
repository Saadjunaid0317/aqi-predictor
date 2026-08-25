import os
import pandas as pd
from datetime import datetime
from feature_pipeline import build_feature_record

HISTORY_FILE = "recent_history.csv"

def get_live_input():
    current = build_feature_record()
    now_ts = current["timestamp"]

    if os.path.exists(HISTORY_FILE):
        history = pd.read_csv(HISTORY_FILE)
    else:
        history = pd.DataFrame(columns=["timestamp", "aqi"])

    if len(history) == 0:
        # No local history yet (fresh checkout, before the hourly job has run) -
        # fall back to "no change" rather than crashing.
        aqi_24h_ago = current["aqi"]
        aqi_48h_ago = current["aqi"]
        aqi_72h_ago = current["aqi"]
        gaps = {"24h": None, "48h": None, "72h": None}
        rolling_mean = current["aqi"]
        rolling_std = 0.0
    else:
        timestamp_to_aqi = dict(zip(history["timestamp"], history["aqi"]))

        def find_nearest_aqi(target_ts):
            # Go through every timestamp we have and remember whichever one
            # is closest to the target - simple loop instead of a fancy min().
            closest_ts = None
            smallest_gap = None
            for ts in history["timestamp"]:
                gap = abs(ts - target_ts)
                if smallest_gap is None or gap < smallest_gap:
                    smallest_gap = gap
                    closest_ts = ts
            gap_hours = smallest_gap / 3600
            return timestamp_to_aqi[closest_ts], gap_hours

        aqi_24h_ago, gap24 = find_nearest_aqi(now_ts - 24 * 3600)
        aqi_48h_ago, gap48 = find_nearest_aqi(now_ts - 48 * 3600)
        aqi_72h_ago, gap72 = find_nearest_aqi(now_ts - 72 * 3600)
        gaps = {"24h": gap24, "48h": gap48, "72h": gap72}
        print(f"Lookup gaps (hours off from exact target): 24h->{gap24:.1f}, 48h->{gap48:.1f}, 72h->{gap72:.1f}")

        recent_24h = history[history["timestamp"] >= now_ts - 24 * 3600]
        if len(recent_24h) > 0:
            rolling_mean = recent_24h["aqi"].mean()
        else:
            rolling_mean = current["aqi"]
        if len(recent_24h) > 1:
            rolling_std = recent_24h["aqi"].std()
        else:
            rolling_std = 0.0

    row = {
        "hour": current["hour"], "day": current["day"], "month": current["month"],
        "day_of_week": current["day_of_week"], "temp": current["temp"],
        "humidity": current["humidity"], "wind_speed": current["wind_speed"],
        "pm2_5": current["pm2_5"], "pm10": current["pm10"], "co": current["co"],
        "no2": current["no2"], "so2": current["so2"], "o3": current["o3"],
        "aqi": current["aqi"], "dominant_pollutant": current["dominant_pollutant"],
        "aqi_24h_ago": aqi_24h_ago, "aqi_48h_ago": aqi_48h_ago, "aqi_72h_ago": aqi_72h_ago,
        "aqi_change_24h": current["aqi"] - aqi_24h_ago,
        "aqi_rolling_mean_24h": rolling_mean,
        "aqi_rolling_std_24h": rolling_std,
    }
    return pd.DataFrame([row]), now_ts, gaps

if __name__ == "__main__":
    features_df, as_of_timestamp, gaps = get_live_input()
    print(f"\nLive input built, based on data as of: {datetime.fromtimestamp(as_of_timestamp)}")
    print(features_df.T)
