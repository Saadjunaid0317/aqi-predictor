import pandas as pd
from datetime import datetime
from feature_pipeline import build_feature_record

HISTORY_FILE = "recent_history.csv"

def get_live_input():
    current = build_feature_record()

    history = pd.read_csv(HISTORY_FILE)
    now_ts = current["timestamp"]
    timestamp_to_aqi = dict(zip(history["timestamp"], history["aqi"]))

    def find_nearest_aqi(target_ts):
        closest_ts = min(history["timestamp"], key=lambda t: abs(t - target_ts))
        gap_hours = abs(closest_ts - target_ts) / 3600
        return timestamp_to_aqi[closest_ts], gap_hours

    aqi_24h_ago, gap24 = find_nearest_aqi(now_ts - 24 * 3600)
    aqi_48h_ago, gap48 = find_nearest_aqi(now_ts - 48 * 3600)
    aqi_72h_ago, gap72 = find_nearest_aqi(now_ts - 72 * 3600)
    print(f"Lookup gaps (hours off from exact target): 24h->{gap24:.1f}, 48h->{gap48:.1f}, 72h->{gap72:.1f}")

    recent_24h = history[history["timestamp"] >= now_ts - 24 * 3600]
    rolling_mean = recent_24h["aqi"].mean() if len(recent_24h) > 0 else current["aqi"]
    rolling_std = recent_24h["aqi"].std() if len(recent_24h) > 1 else 0.0

    row = {
        "hour": current["hour"], "day": current["day"], "month": current["month"],
        "day_of_week": current["day_of_week"], "temp": current["temp"],
        "humidity": current["humidity"], "wind_speed": current["wind_speed"],
        "pm2_5": current["pm2_5"], "pm10": current["pm10"], "co": current["co"],
        "no2": current["no2"], "so2": current["so2"], "o3": current["o3"],
        "aqi": current["aqi"],
        "aqi_24h_ago": aqi_24h_ago, "aqi_48h_ago": aqi_48h_ago, "aqi_72h_ago": aqi_72h_ago,
        "aqi_change_24h": current["aqi"] - aqi_24h_ago,
        "aqi_rolling_mean_24h": rolling_mean,
        "aqi_rolling_std_24h": rolling_std,
    }
    return pd.DataFrame([row]), now_ts

if __name__ == "__main__":
    features_df, as_of_timestamp = get_live_input()
    print(f"\nLive input built, based on data as of: {datetime.fromtimestamp(as_of_timestamp)}")
    print(features_df.T)
