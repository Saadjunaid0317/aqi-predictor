import os
import csv
import joblib

from live_features import get_live_input

MODEL_DIRS = {"24h": "model_24h", "48h": "model_48h", "72h": "model_72h"}
FEATURE_COLUMNS = [
    "hour", "day", "month", "day_of_week",
    "temp", "humidity", "wind_speed",
    "pm2_5", "pm10", "co", "no2", "so2", "o3",
    "aqi", "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
    "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
]
PREDICTIONS_LOG_FILE = "predictions_log.csv"
RETENTION_DAYS = 14   # matches recent_history.csv's window, so every logged
                       # prediction stays verifiable against it while it's still around


def load_models():
    models = {}
    for horizon, model_dir in MODEL_DIRS.items():
        path = os.path.join(model_dir, "model.pkl")
        if os.path.exists(path):
            models[horizon] = joblib.load(path)
    return models


def append_predictions(rows):
    file_exists = os.path.exists(PREDICTIONS_LOG_FILE)
    with open(PREDICTIONS_LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    # Trim to the retention window, same pattern as recent_history.csv
    cutoff = rows[0]["made_at"] - (RETENTION_DAYS * 24 * 3600)
    with open(PREDICTIONS_LOG_FILE, "r") as f:
        all_rows = list(csv.DictReader(f))
    kept = [r for r in all_rows if int(r["made_at"]) >= cutoff]
    with open(PREDICTIONS_LOG_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(kept)


def main():
    features_df, as_of_ts, _ = get_live_input()
    models = load_models()

    rows = []
    for horizon, model in models.items():
        hours_ahead = int(horizon[:-1])
        predicted_aqi = float(model.predict(features_df[FEATURE_COLUMNS])[0])
        rows.append({
            "made_at": as_of_ts,
            "horizon": horizon,
            "target_timestamp": as_of_ts + hours_ahead * 3600,
            "predicted_aqi": round(predicted_aqi, 2),
        })

    if rows:
        append_predictions(rows)
        print(f"Logged {len(rows)} predictions (24h/48h/72h) for verification later.")
    else:
        print("No models found - nothing logged.")


if __name__ == "__main__":
    main()
