# Pearls AQI Predictor — Full Progress Cheat Sheet (v6 — Final Build-Out)

**Project:** 10Pearls SHINE Internship, Data Sciences track
**Goal:** A serverless system forecasting Karachi's AQI 24h/48h/72h ahead — data collection, storage, training, automation, model registry, and a dashboard.

---

## 1. Architecture & Status

```
Weather & Pollution API → Feature Pipeline ──┐
                                              ▼
                          Training Pipeline ↔ Feature Store & Model Registry
                                              ▲
                                    Web App (dashboard) ──┘
```

| Piece | Status |
|---|---|
| Live feature pipeline (OpenWeather) | ✅ Working |
| Feature Store (Hopsworks) | ✅ Working |
| Historical backfill (Open-Meteo, 3 years) | ✅ Done — 26,304 records |
| Automated hourly collection (GitHub Actions) | ✅ Done — 300+ successful runs |
| Training data prep (per-day-average targets) | ✅ Done |
| Model training & evaluation | ✅ Done — beats persistence at all 3 horizons |
| Model Registry | ✅ Done — 3 models registered |
| Live serving (recent lookback) | ✅ Just fixed — local CSV cache, bypasses a Hopsworks recency bug |
| Web app dashboard | 🔶 **Next step** |
| Daily training automation | Deferred — decide after web app, time permitting |
| EDA / SHAP / alerts / multi-city | Not started — stretch goals if time allows |
| Final report | Not started |

**Deadline:** Sept 4, 2026.

---

## 2. Environment, Keys, GitHub — Done ✅ (unchanged since earlier versions)

- Conda env `aqi-predictor` (Python 3.11), VS Code, GitHub repo `Saadjunaid0317/aqi-predictor`.
- `.env` (gitignored) holds `OPENWEATHER_API_KEY` and `HOPSWORKS_API_KEY`; GitHub Secrets hold the same two for automation.
- GitHub Actions workflow `feature_pipeline.yml` runs `push_to_hopsworks.py` every hour (cron `0 * * * *`), now also commits `recent_history.csv` back to the repo each run (see Section 5).
- Full Windows troubleshooting history (conda PATH, twofish, pyarrow, `/tmp`, Hudi vs Delta, confluent-kafka) is in the earlier cheat sheet versions if you need to revisit any of it — not repeated here to keep this version focused on what's new.

---

## 3. The Core Data & AQI Logic — `feature_pipeline.py` (unchanged foundation)

Still the base of everything: fetches live OpenWeather data, computes real EPA-formula AQI from PM2.5/PM10 (`calculate_sub_index` / `calculate_aqi`), extracts time features, and returns one clean `build_feature_record()` dictionary. Every other script in this project either calls this directly or reuses its output shape. Full code is in the earlier cheat sheet version.

---

## 4. Training Data — the BIG structural change: per-day-average targets

**Why this changed:** Ma'am Umema (mentor) and cohort discussion clarified the real target structure — **3 separate models**, one per horizon (24h/48h/72h), each predicting the **average AQI over that day's window** (Day 1 = hours 1-24, Day 2 = hours 25-48, Day 3 = hours 49-72), not a single instantaneous point. This is a big jump from what we originally built (a single point-prediction at exactly 72h).

`prepare_training_data.py` — final structure:

```python
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
    # Reading ~26,000 rows in ONE request was unreliable (Hopsworks EU-West
    # instability - see Section 7). Small time-windowed requests are reliable.
    start_ts = int(datetime.fromisoformat(start_date).timestamp())
    end_ts = int(datetime.fromisoformat(end_date).timestamp())
    chunk_seconds = chunk_days * 86400
    chunks = []
    current_start = start_ts
    while current_start < end_ts:
        current_end = min(current_start + chunk_seconds, end_ts)
        query = fg.filter((fg.timestamp >= current_start) & (fg.timestamp < current_end))
        chunk_df = query.read()
        chunks.append(chunk_df)
        current_start = current_end
    return pd.concat(chunks, ignore_index=True)

def load_training_data():
    df = read_feature_group_in_chunks(fg, "2023-08-11", "2026-08-16", chunk_days=60)
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamp_to_aqi = dict(zip(df["timestamp"], df["aqi"]))

    def get_window_average(row, start_hours, end_hours, min_coverage=0.5):
        # Average AQI over a future window (e.g. hours 1-24 for "Day 1").
        # If more than half the hours are missing, skip rather than average
        # a handful of values and call it representative of a full day.
        values = []
        for h in range(start_hours, end_hours + 1):
            ts = row["timestamp"] + h * 3600
            if ts in timestamp_to_aqi:
                values.append(timestamp_to_aqi[ts])
        expected = end_hours - start_hours + 1
        if len(values) < expected * min_coverage:
            return None
        return sum(values) / len(values)

    df["target_aqi_24h"] = df.apply(lambda row: get_window_average(row, 1, 24), axis=1)
    df["target_aqi_48h"] = df.apply(lambda row: get_window_average(row, 25, 48), axis=1)
    df["target_aqi_72h"] = df.apply(lambda row: get_window_average(row, 49, 72), axis=1)

    # Lag/trend features - what AQI was at points in the PAST
    def make_lag_feature(hours_ago):
        seconds_ago = hours_ago * 3600
        return df.apply(lambda row: timestamp_to_aqi.get(row["timestamp"] - seconds_ago), axis=1)

    df["aqi_24h_ago"] = make_lag_feature(24)
    df["aqi_48h_ago"] = make_lag_feature(48)
    df["aqi_72h_ago"] = make_lag_feature(72)
    df["aqi_change_24h"] = df["aqi"] - df["aqi_24h_ago"]

    # Rolling mean/std over the trailing 24h (TIME-based window, robust to gaps)
    df["_datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("_datetime")
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling("24h").mean()
    df["aqi_rolling_std_24h"] = df["aqi"].rolling("24h").std()
    df = df.reset_index(drop=True)

    required = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h",
                "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
                "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h"]
    df = df.dropna(subset=required)

    feature_columns = [
        "hour", "day", "month", "day_of_week", "temp", "humidity", "wind_speed",
        "pm2_5", "pm10", "co", "no2", "so2", "o3", "aqi",
        "aqi_24h_ago", "aqi_48h_ago", "aqi_72h_ago",
        "aqi_change_24h", "aqi_rolling_mean_24h", "aqi_rolling_std_24h"
    ]
    X = df[feature_columns]
    split_index = int(len(df) * 0.8)   # CHRONOLOGICAL split - train on older data,
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]  # test on newest

    targets = {}
    for horizon in ["24h", "48h", "72h"]:
        y = df[f"target_aqi_{horizon}"]
        targets[horizon] = (y.iloc[:split_index], y.iloc[split_index:])

    return X_train, X_test, targets
```

**Result:** 26,172 usable rows, split 20,937 train / 5,235 test (chronological, to avoid data leakage).

---

## 5. Model Training & The Real Evaluation Standard

### The metrics, in plain terms
- **MAE** — average absolute error, in AQI points.
- **RMSE** — like MAE but punishes big misses harder; RMSE ≥ MAE always.
- **R²** — how much better than "just guess the average" the model is. Mechanically capped low when the test period has low natural variance ("calm" AQI), *regardless* of model quality.

### The critical reframe (from mentor, via cohort discussion)
The **real registration bar is beating the persistence baseline** ("predict tomorrow = today"), *not* an absolute R² ≥ 0.70. Low R² in a calm test window is explicitly acceptable — this matched our own diagnosis of low test-set variance.

### Final results (Ridge Regression won at every horizon):

| Horizon | Our RMSE | Our MAE | Our R² | Persistence RMSE | Persistence R² | Verdict |
|---|---|---|---|---|---|---|
| 24h | 11.49 | 8.26 | 0.569 | 16.76 | 0.083 | **Beats persistence ✅** |
| 48h | 14.48 | 10.78 | 0.285 | 21.44 | -0.567 | **Beats persistence ✅** |
| 72h | 15.02 | 11.18 | 0.176 | 23.12 | -0.952 | **Beats persistence ✅** |

Ridge beat Random Forest, Gradient Boosting, and XGBoost at every horizon — the tree models went **negative R² at 72h** (overfitting on the noisier long-horizon target). This pattern — Day 1 much better than Day 2/3, simplest model winning — matched what nearly the entire cohort reported.

`train_model.py` core evaluation function:
```python
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:20s} RMSE: {rmse:6.2f}   MAE: {mae:6.2f}   R2: {r2:6.3f}")
    return r2
```
Persistence baseline used for comparison: `X_test["aqi"]` (today's current reading) evaluated against each horizon's target.

---

## 6. Model Registry — `register_models.py`

```python
import joblib
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
mr = project.get_model_registry()   # same handshake pattern as get_feature_store()

for horizon in ["24h", "48h", "72h"]:
    y_train, y_test = targets[horizon]
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    # ... compute rmse, mae, r2, persistence_rmse ...

    model_dir = f"model_{horizon}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))   # serialize with joblib (ships with sklearn)

    py_model = mr.python.create_model(
        name=f"aqi_ridge_{horizon}",
        metrics={"rmse": rmse, "mae": mae, "r2": r2, "persistence_rmse": persistence_rmse},
        description=f"Ridge regression for {horizon} AQI forecast. Beats persistence.",
        input_example=X_test.iloc[[0]]
    )
    py_model.save(model_dir)
```

**Confirmed working** — `aqi_ridge_24h`, `aqi_ridge_48h`, `aqi_ridge_72h` all registered, each with real metrics attached, viewable on the Hopsworks dashboard.

---

## 7. Live Serving — the Hopsworks recency bug, and the fix

**Bug found:** despite 300+ successful hourly automation runs, querying Hopsworks for anything from roughly the last 1-2 weeks reliably returned **0 rows** — confirmed with multiple query methods including the same chunked function that works perfectly for older historical data. Root cause (best understanding, not 100% confirmed): Hudi (Hopsworks' storage engine) needs to "materialize" newly inserted small hourly rows into efficiently queryable files; that background compaction appears to be lagging significantly behind on the free tier.

**The fix — bypass Hopsworks entirely for recent lookback data.** The hourly script already computes each record in memory before sending it to Hopsworks — it can just also keep its own tiny local log of what it just sent, sidestepping the need to query Hopsworks for anything recent at all.

**`push_to_hopsworks.py` addition:**
```python
import csv

HISTORY_FILE = "recent_history.csv"

def append_to_local_history(record):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

    # Keep only the last 14 days so the file never grows unbounded
    cutoff = record["timestamp"] - (14 * 24 * 3600)
    with open(HISTORY_FILE, "r") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if int(r["timestamp"]) >= cutoff]
    with open(HISTORY_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        writer.writeheader()
        writer.writerows(rows)
```
Called right after `fg.insert(df)` in the same script.

**GitHub Actions workflow addition** (commits the updated CSV back to the repo each run):
```yaml
      - name: Commit updated recent history
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add recent_history.csv
          git diff --staged --quiet || git commit -m "Update recent history [automated]"
          git push
```

**`live_features.py` — final version, reads the local CSV instead of querying Hopsworks:**
```python
import pandas as pd
from datetime import datetime
from feature_pipeline import build_feature_record

HISTORY_FILE = "recent_history.csv"

def get_live_input():
    current = build_feature_record()   # right now, straight from OpenWeather
    history = pd.read_csv(HISTORY_FILE)
    now_ts = current["timestamp"]
    timestamp_to_aqi = dict(zip(history["timestamp"], history["aqi"]))

    def find_nearest_aqi(target_ts):
        # Find the CLOSEST available timestamp rather than requiring an exact
        # match - robust against any gaps in the local history.
        closest_ts = min(history["timestamp"], key=lambda t: abs(t - target_ts))
        gap_hours = abs(closest_ts - target_ts) / 3600
        return timestamp_to_aqi[closest_ts], gap_hours

    aqi_24h_ago, _ = find_nearest_aqi(now_ts - 24 * 3600)
    aqi_48h_ago, _ = find_nearest_aqi(now_ts - 48 * 3600)
    aqi_72h_ago, _ = find_nearest_aqi(now_ts - 72 * 3600)

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
```

**Important note for the report:** this CSV only starts accumulating from the moment it was added — full 72h accuracy needs 3 days to build up. Hopsworks remains the actual required feature store for training; this local file exists purely to serve live predictions reliably, working around a platform-side query limitation.

---

## 8. Key Concepts Learned (this stage)

- **Persistence baseline as the real evaluation bar** — "beat predicting no change" matters more than hitting an absolute R² number, especially in a low-variance test period.
- **Day-averaged, per-horizon targets** — matches how AQI forecasts are conventionally reported, and is simpler + more learnable than one instantaneous 72h-ahead point.
- **Chronological + per-horizon Direct modeling** — 3 independent models, not recursive chaining (which a classmate found makes error compound).
- **Materialization lag** — newly-inserted data in a Hudi-backed store isn't necessarily *immediately* queryable; a real operational gotcha distinct from "the insert failed."
- **Nearest-timestamp lookup** — a robust pattern for time-series data with gaps: find the closest match instead of demanding an exact one, and report how far off it was.
- **Bypassing a slow/unreliable dependency with a lightweight local cache** — a generally useful engineering pattern, not just specific to this project.

---

## 9. Decisions & Why (this stage)

| Decision | Reasoning |
|---|---|
| Per-day-average, 3-separate-model targets | Matches mentor/cohort-confirmed correct structure |
| Direct strategy (not recursive chaining) | A classmate found chaining made Day 2/3 worse |
| Stopped tuning once Ridge beat persistence at all horizons | Time-smart given the real registration criterion, protects time for the remaining pipeline |
| Skepticism toward a friend's much-better (SVR/CatBoost) results | RMSE/R² combination was mathematically inconsistent with the claimed data's variance; likely a leakage or non-comparable-dataset issue, not a technique to copy blindly |
| Local `recent_history.csv` cache for live serving | Hopsworks remains the true feature store; this only works around a confirmed recent-query reliability issue for the live app specifically |

---

## 10. What's Next

1. **Build the web app dashboard** (Streamlit) — load the 3 registered models + `live_features.py`, show a 3-day AQI forecast.
2. Decide on daily training automation given remaining time.
3. Stretch goals if time allows: EDA writeup, SHAP explainability, hazardous-AQI alerts, multi-city support.
4. Final report — including the model comparison, persistence-baseline reasoning, and known limitations (materialization lag workaround, approximate early lookback data).
