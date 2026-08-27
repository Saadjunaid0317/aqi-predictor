import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from prepare_training_data import load_training_data

tf.random.set_seed(42)

X_train, X_test, targets = load_training_data()

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {name:20s} RMSE: {rmse:6.2f}   MAE: {mae:6.2f}   R2: {r2:6.3f}")
    return r2

models = {
    "Ridge": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1),
}

results = {}
for horizon in ["24h", "48h", "72h"]:
    y_train, y_test = targets[horizon]
    print(f"\n=== Day target: {horizon} (average AQI over that day) ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = evaluate(name, y_test, model.predict(X_test))
        results[(horizon, name)] = r2

# --- LSTM (deep learning candidate) ---
# Unlike the models above, an LSTM needs sequence-shaped input (a window of
# past hours, not a single row) and scaled features, so it's built and
# evaluated separately here, then folded into the same results/summary.
WINDOW_SIZE = 24  # hours of past context per prediction

X_full = pd.concat([X_train, X_test])
split_index = len(X_train)

scaler = StandardScaler().fit(X_train)
X_full_scaled = scaler.transform(X_full)

def build_sequences(X_values, y_values, window_size):
    Xs, ys, target_rows = [], [], []
    for i in range(window_size, len(X_values) + 1):
        Xs.append(X_values[i - window_size:i])
        ys.append(y_values[i - 1])
        target_rows.append(i - 1)
    return np.array(Xs), np.array(ys), np.array(target_rows)

def build_lstm(n_features):
    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, n_features)),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

for horizon in ["24h", "48h", "72h"]:
    y_train, y_test = targets[horizon]
    y_full = pd.concat([y_train, y_test]).values

    X_seq, y_seq, target_rows = build_sequences(X_full_scaled, y_full, WINDOW_SIZE)
    is_test = target_rows >= split_index
    X_seq_train, y_seq_train = X_seq[~is_test], y_seq[~is_test]
    X_seq_test, y_seq_test = X_seq[is_test], y_seq[is_test]

    lstm = build_lstm(n_features=X_seq.shape[2])
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lstm.fit(
        X_seq_train, y_seq_train,
        validation_split=0.1, epochs=50, batch_size=32,
        callbacks=[early_stop], verbose=0,
    )

    print(f"\n=== Day target: {horizon} (LSTM) ===")
    r2 = evaluate("LSTM", y_seq_test, lstm.predict(X_seq_test, verbose=0).ravel())
    results[(horizon, "LSTM")] = r2

model_names = list(models.keys()) + ["LSTM"]

print("\n=== Best model per horizon ===")
for horizon in ["24h", "48h", "72h"]:
    best_name = None
    best_r2 = None
    for name in model_names:
        r2 = results[(horizon, name)]
        if best_r2 is None or r2 > best_r2:
            best_r2 = r2
            best_name = name
    print(f"{horizon}: {best_name} (R2 = {best_r2:.3f})")

print("\n=== Persistence baseline check (per mentor's real bar: beat this, not R2>=0.70) ===")
for horizon in ["24h", "48h", "72h"]:
    y_train, y_test = targets[horizon]
    persistence_pred = X_test["aqi"]   # "assume tomorrow's average = today's current reading"
    print(f"\n{horizon} persistence baseline:")
    persistence_r2 = evaluate("Persistence", y_test, persistence_pred)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_r2 = evaluate("Our Ridge model", y_test, ridge.predict(X_test))

    if ridge_r2 > persistence_r2:
        verdict = "BEATS persistence ✅"
    else:
        verdict = "loses to persistence ❌"
    print(f"  -> {verdict}")