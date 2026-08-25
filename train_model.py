import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from prepare_training_data import load_training_data

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

print("\n=== Best model per horizon ===")
for horizon in ["24h", "48h", "72h"]:
    best_name = max(models.keys(), key=lambda n: results[(horizon, n)])
    print(f"{horizon}: {best_name} (R2 = {results[(horizon, best_name)]:.3f})")

print("\n=== Persistence baseline check (per mentor's real bar: beat this, not R2>=0.70) ===")
for horizon in ["24h", "48h", "72h"]:
    y_train, y_test = targets[horizon]
    persistence_pred = X_test["aqi"]   # "assume tomorrow's average = today's current reading"
    print(f"\n{horizon} persistence baseline:")
    persistence_r2 = evaluate("Persistence", y_test, persistence_pred)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_r2 = evaluate("Our Ridge model", y_test, ridge.predict(X_test))

    verdict = "BEATS persistence ✅" if ridge_r2 > persistence_r2 else "loses to persistence ❌"
    print(f"  -> {verdict}")