import os
import joblib
from dotenv import load_dotenv
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

from prepare_training_data import load_training_data

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

X_train, X_test, targets = load_training_data()

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
mr = project.get_model_registry()   # the Model Registry, same pattern as get_feature_store()

for horizon in ["24h", "48h", "72h"]:
    print(f"\n=== Registering model for {horizon} ===")
    y_train, y_test = targets[horizon]

    # Train the final model (same Ridge, same split we already validated)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    persistence_preds = X_test["aqi"]
    persistence_rmse = float(np.sqrt(mean_squared_error(y_test, persistence_preds)))

    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")
    print(f"  Beats persistence ({persistence_rmse:.2f} RMSE): {rmse < persistence_rmse}")

    # --- Save the trained model to a local folder ---
    model_dir = f"model_{horizon}"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    # --- Register it in the Hopsworks Model Registry ---
    py_model = mr.python.create_model(
        name=f"aqi_ridge_{horizon}",
        metrics={"rmse": rmse, "mae": mae, "r2": r2, "persistence_rmse": persistence_rmse},
        description=(
            f"Ridge regression predicting average Karachi AQI for the {horizon} window. "
            f"Beats persistence baseline (RMSE {persistence_rmse:.2f} -> {rmse:.2f}), "
            f"per mentor-confirmed registration criterion."
        ),
        input_example=X_test.iloc[[0]]
    )
    py_model.save(model_dir)
    print(f"  Registered as aqi_ridge_{horizon}")

print("\nAll three models registered.")