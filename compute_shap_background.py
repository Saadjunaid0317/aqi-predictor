import numpy as np
import pandas as pd

from prepare_training_data import load_training_data

BACKGROUND_FILE = "shap_background.csv"
N_SAMPLES = 150

X_train, X_test, targets = load_training_data()

idx = np.linspace(0, len(X_train) - 1, N_SAMPLES, dtype=int)
background = X_train.iloc[idx].reset_index(drop=True)
background.to_csv(BACKGROUND_FILE, index=False)

print(f"Saved {len(background)} background rows (evenly spaced across the training period) to {BACKGROUND_FILE}")
