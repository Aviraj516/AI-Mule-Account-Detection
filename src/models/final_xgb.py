# ==========================================
# FINAL XGBOOST TRAINING
# ==========================================

import os
import joblib

from xgboost import XGBClassifier

from src.utils.helpers import (
    load_training_data
)

print("=" * 60)
print("LOADING FULL DATASET")
print("=" * 60)

X, y = load_training_data()

# ------------------------------------------
# REMOVE F2230
# ------------------------------------------

if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

print("Dataset Shape:", X.shape)

# ------------------------------------------
# FINAL MODEL
# ------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

print("\nTraining Final XGBoost Model...")

model.fit(X, y)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

os.makedirs(
    "models/saved_models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/saved_models/final_xgboost.pkl"
)

print("\nFinal XGBoost Saved Successfully")