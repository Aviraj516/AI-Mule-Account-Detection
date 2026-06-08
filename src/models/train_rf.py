# ==========================================
# RANDOM FOREST TRAINING
# ==========================================

import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from src.utils.helpers import (
    load_training_data
)

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X, y = load_training_data()

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# ------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# ------------------------------------------
# MODEL
# ------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------
# PREDICTIONS
# ------------------------------------------

predictions = model.predict(X_test)

# ------------------------------------------
# RESULTS
# ------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions
    )
)

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------

os.makedirs(
    "models/saved_models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/saved_models/random_forest.pkl"
)

print("\nRandom Forest Model Saved")