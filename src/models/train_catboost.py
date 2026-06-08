# ==========================================
# CATBOOST TRAINING
# ==========================================

import os
import joblib

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from src.utils.helpers import (
    load_training_data
)

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

X, y = load_training_data()

# ------------------------------------------
# REMOVE F2230
# ------------------------------------------

if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

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
# CATBOOST MODEL
# ------------------------------------------

print("\nTraining CatBoost...")

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=False
)

model.fit(
    X_train,
    y_train
)

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
    "models/saved_models/catboost.pkl"
)

print("\nCatBoost Model Saved")