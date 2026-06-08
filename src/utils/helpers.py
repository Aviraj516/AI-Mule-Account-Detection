# ==========================================
# HELPER FUNCTIONS
# ==========================================

import pandas as pd

from src.utils.config import (
    DATA_PATH,
    FEATURE_PATH,
    TARGET_COLUMN
)

def load_training_data():

    df = pd.read_csv(DATA_PATH)

    feature_df = pd.read_csv(FEATURE_PATH)

    selected_features = (
        feature_df["Selected_Features"]
        .tolist()
    )

    if "F2230" in selected_features:
        selected_features.remove("F2230")

    X = df[selected_features]

    y = df[TARGET_COLUMN]

    for col in X.columns:

        if X[col].dtype == "object":

            X[col] = (
                X[col]
                .astype("category")
                .cat.codes
            )

    return X, y