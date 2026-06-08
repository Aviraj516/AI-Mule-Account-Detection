# ==========================================
# PREDICT REAL ACCOUNT + SHAP EXPLANATION
# ==========================================

import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.llm.generate_report import generate_report

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------

print("\nLoading Dataset...")

df = pd.read_csv(
    "data/processed/preprocessed_data.csv"
)

# ------------------------------------------
# USER INPUT
# ------------------------------------------

account_number = int(
    input(
        "\nEnter Account Row Number: "
    )
)

# ------------------------------------------
# ACTUAL LABEL
# ------------------------------------------

actual_label = df.iloc[
    account_number
]["F3924"]

# ------------------------------------------
# PREPARE FEATURES
# ------------------------------------------

X = df.drop(
    columns=["F3924"]
)

# Remove leakage feature

if "F2230" in X.columns:
    X = X.drop(
        columns=["F2230"]
    )

# Final 42 Features

final_features = [
    "F1057","F1165","F1381","F1387","F1393",
    "F1489","F1495","F1501","F159","F1597",
    "F1603","F1609","F162","F1707","F1713",
    "F1717","F1719","F1755","F1814","F1815",
    "F1819","F1821","F1825","F1827","F1861",
    "F1863","F2486","F2489","F267","F2686",
    "F270","F3484","F3532","F3640","F3748",
    "F3800","F3801","F3805","F3811","F3898",
    "F3912","F949"
]

X = X[final_features]

# ------------------------------------------
# SELECT ACCOUNT
# ------------------------------------------

account = X.iloc[[account_number]]

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

print("\nLoading Model...")

model = joblib.load(
    "models/saved_models/final_xgboost.pkl"
)

# ------------------------------------------
# PREDICTION
# ------------------------------------------

prediction = model.predict(
    account
)[0]

probability = model.predict_proba(
    account
)[0][1]

risk_score = round(
    probability * 100,
    2
)

# ------------------------------------------
# ALERT LEVEL
# ------------------------------------------

if risk_score >= 80:
    alert = "HIGH"

elif risk_score >= 50:
    alert = "MEDIUM"

else:
    alert = "LOW"

# ------------------------------------------
# LABEL TEXT
# ------------------------------------------

actual_text = (
    "MULE ACCOUNT"
    if actual_label == 1
    else "NORMAL ACCOUNT"
)

prediction_text = (
    "MULE ACCOUNT"
    if prediction == 1
    else "NORMAL ACCOUNT"
)

# ------------------------------------------
# RESULT
# ------------------------------------------

if prediction == actual_label:
    result = "CORRECT"
else:
    result = "INCORRECT"

# ------------------------------------------
# SHAP ANALYSIS
# ------------------------------------------

print("\nGenerating SHAP Explanation...")

explainer = shap.TreeExplainer(
    model.get_booster()
)

shap_values = explainer.shap_values(
    account
)
os.makedirs(
    "reports/account_shap",
    exist_ok=True
)

explanation = explainer(account)

plt.figure()

shap.plots.waterfall(
    explanation[0],
    show=False
)

plt.savefig(
    f"reports/account_shap/account_{account_number}_waterfall.png",
    bbox_inches="tight"
)

plt.close()
feature_importance = pd.DataFrame({
    "Feature": account.columns,
    "SHAP_Value":
        np.abs(
            shap_values[0]
        )
})

feature_importance = (
    feature_importance
    .sort_values(
        by="SHAP_Value",
        ascending=False
    )
)

top_features = feature_importance.head(5)

# ------------------------------------------
# PREPARE FEATURES FOR LLM
# ------------------------------------------

top_features_for_llm = []

for row in feature_importance.head(5).itertuples():

    contribution = shap_values[0][
        list(account.columns).index(
            row.Feature
        )
    ]

    top_features_for_llm.append(
        (
            row.Feature,
            contribution
        )
    )
# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\n" + "=" * 60)
print("ACCOUNT ANALYSIS")
print("=" * 60)

print(f"Account Row   : {account_number}")

print(f"\nActual Label  : {actual_text}")

print(f"Prediction    : {prediction_text}")

print(f"Risk Score    : {risk_score}%")

print(f"Alert Level   : {alert}")

print(f"Result        : {result}")

print("\nTop 5 SHAP Feature Contributions")

for i, row in enumerate(
    top_features.itertuples(),
    start=1
):

    contribution = shap_values[0][
        list(account.columns).index(row.Feature)
    ]

    direction = (
        "INCREASED MULE RISK"
        if contribution > 0
        else "DECREASED MULE RISK"
    )

    print(
        f"{i}. {row.Feature}"
    )

    print(
        f"   SHAP Value : {contribution:.4f}"
    )

    print(
        f"   Effect     : {direction}"
    )


print("=" * 60)

# ------------------------------------------
# LLM EXPLANATION
# ------------------------------------------
print("\nStarting LLM...")

llm_summary = generate_report(
    prediction_text=prediction_text,
    risk_score=risk_score,
    alert_level=alert,
    top_features=top_features_for_llm,
    model_name="phi3"
)
print("LLM Finished")
print("\n")
print("=" * 60)
print("AI FRAUD ANALYST EXPLANATION")
print("=" * 60)

print(llm_summary)

print("=" * 60)