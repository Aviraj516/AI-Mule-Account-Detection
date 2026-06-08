# ==========================================
# SHAP ANALYSIS FOR XGBOOST
# ==========================================

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.helpers import load_training_data

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------

X, y = load_training_data()

if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

print("Dataset Shape:", X.shape)

# ------------------------------------------
# LOAD MODEL
# ------------------------------------------

model = joblib.load(
    "models/saved_models/final_xgboost.pkl"
)

print("Model Loaded")

# ------------------------------------------
# SAMPLE DATA
# ------------------------------------------

X_sample = X.sample(
    n=min(500, len(X)),
    random_state=42
)

# ------------------------------------------
# TREE EXPLAINER
# ------------------------------------------

print("\nCreating SHAP Explainer...")

# Use underlying booster
explainer = shap.TreeExplainer(
    model.get_booster()
)

print("Generating SHAP Values...")

shap_values = explainer.shap_values(
    X_sample
)

# ------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------

os.makedirs(
    "reports/shap",
    exist_ok=True
)

# ------------------------------------------
# SUMMARY PLOT
# ------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.savefig(
    "reports/shap/shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("SHAP Summary Plot Saved")

# ------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "SHAP_Importance":
        abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    by="SHAP_Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 20 SHAP FEATURES")
print("=" * 60)

print(
    importance.head(20)
)

importance.to_csv(
    "reports/shap/shap_feature_importance.csv",
    index=False
)

# ------------------------------------------
# WATERFALL EXPLANATION
# ------------------------------------------

try:

    explanation = explainer(
        X_sample.iloc[:1]
    )

    plt.figure()

    shap.plots.waterfall(
        explanation[0],
        show=False
    )

    plt.savefig(
        "reports/shap/sample_waterfall.png",
        bbox_inches="tight"
    )

    plt.close()

    print("Waterfall Plot Saved")

except Exception as e:

    print(
        "Waterfall Plot Failed:",
        e
    )

print("\nSHAP ANALYSIS COMPLETE")

print("\nFiles Generated:")
print("reports/shap/shap_summary.png")
print("reports/shap/shap_feature_importance.csv")
print("reports/shap/sample_waterfall.png")


"""# SHAP Analysis Summary

## Objective

After training the final XGBoost model, SHAP (SHapley Additive Explanations) analysis was performed to understand how the model makes decisions while identifying mule accounts.

While the model can accurately predict whether an account is suspicious or legitimate, SHAP helps explain the reasons behind those predictions. This increases transparency and trust in the AI system.

---

## Why SHAP Was Used

Machine learning models often act like a "black box", meaning they provide predictions without explaining how those predictions were made.

SHAP solves this problem by identifying which features have the greatest influence on the model's decisions.

This allows us to answer questions such as:

* Why was a particular account classified as a mule account?
* Which features contribute most to suspicious behavior?
* Which features have the strongest impact on the model's predictions?

---

## SHAP Analysis Results

The SHAP analysis identified the most influential features used by the XGBoost model for mule account detection.

Top Important Features:

1. F3912
2. F3898
3. F162
4. F1597
5. F3811
6. F3484
7. F3805
8. F1165
9. F3800
10. F2686

Among all features, F3912 had the highest impact on the model's decisions, making it the strongest indicator of mule account behavior within the available dataset.

---

## Interpretation

The SHAP results show that the model does not rely on a single feature to make decisions. Instead, it analyzes multiple behavioral and transactional patterns simultaneously before classifying an account.

Certain features contribute positively toward identifying suspicious activity, while others reduce the likelihood of an account being classified as a mule account.

This demonstrates that the model is learning meaningful fraud patterns rather than making random predictions.

---

## Importance of SHAP in Banking

In real-world banking environments, simply predicting that an account is suspicious is not enough. Investigators and compliance teams need to understand why the account was flagged.

SHAP provides this explainability by highlighting the factors that influenced the model's decision.

This helps:

* Improve trust in AI-generated alerts.
* Support fraud investigation teams.
* Provide transparency for compliance and auditing purposes.
* Assist banks in understanding suspicious account behavior.

---

## Conclusion

SHAP analysis successfully explained the decision-making process of the final XGBoost model. The analysis identified the most influential features contributing to mule account detection, with F3912 emerging as the strongest indicator. By providing interpretable explanations alongside predictions, SHAP improves the transparency, reliability, and practical usability of the mule account detection system.
"""