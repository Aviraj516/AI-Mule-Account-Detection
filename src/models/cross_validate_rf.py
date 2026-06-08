import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier

from src.utils.helpers import (
    load_training_data
)

X, y = load_training_data()

# remove F2230 again
if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="recall"
)

print("\nRecall Scores:")
print(scores)

print("\nAverage Recall:")
print(np.mean(scores))

"""# Step 5: Random Forest Model Training

After selecting the final 43 important features, we trained a Random Forest classifier.

## What is Random Forest?

Random Forest is a machine learning algorithm made up of many decision trees.

Each decision tree independently analyzes the account data and predicts whether an account is a mule account or a legitimate account.

The final prediction is determined by combining the votes of all decision trees.

Simple Analogy:

Imagine 300 fraud investigators independently examining the same account. Each investigator gives an opinion, and the final decision is made based on majority voting.

---

# Initial Training Results

Confusion Matrix:

[[1801    0]
[   0   16]]

Results:

* Accuracy = 100%
* Precision = 100%
* Recall = 100%
* F1 Score = 100%

At first, these results appeared excellent because the model correctly classified all normal and mule accounts.

However, such perfect results are very uncommon in real-world fraud detection problems.

Therefore, further investigation was performed to verify whether the model was genuinely learning fraud patterns or whether a hidden issue existed in the data.

---

# Problem Discovered: Data Leakage

During feature investigation, feature F2230 was analyzed.

Values present in F2230:

* Oct25
* Sep25
* Nov25
* Dec25

Analysis showed:

* Oct25 contained only normal accounts.
* Sep25 contained only mule accounts.
* Nov25 contained only mule accounts.
* Dec25 contained only mule accounts.

This meant that the feature was indirectly revealing the target class.

The model was not fully learning fraud behaviour. Instead, it could simply use F2230 to determine the account type.

This problem is known as Data Leakage.

---

# What is Data Leakage?

Data Leakage occurs when one or more features contain information that directly or indirectly reveals the correct answer.

As a result, the model appears extremely accurate during testing but may fail when used on real-world unseen data.

Because of this, leaked features must be removed before final model development.

---

# Solution Applied

Feature F2230 was removed from the training dataset.

The Random Forest model was then retrained using the remaining 42 features.

---

# Results After Removing F2230

Confusion Matrix:

[[1801    0]
[   1   15]]

Results:

* Precision = 100%
* Recall = 94%
* F1 Score = 97%

Interpretation:

The model correctly detected 15 out of 16 mule accounts and missed only 1 mule account.

Although performance decreased slightly, the results became much more realistic and trustworthy.

This demonstrated that the remaining features still contained meaningful fraud-related information.

---

# Cross Validation

To ensure that the model performance was not dependent on a single train-test split, 5-Fold Stratified Cross Validation was performed.

Recall Scores:

* 0.75
* 0.94
* 0.75
* 0.75
* 0.81

Average Recall:

80%

---

# Why Cross Validation Was Performed

A single train-test split may sometimes produce overly optimistic results.

Cross Validation repeatedly trains and evaluates the model using different portions of the dataset.

This provides a more reliable estimate of real-world performance.

---

# Understanding Evaluation Metrics

## Precision

Precision measures how many accounts predicted as mule accounts were actually mule accounts.

Formula:

Precision = True Positives / (True Positives + False Positives)

High Precision means fewer false alarms.

---

## Recall

Recall measures how many actual mule accounts were successfully detected.

Formula:

Recall = True Positives / (True Positives + False Negatives)

Example:

If there are 100 mule accounts and the model correctly identifies 80 of them:

Recall = 80%

For fraud detection systems, Recall is one of the most important performance metrics because missing fraudulent accounts can result in financial losses.

---

## F1 Score

F1 Score provides a balance between Precision and Recall.

It combines both metrics into a single performance measure.

A high F1 Score indicates that the model achieves both strong fraud detection capability and low false alarm rates.

---

# Conclusion

The Random Forest model was successfully trained and evaluated for mule account detection.

During evaluation, a data leakage issue involving feature F2230 was identified and resolved. After removing the leaked feature, the model achieved a more realistic performance with a cross-validated recall of approximately 80%.

This indicates that the model is capable of detecting the majority of mule accounts while maintaining very high precision and forms a strong baseline for comparison with XGBoost and CatBoost models.
"""