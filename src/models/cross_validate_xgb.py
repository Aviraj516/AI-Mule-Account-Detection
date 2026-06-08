import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from xgboost import XGBClassifier

from src.utils.helpers import (
    load_training_data
)

# ------------------------------------------
# LOAD DATA
# ------------------------------------------

X, y = load_training_data()

# ------------------------------------------
# REMOVE F2230
# ------------------------------------------

if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

# ------------------------------------------
# XGBOOST MODEL
# ------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

# ------------------------------------------
# CROSS VALIDATION
# ------------------------------------------

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

# ------------------------------------------
# RESULTS
# ------------------------------------------

print("\nRecall Scores:")
print(scores)

print("\nAverage Recall:")
print(np.mean(scores))