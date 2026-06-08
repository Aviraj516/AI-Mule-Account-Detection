import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)

from catboost import CatBoostClassifier

from src.utils.helpers import (
    load_training_data
)

X, y = load_training_data()

if "F2230" in X.columns:
    X = X.drop(columns=["F2230"])

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=False
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