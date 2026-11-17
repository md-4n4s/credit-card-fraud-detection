from sklearn.ensemble import RandomForestClassifier

from modeling.logistic_regression.feature_selection import (
    try_keep_features,
    try_test_features,
)
from modeling.random_forest.model_setup import *

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
)

if __name__ == "__main__":
    try_test_features(
        keep_features=keep,
        selected_model=model,
        model_name="RandomForest",
        scaling="None",
        max_iter=100,
    )
    try_keep_features(
        keep_features=keep,
        selected_model=model,
        model_name="RandomForest",
        scaling="None",
        max_iter=100,
    )
