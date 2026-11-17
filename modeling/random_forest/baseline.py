from sklearn.ensemble import RandomForestClassifier

from modeling.logistic_regression.baseline import (
    report_cross_validation,
    report_keep_features,
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
    report_cross_validation(
        model, model_name="RandomForest", scaling="None", max_iter=100, features=keep
    )
    report_keep_features(
        selected_model=model,
        features=keep,
        model_name="RandomForest",
        scaling="None",
        max_iter=100,
    )
