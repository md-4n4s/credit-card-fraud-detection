from modeling.KNN.model_setup import *
from modeling.logistic_regression.baseline import (
    report_cross_validation,
    report_keep_features,
)

if __name__ == "__main__":
    report = report_cross_validation(
        model, model_name="KNN", scaling="StandardScaler", max_iter=5, features=keep
    )
    report_keep_features(
        selected_model=model,
        features=keep,
        model_name="KNN",
        scaling="StandardScaler",
        max_iter=5,
    )
