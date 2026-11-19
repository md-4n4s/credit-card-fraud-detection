from modeling.logistic_regression.baseline import (
    report_cross_validation,
    report_keep_features,
)
from modeling.SVM.model_setup import *

if __name__ == "__main__":
    report = report_cross_validation(
        model, model_name="SVM", scaling="StandardScaler", max_iter=None, features=keep
    )
    report_keep_features(
        selected_model=model,
        features=keep,
        model_name="SVM",
        scaling="StandardScaler",
        max_iter=None,
        roc_auc=False,
    )
