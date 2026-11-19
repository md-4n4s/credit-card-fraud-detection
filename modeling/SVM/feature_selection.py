from modeling.logistic_regression.feature_selection import (
    try_keep_features,
    try_test_features,
)
from modeling.SVM.model_setup import *

if __name__ == "__main__":
    try_test_features(
        keep_features=keep,
        selected_model=model,
        model_name="SVM",
        scaling="StandardScaler",
        max_iter=None,
        test_features=test,
        roc_auc=False,
    )
    try_keep_features(
        keep_features=keep,
        selected_model=model,
        model_name="SVM",
        scaling="StandardScaler",
        max_iter=None,
        roc_auc=False,
    )
