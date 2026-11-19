from modeling.KNN.model_setup import *
from modeling.logistic_regression.feature_selection import (
    try_keep_features,
    try_test_features,
)

if __name__ == "__main__":
    try_test_features(
        keep_features=keep,
        selected_model=model,
        model_name="KNN",
        scaling="StandardScaler",
        max_iter=5,
        test_features=test,
    )
    try_keep_features(
        keep_features=keep,
        selected_model=model,
        model_name="KNN",
        scaling="StandardScaler",
        max_iter=5,
    )
