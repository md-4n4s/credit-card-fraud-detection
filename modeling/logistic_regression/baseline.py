from typing import Any

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from modeling.cross_validation import evaluate_cv
from modeling.experiment_tracker import save_experiment
from modeling.logistic_regression.model_setup import (
    X_train,
    X_val,
    keep,
    y_train,
    y_val,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=0)),
    ]
)


def report_cross_validation(
    selected_model: BaseEstimator,
    features: list[str],
    model_name: str = "LogisticRegression",
    scaling: str = "StandardScaler",
    max_iter: int | None = 500,
) -> None:
    """
    Evaluate a model using cross-validation and save experiment results.

    Parameters:
        selected_model (BaseEstimator): Machine learning model to evaluate.
        features (list[str]): Features to train and evaluate.
        model_name (str): Name of the model to evaluate.
        scaling (str): Scaling method to use to scale the features.
        max_iter (int): Maximum number of iterations used by the model.
    Returns:
        None
    """

    cv_results = evaluate_cv(selected_model, X_train[features], y_train)

    results = {
        "model": f"{model_name} (Cross-validation)",
        "features": ",".join(features),
        "n_features": len(features),
        "scaling": scaling,
        "max_iter": max_iter,
        "precision": cv_results["precision"],
        "recall": cv_results["recall"],
        "f1": cv_results["f1"],
        "accuracy": cv_results["accuracy"],
        "roc_auc": cv_results["roc_auc"],
        "threshold": "Default",
        "class_distribution": "original",
    }

    save_experiment(results)


def report_keep_features(
    selected_model: Any,
    features: list[str],
    model_name: str = "LogisticRegression",
    scaling: str = "StandardScaler",
    max_iter: int | None = 500,
    roc_auc: bool = True,
) -> None:
    """
    Train a model using selected features, evaluate it on validation data,
    and save the experiment results.

    Parameters:
        selected_model (BaseEstimator): Machine learning model to train and evaluate.
        features (list[str]): List of selected feature names.
        model_name (str): Name of the model to evaluate.
        scaling (str): Scaling method to use to scale the features.
        max_iter (int): Maximum number of iterations used by the model.
        roc_auc (bool): Whether to use roc auc score.
    Returns:
        None
    """

    selected_model.fit(X_train[features], y_train)

    y_predict = selected_model.predict(X_val[features])

    precision = precision_score(y_val, y_predict)
    recall = recall_score(y_val, y_predict)
    f1 = f1_score(y_val, y_predict)
    accuracy = accuracy_score(y_val, y_predict)
    if roc_auc:
        y_prob = selected_model.predict_proba(X_val[features])[:, 1]

        roc_auc_value = roc_auc_score(y_val, y_prob)
    else:
        roc_auc_value = None

    results = {
        "model": model_name,
        "features": ",".join(features),
        "n_features": len(features),
        "scaling": scaling,
        "max_iter": max_iter,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc_value,
        "threshold": "Default",
        "class_distribution": "original",
    }

    save_experiment(results)


if __name__ == "__main__":
    report_cross_validation(selected_model=model, features=keep)
    report_keep_features(selected_model=model, features=keep)
