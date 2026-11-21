from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.experiment_tracker import save_experiment
from modeling.logistic_regression.model_setup import (
    X_train,
    X_val,
    keep,
    test,
    y_train,
    y_val,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=0)),
    ]
)


def try_test_features(
    keep_features: list[str],
    selected_model: Any,
    model_name: str = "LogisticRegression",
    scaling: str = "StandardScaler",
    max_iter: int | None = 500,
    test_features: list[str] = test,
    roc_auc: bool = True,
) -> None:
    """
    Test additional features individually alongside the existing selected
    features, evaluate the model, and save experiment results.

    Parameters:
        keep_features (list[str]): Features already selected for the model.
        selected_model (Any): Machine learning model to train and evaluate.
        model_name (str): Name of the model.
        scaling (str): Scaling method to use.
        max_iter (int): Maximum number of iterations used by the model.
        test_features (list[str]): Additional features to test individually.
        roc_auc (bool): Whether to use roc auc score.
    Returns:
        None
    """

    for feature in test_features:
        selected_features = keep_features + [feature]
        selected_model.fit(X_train[selected_features], y_train)

        y_predict = selected_model.predict(X_val[selected_features])

        precision = precision_score(y_val, y_predict)
        recall = recall_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict)
        accuracy = accuracy_score(y_val, y_predict)
        if roc_auc:
            y_prob = selected_model.predict_proba(X_val[selected_features])[:, 1]

            roc_auc_value = roc_auc_score(y_val, y_prob)
        else:
            roc_auc_value = None

        results = {
            "model": model_name,
            "features": ",".join(selected_features),
            "n_features": len(selected_features),
            "scaling": scaling,
            "max_iter": max_iter,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "roc_auc_value": roc_auc_value,
            "threshold": "Default",
            "class_distribution": "original",
        }

        save_experiment(results)


def try_keep_features(
    keep_features: list[str],
    selected_model: Any,
    model_name: str = "LogisticRegression",
    scaling: str = "StandardScaler",
    max_iter: int | None = 500,
    roc_auc: bool = True,
) -> None:
    """
    Test the effect of removing each selected feature individually,
    evaluate the model, and save experiment results.

    Parameters:
        keep_features (list[str]): Features already selected for the model.
        selected_model (Any): Machine learning model to train and evaluate.
        model_name (str): Name of the model.
        scaling (str): Scaling method to use.
        max_iter (int): Maximum number of iterations used by the model.
        roc_auc (bool): Whether to use roc auc score.
    Returns:
        None
    """

    for feature in keep_features:
        selected_features = [f for f in keep_features if f != feature]

        selected_model.fit(X_train[selected_features], y_train)

        y_predict = selected_model.predict(X_val[selected_features])

        precision = precision_score(y_val, y_predict)
        recall = recall_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict)
        accuracy = accuracy_score(y_val, y_predict)

        if roc_auc:
            y_prob = selected_model.predict_proba(X_val[selected_features])[:, 1]

            roc_auc_value = roc_auc_score(y_val, y_prob)
        else:
            roc_auc_value = None

        results = {
            "model": model_name,
            "features": ",".join(selected_features),
            "n_features": len(selected_features),
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
    try_test_features(keep_features=keep, selected_model=model)
    try_keep_features(keep_features=keep, selected_model=model)
