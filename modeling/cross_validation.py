from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_cv(
    model: Any, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> dict[str, float]:
    """
    Evaluate a machine learning model using stratified cross-validation
    and calculate the mean evaluation scores.

    Parameters:
        model (Any): Machine learning model to evaluate.
        X (pd.DataFrame): Feature dataset to evaluate.
        y (pd.Series): Target labels.
        n_splits (int, optional): Number of cross-validation folds. Defaults to 5.
    Returns:
        dict[str, float]: Dictionary containing mean precision, recall, F1, accuracy, and ROC-AUC scores across all folds.
    """

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = ["precision", "recall", "f1", "accuracy", "roc_auc"]

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {metric: scores[f"test_{metric}"].mean() for metric in scoring}
