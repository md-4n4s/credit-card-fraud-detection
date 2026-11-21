from typing import Any

from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from modeling.experiment_tracker import save_experiment
from modeling.random_forest.baseline import model
from modeling.random_forest.model_setup import *


def use_SMOTE(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    val_features: pd.DataFrame,
    val_target: pd.Series,
    selected_model: Any,
    model_name: str,
    max_iter: int | None,
    scaling: str = "None",
) -> None:
    """
    Apply SMOTE to training data, train the selected model,
    evaluate it on validation data, and save experiment results.

    Parameters:
        train_features (pd.DataFrame): Training features dataframe.
        train_target (pd.Series): Training target labels.
        val_features (pd.DataFrame): Validation features dataframe.
        val_target (pd.Series): Validation target labels.
        selected_model (Any): Machine learning model to train and evaluate.
        model_name (str): Machine learning model name.
        max_iter (int): Maximum number of iterations used by the model.
        scaling (str): Scaling method applied to features.
    Returns:
        None
    """

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(train_features, train_target)

    selected_model.fit(X_resampled, y_resampled)

    y_predict = selected_model.predict(val_features)

    precision = precision_score(val_target, y_predict)
    recall = recall_score(val_target, y_predict)
    f1 = f1_score(val_target, y_predict)
    accuracy = accuracy_score(val_target, y_predict)
    y_prob = selected_model.predict_proba(val_features)[:, 1]

    roc_auc = roc_auc_score(val_target, y_prob)

    results = {
        "model": model_name,
        "features": ",".join(train_features.columns),
        "n_features": train_features.shape[1],
        "scaling": scaling,
        "max_iter": max_iter,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "threshold": "Default",
        "class_distribution": "SMOTE",
    }

    save_experiment(results)


if __name__ == "__main__":
    use_SMOTE(
        train_features=X_train[keep],
        train_target=y_train,
        val_features=X_val[keep],
        val_target=y_val,
        selected_model=model,
        model_name="RandomForest",
        max_iter=100,
    )
