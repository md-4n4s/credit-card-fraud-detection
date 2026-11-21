import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from modeling.KNN.model_setup import X_test, X_val, keep, y_test, y_val

model = joblib.load("../model/knn_model.pkl")


def evaluate_knn(
    feature: pd.DataFrame, target: pd.Series, test_data: bool = True
) -> None:
    """
    Evaluate a KNN model, calculate classification metrics,
    and save a performance metrics bar chart.

    Parameters:
        feature (pd.DataFrame):  Feature dataset used for evaluation.
        target (pd.Series): Actual target labels.
        test_data (bool, optional): Whether to evaluate test data. Defaults to True. If False, validation data is assumed.
    Returns:
        None
    """

    y_predict = model.predict(feature)

    y_prob = model.predict_proba(feature)[:, 1]

    accuracy = accuracy_score(target, y_predict)
    precision = precision_score(target, y_predict)
    recall = recall_score(target, y_predict)
    f1 = f1_score(target, y_predict)
    roc_auc = roc_auc_score(target, y_prob)
    pr_auc = average_precision_score(target, y_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    plt.figure(figsize=(10, 6))

    bars = plt.bar(list(metrics.keys()), list(metrics.values()))

    plt.ylim(0, 1)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Final Model Performance")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    data = "test" if test_data else "validation"

    plt.savefig(
        f"../reports/final_model_{data}_data_metrics.png", dpi=300, bbox_inches="tight"
    )


if __name__ == "__main__":
    evaluate_knn(X_test[keep], y_test)
    evaluate_knn(X_val[keep], y_val, test_data=False)
