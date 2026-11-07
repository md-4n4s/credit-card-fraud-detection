from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

from modeling.cross_validation import evaluate_cv
from modeling.experiment_tracker import save_experiment
from modeling.logistic_regression.model_setup import (
    X_train,
    y_train,
    X_val,
    y_val,
    keep,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=0)),
    ]
)


def report_cross_validation():
    cv_results = evaluate_cv(model, X_train[keep], y_train)

    results = {
        "model": "LogisticRegression (Cross-validation)",
        "features": ",".join(keep),
        "n_features": len(keep),
        "scaling": "StandardScaler",
        "max_iter": 500,
        "precision": cv_results["precision"],
        "recall": cv_results["recall"],
        "f1": cv_results["f1"],
        "accuracy": cv_results["accuracy"],
        "roc_auc": cv_results["roc_auc"],
        "threshold": "Default",
        "class_distribution": "original",
    }

    save_experiment(results)


def report_keep_features():
    model.fit(X_train[keep], y_train)

    y_predict = model.predict(X_val[keep])

    precision = precision_score(y_val, y_predict)
    recall = recall_score(y_val, y_predict)
    f1 = f1_score(y_val, y_predict)
    accuracy = accuracy_score(y_val, y_predict)
    y_prob = model.predict_proba(X_val[keep])[:, 1]

    roc_auc = roc_auc_score(y_val, y_prob)

    results = {
        "model": "LogisticRegression",
        "features": ",".join(keep),
        "n_features": len(keep),
        "scaling": "StandardScaler",
        "max_iter": 500,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "threshold": "Default",
        "class_distribution": "original",
    }

    save_experiment(results)


if __name__ == "__main__":
    report_cross_validation()
    report_keep_features()
