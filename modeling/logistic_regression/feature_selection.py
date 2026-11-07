from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from modeling.experiment_tracker import save_experiment
from modeling.logistic_regression.model_setup import (
    X_train,
    y_train,
    X_val,
    y_val,
    keep,
    test,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=500, random_state=0)),
    ]
)

def try_test_features():

    for feature in test:
        selected_features = keep + [feature]
        model.fit(X_train[selected_features], y_train)

        y_predict = model.predict(X_val[selected_features])

        precision = precision_score(y_val, y_predict)
        recall = recall_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict)
        accuracy = accuracy_score(y_val, y_predict)
        y_prob = model.predict_proba(X_val[selected_features])[:, 1]

        roc_auc = roc_auc_score(y_val, y_prob)

        results = {
            "model": "LogisticRegression",
            "features": ",".join(selected_features),
            "n_features": len(keep) + 1,
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

def try_keep_features():
    for feature in keep:
        selected_features = [f for f in keep if f != feature]

        model.fit(X_train[selected_features], y_train)

        y_predict = model.predict(X_val[selected_features])

        precision = precision_score(y_val, y_predict)
        recall = recall_score(y_val, y_predict)
        f1 = f1_score(y_val, y_predict)
        accuracy = accuracy_score(y_val, y_predict)

        y_prob = model.predict_proba(X_val[selected_features])[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)

        results = {
            "model": "LogisticRegression",
            "features": ",".join(selected_features),
            "n_features": len(selected_features),
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
    try_test_features()
    try_keep_features()
