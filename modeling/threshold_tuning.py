import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from modeling.experiment_tracker import save_experiment
from modeling.KNN.model_setup import *

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier(
                metric="manhattan", n_neighbors=5, n_jobs=-1, weights="distance"
            ),
        ),
    ]
)

model.fit(X_train[keep], y_train)

y_prob = model.predict_proba(X_val[keep])[:, 1]

thresholds = np.arange(0.25, 0.51, 0.05)

for threshold in thresholds:
    y_predict = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_predict)
    precision = precision_score(y_val, y_predict)
    recall = recall_score(y_val, y_predict)

    results = {
        "model": "KNN",
        "features": ",".join(keep),
        "n_features": len(keep),
        "scaling": "StandardScaler",
        "max_iter": None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": None,
        "roc_auc": None,
        "threshold": threshold,
        "class_distribution": "original",
    }

    save_experiment(results, file="../reports/results.csv")
