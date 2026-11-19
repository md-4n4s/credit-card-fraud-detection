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
    train_features,
    train_target,
    val_features,
    val_target,
    selected_model,
    model_name,
    max_iter,
    scaling="None",
):
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
        "features": ",".join(train_features),
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
