from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_cv(model, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = ["precision", "recall", "f1", "accuracy", "roc_auc"]

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {metric: scores[f"test_{metric}"].mean() for metric in scoring}
