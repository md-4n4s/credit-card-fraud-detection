from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate_cv(model, X, y, scoring="roc_auc", n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {"scores": scores, "mean": scores.mean()}
