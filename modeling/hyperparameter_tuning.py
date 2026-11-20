from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modeling.KNN.model_setup import *

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)

param_grid = {
    "knn__n_neighbors": [3, 5, 7],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
)

grid_search.fit(X_train[keep], y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_}")
