import joblib

from modeling.KNN.model_setup import *

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ]
)

model.fit(X_train[keep], y_train)
joblib.dump(model, "../model/knn_model.pkl")
