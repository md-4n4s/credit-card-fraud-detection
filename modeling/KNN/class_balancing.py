from modeling.KNN.model_setup import *
from modeling.random_forest.class_balancing import use_SMOTE

if __name__ == "__main__":
    use_SMOTE(
        train_features=X_train[keep],
        train_target=y_train,
        val_features=X_val[keep],
        val_target=y_val,
        selected_model=model,
        model_name="KNN",
        max_iter=5,
        scaling="StandardScaler",
    )
