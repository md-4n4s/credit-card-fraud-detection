import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split the dataset into training, validation, and testing sets
    using stratified sampling.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        y (pd.Series): Target labels.
        test_size (float, optional): Fraction of the dataset to include in the test set. Defaults to 0.15.
        val_size (float, optional): Fraction of the dataset to include in the validation set. Defaults to 0.15.
        random_state (int, optional): Seed for the random number generator. Defaults to 42.
    Returns:
        tuple
            X_train, X_test, y_train, y_val, y_test
    """

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
