import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling.train_test_split import split_data

drop = ["Time", "V6", "V13", "V20", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

test = ["V1", "V5", "V8", "V21", "Amount"]

keep = [
    "V7",
    "V9",
    "V10",
    "V11",
    "V14",
    "V16",
    "V17",
    "V19",
]

# during feature selection
removed_from_keep = [
    "V2",
    "V3",
    "V4",
    "V12",
    "V18",
]
test_to_keep = [
    "V19",
]

if __name__ == "__main__":
    df = pd.read_csv("../../data/preprocessed/creditcard.csv")
else:
    df = pd.read_csv("../data/preprocessed/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    ]
)
