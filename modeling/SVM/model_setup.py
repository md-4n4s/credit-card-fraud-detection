import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from modeling.train_test_split import split_data

drop = ["Time", "V6", "V13", "V20", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

test = ["V5", "V8", "V19", "V21", "Amount"]

keep = [
    "V1",
    "V2",
    "V3",
    "V9",
    "V10",
    "V11",
    "V12",
    "V16",
    "V17",
]

# during feature selection
removed_from_keep = [
    "V4",
    "V7",
    "V14",
    "V18",
]
test_to_keep = ["V1"]

df = pd.read_csv("../../data/preprocessed/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", gamma="scale", C=100)),
    ]
)
