import pandas as pd

from modeling.train_test_split import split_data

drop = ["Time", "V6", "V13", "V20", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]

test = ["V1", "V5", "V8", "V19", "V21", "Amount"]

keep = ["V2", "V3", "V4", "V7", "V9", "V10", "V11", "V12", "V14", "V16", "V17", "V18"]

df = pd.read_csv("../data/preprocessed/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
