import os

import pandas as pd


def save_experiment(result, file="reports/results.csv"):
    os.makedirs(os.path.dirname(file), exist_ok=True)

    pd.DataFrame([result]).to_csv(
        file, mode="a", index=False, header=not os.path.exists(file)
    )
