import os
from typing import Any

import pandas as pd


def save_experiment(result: dict[str, Any], file: str = "../../reports/results.csv"):
    """
    Append experiment results to a CSV file, creating the directory
    and file if they do not already exist.

    Parameters:
        result (dict[str, Any]): Dictionary containing experiment metrics and model details.
        file (str, optional): File path to save experiment results. Defaults to "../../reports/results.csv".
    Returns:
        None
    """

    os.makedirs(os.path.dirname(file), exist_ok=True)

    pd.DataFrame([result]).to_csv(
        file, mode="a", index=False, header=not os.path.exists(file)
    )
