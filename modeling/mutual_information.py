import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from modeling.model_setup import X_train, y_train

scores = mutual_info_classif(X_train, y_train, random_state=42)

feature_scores = pd.Series(
    scores,
    index=X_train.columns
).sort_values(ascending=False)

print(feature_scores)