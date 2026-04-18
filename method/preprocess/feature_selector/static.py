import pandas as pd

from method.datasets import DatasetBundle
from .config import SelectorParams


class StaticSelector:
    def __init__(self, config: SelectorParams | None = None) -> None:
        self.config = config or SelectorParams()

    def fit(self, X: pd.DataFrame | None = None, y: pd.DataFrame | None = None):
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> pd.DataFrame:
        select_features = self.config.select_features
        select_features_set = set(select_features)
        if len(select_features) == 0:
            raise ValueError("'select_features' lenght is zero ")

        features_set = set(X.columns)

        if select_features_set - features_set:
            raise ValueError("Undefined features", select_features_set - features_set)

        return X.loc[:, select_features]

    def get_feature_names_out(self):
        return self.config.select_features
