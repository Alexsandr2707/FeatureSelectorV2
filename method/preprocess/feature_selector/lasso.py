import pandas as pd
import numpy as np
from sklearn.linear_model import MultiTaskElasticNetCV
from typing import Literal

ThresholdType = Literal["median", "mean", "top_k"] | float


class LassoTransformer:
    def __init__(self, l1_ratio=0.7, threshold: ThresholdType = "median", top_k=None):
        self.l1_ratio = l1_ratio
        self.threshold = threshold
        self.top_k = top_k
        self.model = MultiTaskElasticNetCV(l1_ratio=l1_ratio, n_jobs=-1)
        self.is_fit = False

    def fit(self, X, y):
        self.model.fit(X, y)
        coef = self.model.coef_

        importance = np.linalg.norm(coef, axis=0)

        if self.threshold == "median":
            thr = np.median(importance)
        elif self.threshold == "mean":
            thr = np.mean(importance)
        elif self.threshold == "top_k":
            if self.top_k is None:
                raise ValueError("top_k is not set")
            thr = np.sort(importance)[-self.top_k]
        else:
            thr = self.threshold

        min_thr = np.min(importance[importance > 0])
        thr = max(thr, min_thr)

        self.selected_features = X.columns[importance >= thr]
        self.is_fit = True
        return self

    def transform(self, X, y=None):
        if not self.is_fit:
            raise RuntimeError("Not fitted")

        if y is None:
            return X[self.selected_features]
        return X[self.selected_features], y

    def get_feature_names_out(self):
        return list(self.selected_features)
