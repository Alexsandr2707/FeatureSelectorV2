from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Self
from enum import StrEnum
from sklearn.linear_model import Ridge, Lasso
from dataclasses import dataclass

from logging_tools.logging_tools import ClassLogger, log_method
from method.core.config_base import BaseConfig


class WeightsType(StrEnum):
    EXP = "exp"
    LINEAR = "linear"
    EQUAL = "equal"


@dataclass(frozen=True)
class MetaModelParams(BaseConfig):
    alpha: float = 0.1  # for Ridge, Lasso
    fit_intercept: bool = True  # for Ridge, Lasso
    weights_type: WeightsType = WeightsType.EQUAL
    eps: float = 1e-8  # for MSEWeights


class MetaModelType(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MSE_WEIGHTED = "mse_weighted"
    RIDGE = "ridge"
    LASSO = "lasso"


def time_decay_weights(
    n: int,
    scheme: WeightsType = WeightsType.EQUAL,
    alpha: float = 0.05,
    normalize: bool = True,
) -> np.ndarray:

    idx = np.arange(n)

    if scheme == WeightsType.EQUAL:
        weights = np.ones(n)
    elif scheme == WeightsType.EXP:
        weights = np.exp(alpha * idx)
    elif scheme == "linear":
        weights = idx + 1
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    if normalize:
        weights = weights / weights.sum()

    return weights


class BaseEnsembleMeta(ABC):
    def __init__(self):
        super().__init__()
        self.is_fit = False

    def fit(self, P: np.ndarray, y: np.ndarray) -> Self:
        """
        P: (n_samples, n_models)
        y: (n_samples,)
        """
        self.is_fit = True
        return self

    @abstractmethod
    def transform(self, P: np.ndarray) -> np.ndarray:
        """
        Aggregated forecast (n_samples,)
        """
        pass

    def fit_transform(self, P: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(P, y)
        return self.transform(P)


class MeanMeta(BaseEnsembleMeta, ClassLogger):
    @log_method()
    def fit(self, P: np.ndarray, y: np.ndarray):
        self.is_fit = True
        return self

    @log_method()
    def transform(self, P):
        assert self.is_fit
        return np.nanmean(P, axis=1)


class MedianMeta(BaseEnsembleMeta, ClassLogger):
    @log_method()
    def fit(self, P: np.ndarray, y: np.ndarray):
        self.is_fit = True
        return self

    @log_method()
    def transform(self, P):
        assert self.is_fit
        return np.nanmedian(P, axis=1)


class MSEWeightedMeta(BaseEnsembleMeta, ClassLogger):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weights = None

    @log_method()
    def fit(self, P, y):
        # MSE
        errors = ((P - y[:, None]) ** 2).mean(axis=0)

        inv = 1.0 / (errors + self.eps)
        self.weights = inv / inv.sum()

        self.is_fit = True
        self.log_params("weights:", self.weights)
        return self

    @log_method()
    def transform(self, P):
        assert self.is_fit
        return P @ self.weights


class RidgeMeta(BaseEnsembleMeta, ClassLogger):
    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        weights_type: WeightsType = WeightsType.EQUAL,
    ):
        super().__init__()
        self.weights_type = weights_type
        self.model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    @log_method()
    def fit(self, P, y):
        self.weights = time_decay_weights(len(y), scheme=self.weights_type)
        self.model.fit(P, y, sample_weight=self.weights)
        self.is_fit = True
        self.log_params("weights:", self.model.coef_)
        return self

    @log_method()
    def transform(self, P):
        assert self.is_fit
        return self.model.predict(P)


class LassoMeta(BaseEnsembleMeta, ClassLogger):
    def __init__(
        self,
        alpha=0.01,
        fit_intercept=True,
        weights_type: WeightsType = WeightsType.EQUAL,
    ):
        super().__init__()
        self.weights_type = weights_type
        self.model = Lasso(alpha=alpha, fit_intercept=fit_intercept)

    @log_method()
    def fit(self, P, y):
        self.weights = time_decay_weights(len(y), scheme=self.weights_type)
        self.model.fit(P, y, sample_weight=self.weights)
        self.is_fit = True
        self.log_params("weights:", self.model.coef_)
        return self

    @log_method()
    def transform(self, P):
        assert self.is_fit
        return self.model.predict(P)


def meta_model(strategy: MetaModelType, p: MetaModelParams) -> BaseEnsembleMeta:
    if strategy == MetaModelType.MEAN:
        return MeanMeta()
    elif strategy == MetaModelType.MEDIAN:
        return MedianMeta()
    elif strategy == MetaModelType.MSE_WEIGHTED:
        return MSEWeightedMeta(eps=p.eps)
    elif strategy == MetaModelType.RIDGE:
        return RidgeMeta(
            alpha=p.alpha,
            fit_intercept=p.fit_intercept,
            weights_type=p.weights_type,
        )
    elif strategy == MetaModelType.LASSO:
        return LassoMeta(
            alpha=p.alpha,
            fit_intercept=p.fit_intercept,
            weights_type=p.weights_type,
        )
    else:
        raise ValueError("Undefined strategy type", strategy)
