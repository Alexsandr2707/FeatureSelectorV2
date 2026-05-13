import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .new_scalers import RollingStandardScaler, RollingRobustScaler
from typing import cast, Self
from copy import deepcopy

from .config import ScalerConfig, ScalerType, ScalerParams
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import log_method, ClassLogger


def get_scaler(params: ScalerParams):
    dtype = params.dtype
    if dtype == ScalerType.STANDARD:
        return cast(StandardScaler, StandardScaler().set_output(transform="pandas"))
    elif dtype == ScalerType.MINMAX:
        return cast(MinMaxScaler, MinMaxScaler().set_output(transform="pandas"))
    elif dtype == ScalerType.ROBUST:
        return cast(RobustScaler, RobustScaler().set_output(transform="pandas"))
    elif dtype == ScalerType.STANDARD_ROLLING:
        return RollingStandardScaler(params.window)
    elif dtype == ScalerType.ROBUST_ROLLING:
        return RollingRobustScaler(params.window)
    else:
        raise ValueError(f"Unknown scaler type: {dtype}")


class Scaler(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: ScalerConfig | None = None):
        super().__init__()
        self.config = config or ScalerConfig()
        self.scaler_X = None
        self.scaler_y = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.log_params("config", self.config)
        if self.config.X.enabled:
            self.scaler_X = get_scaler(self.config.X).fit(data.train.X)
        if self.config.y.enabled:
            self.scaler_y = get_scaler(self.config.y).fit(data.train.y)

        return self

    def transform_dataset(self, data: Dataset) -> Dataset:
        data = data.replace(
            new_X_scaler=deepcopy(self.scaler_X),
            new_y_scaler=deepcopy(self.scaler_y),
        )

        data = data.scale(
            scale_X=self.config.X.enabled,
            scale_y=self.config.y.enabled,
            safe=False,
        )

        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("scaler disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x)
        valid_fn = lambda x: self.transform_dataset(x)
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)
        self.log_params("result stats", *data.stats())
        return data
