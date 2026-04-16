import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import cast, Self

from .config import ScalerConfig, ScalerType
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import log_method, ClassLogger


def get_scaler(dtype: ScalerType):
    if dtype == ScalerType.STANDARD:
        return StandardScaler()
    elif dtype == ScalerType.MINMAX:
        return MinMaxScaler()
    elif dtype == ScalerType.ROBUST:
        return RobustScaler()
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
            self.scaler_X = get_scaler(self.config.X.dtype).fit(data.train.X)
            self.scaler_X.set_output(transform="pandas")
        if self.config.y.enabled:
            self.scaler_y = get_scaler(self.config.y.dtype).fit(data.train.y)
            self.scaler_y.set_output(transform="pandas")

        return self

    def transform_dataset(self, data: Dataset) -> Dataset:
        data = data.replace(
            new_X_scaler=self.scaler_X,
            new_y_scaler=self.scaler_y,
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

        return data
