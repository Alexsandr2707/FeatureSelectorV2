import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import cast

from .config import ScalerConfig, ScalerType
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset
from logging_tools.logging_tools import log_method, ClassLogger


def get_scaler(dtype: ScalerType):
    if dtype == "standard":
        return StandardScaler()
    elif dtype == "minmax":
        return MinMaxScaler()
    elif dtype == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {dtype}")


ScalerInput = Dataset | tuple[Dataset, Dataset]
ScalerOutput = Dataset | tuple[Dataset, Dataset]


class Scaler(BasePipelineStep[ScalerInput, ScalerOutput], ClassLogger):
    def __init__(self, config: ScalerConfig | None = None):
        super().__init__()
        self.config = config or ScalerConfig()
        self.scaler_X = None
        self.scaler_y = None

    def base_transform(self, data: Dataset) -> Dataset:
        if not self.config.enabled:
            self.log("Scaler disabled")
            return data

        X, y = data.copy().data

        if self.config.X.enabled:
            self.log_params("Scale X", self.config.X.dtype)
            self.scaler_X = self.scaler_X or get_scaler(self.config.X.dtype).fit(X)
            self.scaler_X.set_output(transform="pandas")
            X = self.scaler_X.transform(X)
            X = cast(pd.DataFrame, X)
        else:
            self.log("Not scaling X")

        if self.config.y.enabled:
            self.log_params("Scale y", self.config.y.dtype)
            self.scaler_y = self.scaler_y or get_scaler(self.config.X.dtype).fit(y)
            self.scaler_y.set_output(transform="pandas")
            y = self.scaler_y.fit_transform(y)
            y = cast(pd.DataFrame, y)
        else:
            self.log("Not scaling y")

        return Dataset(X, y)

    @log_method()
    def transform(
        self,
        data: ScalerInput,
    ) -> ScalerOutput:
        self.log("Scaling train data")
        if isinstance(data, tuple):
            data_train, data_valid = data
        else:
            data_train, data_valid = data, None

        data_train = self.base_transform(data_train)
        result = data_train

        if data_valid is not None:
            self.log("Scaling validation data")
            data_valid = self.base_transform(data_valid)
            result = data_train, data_valid
        else:
            self.log("Validation data not provided")

        self.log("Scaling completed", level=logging.INFO)
        return result
