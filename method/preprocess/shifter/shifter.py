import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import cast, Self

from .config import ShifterConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import log_method, ClassLogger


class Shifter(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: ShifterConfig | None = None):
        super().__init__()
        self.config = config or ShifterConfig()

    def transform_dataset(self, data: Dataset) -> Dataset:
        X, y = data.copy().data
        X, y = X.asfreq(self.config.freq), y.asfreq(self.config.freq)
        y = y.shift(-self.config.horizon)
        data = data.replace(new_X=X, new_y=y).dropna(how="all")
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("disabled", level=logging.WARNING)
            return data

        self.log_params("config:", self.config)
        self.log("creating shift")
        train_fn = lambda x: self.transform_dataset(x)
        valid_fn = lambda x: self.transform_dataset(x)
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        return data
