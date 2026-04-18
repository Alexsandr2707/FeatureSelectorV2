import numpy as np
import pandas as pd
import logging

from .config import DifferConfig, DiffHow
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class Differ(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: DifferConfig | None = None):
        super().__init__()
        self.config = config or DifferConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        X_diff = X.diff()
        if self.config.params.how == DiffHow.ADD:
            X_diff.rename(columns=lambda x: x + "_diff", inplace=True)
            X = pd.concat([X, X_diff], axis=1)
        elif self.config.params.how == DiffHow.REPLACE:
            X = X_diff

        return data.replace(X, y)[1:]

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x, name="train")
        valid_fn = lambda x: self.transform_dataset(x, name="valid")
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        self.log_params("result stats", *data.stats())
        return data
