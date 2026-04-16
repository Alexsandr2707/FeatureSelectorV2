import numpy as np
import pandas as pd
import logging

from .config import FilterConfig, FilterParams
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _filter_data(
    data: pd.DataFrame,
    config: FilterParams,
) -> pd.DataFrame:
    filter = (
        data.resample(config.filter_freq)
        .mean()
        .diff(1)
        .abs()
        .resample(config.freq)
        .first()
        .interpolate(method="nearest")
    )
    data[filter > config.max_diff] = np.nan
    return data


class Filter(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: FilterConfig | None = None):
        super().__init__()
        self.config = config or FilterConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        if self.config.X.enabled:
            self.log_params(f"filtering X_{name}", self.config.X.params)
            X = _filter_data(X, self.config.X.params)
        if self.config.y.enabled:
            self.log_params(f"filtering y_{name}", self.config.y.params)
            y = _filter_data(y, self.config.y.params)

        data = Dataset(X, y, make_copy=False).dropna(how="all")
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("filter disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x, name="train")
        valid_fn = lambda x: self.transform_dataset(x, name="valid")
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        self.log_params("result stats", *data.stats())
        return data
