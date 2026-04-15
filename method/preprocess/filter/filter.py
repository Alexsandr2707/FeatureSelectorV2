import numpy as np
import pandas as pd
import logging

from .config import FilterConfig, FilterParams
from method.datasets import Dataset
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


class Filter(BasePipelineStep[Dataset, Dataset], ClassLogger):
    def __init__(self, config: FilterConfig | None = None):
        super().__init__()
        self.config = config or FilterConfig()

    @log_method()
    def transform(self, data: Dataset) -> Dataset:
        if not self.config.enabled:
            self.log("Filter disabled")
            return data

        X, y = data.copy().data
        if self.config.X.enabled:
            self.log_params("Filtering X", self.config.X.params)
            X = _filter_data(X, self.config.X.params)
        if self.config.y.enabled:
            self.log_params("Filtering y", self.config.y.params)
            y = _filter_data(y, self.config.y.params)

        data = Dataset(X, y).dropna(how="all")
        self.log_params("Result shapes X, y", data.notna_count())
        self.log("Data filtered", level=logging.INFO)
        return data
