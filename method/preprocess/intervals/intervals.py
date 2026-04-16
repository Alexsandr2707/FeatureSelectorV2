import pandas as pd
import numpy as np
import logging

from .config import IntervalDropperConfig
from method.datasets import DatasetBundle, Dataset
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class IntervalDropper(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: IntervalDropperConfig | None = None):
        super().__init__()
        self.config = config or IntervalDropperConfig()

    def transform_dataset(self, data: Dataset):
        df = data.join_data()
        for start, end in self.config.intervals:
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            df.loc[start:end] = np.nan

        data = data.frame_reconstruct(df, make_copy=False).dropna(how="all")
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("dropper disabled", level=logging.WARNING)
            return data

        self.log_params("dropping intervals", *self.config.intervals)
        data = data.transform(
            train_fn=self.transform_dataset,
            valid_fn=self.transform_dataset,
        )

        self.log_params("result stats", *data.stats())
        return data
