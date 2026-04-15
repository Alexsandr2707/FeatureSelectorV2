import pandas as pd
import numpy as np
import logging
from rich.pretty import pretty_repr

from .config import IntervalDropperConfig
from method.datasets import Dataset
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class IntervalDropper(BasePipelineStep[Dataset, Dataset], ClassLogger):
    def __init__(self, config: IntervalDropperConfig | None = None):
        super().__init__()
        self.config = config or IntervalDropperConfig()

    @log_method()
    def transform(self, data: Dataset) -> Dataset:
        if not self.config.enabled:
            self.log("Dropper disabled")
            return data

        self.log_params("Droping intervals", *self.config.intervals)
        df = data.join_data()
        for start, end in self.config.intervals:
            start, end = pd.to_datetime(start), pd.to_datetime(end)
            df.loc[start:end] = np.nan

        data = data.reconstruct(df).dropna(how="all")
        self.log_params("Result shapes X, y", data.notna_count())
        return data
