from dataclasses import asdict
import logging

from .config import InterpConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset
from logging_tools.logging_tools import log_method, ClassLogger


class Interpolator(BasePipelineStep[Dataset, Dataset], ClassLogger):
    def __init__(self, config: InterpConfig | None = None):
        super().__init__()
        self.config = config or InterpConfig()

    @log_method()
    def transform(self, data: Dataset) -> Dataset:
        if not self.config.enabled:
            self.log("Interpolator disabled")
            return data

        X, y = data.copy().data
        X, y = X.asfreq(self.config.X.freq), y.asfreq(self.config.y.freq)

        if self.config.X.enabled:
            self.log_params("Interpolate X", self.config.X.params)
            X = X.interpolate(**asdict(self.config.X.params))
        else:
            self.log("Not interpolate X")

        if self.config.y.enabled:
            self.log_params("Interpolate y", self.config.y.params)
            y = y.interpolate(**asdict(self.config.y.params))
        else:
            self.log("Not interpolate y")

        # Drop NaN
        data = Dataset(X, y).dropna(how="all")
        self.log_params("Result shapes X, y", data.notna_count())
        self.log("Data interpolated", level=logging.INFO)
        return data
