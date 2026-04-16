from dataclasses import asdict
import logging

from .config import InterpConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import log_method, ClassLogger


class Interpolator(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: InterpConfig | None = None):
        super().__init__()
        self.config = config or InterpConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        X, y = X.asfreq(self.config.X.freq), y.asfreq(self.config.y.freq)

        if self.config.X.enabled:
            self.log_params(f"interpolate X_{name}", self.config.X.params)
            X = X.interpolate(**asdict(self.config.X.params))
        else:
            self.log(f"not interpolate X_{name}")

        if self.config.y.enabled:
            self.log_params(f"interpolate y_{name}", self.config.y.params)
            y = y.interpolate(**asdict(self.config.y.params))
        else:
            self.log(f"not interpolate y_{name}")

        # Drop NaN
        data = data.replace(X, y, make_copy=False).dropna(how="all")
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("interpolator disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x, name="train")
        valid_fn = lambda x: self.transform_dataset(x, name="valid")
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        self.log_params("result stats", *data.stats())
        return data
