from dataclasses import asdict
import logging
import pandas as pd

from .config import InterpConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import log_method, ClassLogger


def _sparse_df(df_raw: pd.DataFrame, df_interp: pd.DataFrame, sparsify_step: int):
    df_raw = df_raw.copy()
    sparsify_step = max(sparsify_step, 1)

    df_sparsed = df_interp[::sparsify_step]
    df_common = df_raw.index.intersection(df_sparsed.index)
    df_raw.loc[df_common] = df_sparsed.loc[df_common]

    return df_raw


class Interpolator(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: InterpConfig | None = None):
        super().__init__()
        self.config = config or InterpConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        X, y = X.asfreq(self.config.X.freq), y.asfreq(self.config.y.freq)
        X_interp, y_interp = X.copy(), y.copy()

        if self.config.X.enabled:
            self.log_params(f"interpolate X_{name}", self.config.X.params)
            X_interp = X.interpolate(**asdict(self.config.X.params))
        else:
            self.log(f"not interpolate X_{name}")

        if self.config.y.enabled:
            self.log_params(f"interpolate y_{name}", self.config.y.params)
            y_interp = y.interpolate(**asdict(self.config.y.params))
        else:
            self.log(f"not interpolate y_{name}")

        # Sparse data
        X = _sparse_df(X, X_interp, self.config.X.sparsify_step)
        y = _sparse_df(y, y_interp, self.config.y.sparsify_step)

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
