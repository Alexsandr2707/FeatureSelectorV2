import numpy as np
import pandas as pd
import logging
from statsmodels.nonparametric.smoothers_lowess import lowess

from .config import SmootherConfig, SmoothMethod, SmootherParams
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _mean_smooth(df: pd.DataFrame, params: SmootherParams) -> pd.DataFrame:
    df = df.rolling(params.limit, center=True, min_periods=1).mean()
    return df


def _lowess_smooth(
    y: pd.DataFrame,
    params: SmootherParams,
) -> pd.DataFrame:

    x_col = y.index.to_series()
    y_result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    for col in y.columns:
        y_col = y.loc[:, col]

        mask = x_col.notna()  # & y_col.notna()
        smoothed = lowess(
            endog=y_col[mask],
            exog=x_col[mask],
            frac=params.frac,
            it=0,
            return_sorted=False,
        )

        y_result.loc[mask, col] = smoothed

    return y_result


def _get_smooth_func(method: SmoothMethod):
    if method == SmoothMethod.MEAN:
        return _mean_smooth
    elif method == SmoothMethod.LOESS:
        return _lowess_smooth
    else:
        raise ValueError("Undefined smoothing method")


class Smoother(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: SmootherConfig | None = None):
        super().__init__()
        self.config = config or SmootherConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        if self.config.X.enabled:
            self.log("smoothing X_%s", name)
            smooth_fn = _get_smooth_func(self.config.X.method)
            X = smooth_fn(X, self.config.X.params)

        if self.config.y.enabled:
            self.log("smoothing y_%s", name)
            smooth_fn = _get_smooth_func(self.config.y.method)
            y = smooth_fn(y, self.config.y.params)

        return data.replace(X, y)

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
