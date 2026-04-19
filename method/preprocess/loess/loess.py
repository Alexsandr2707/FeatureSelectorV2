import numpy as np
import pandas as pd
import logging
from statsmodels.nonparametric.smoothers_lowess import lowess


from .config import LoessConfig
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _lowess_df(
    y: pd.DataFrame,
    X: pd.DataFrame | None = None,
    feature_name: str | None = None,
    index_as_feature: bool = True,
    frac: float = 0.2,
    robust: bool = False,
    iters: int = 3,
) -> pd.DataFrame:

    if index_as_feature or X is None:
        x_col = y.index.to_series()
    elif feature_name is None:
        x_col = X.iloc[:, 0]
    elif feature_name in set(X.columns):
        x_col = X.loc[:, feature_name]
    else:
        raise ValueError("Undefined feature name", feature_name)

    y_result = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)

    for col in y.columns:
        y_col = y.loc[:, col]

        mask = x_col.notna()  # & y_col.notna()
        smoothed = lowess(
            endog=y_col[mask],
            exog=x_col[mask],
            frac=frac,
            it=iters if robust else 0,
            return_sorted=False,
        )

        y_result.loc[mask, col] = smoothed

    return y_result


class Loess(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: LoessConfig | None = None):
        super().__init__()
        self.config = config or LoessConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
        X, y = data.copy().data
        X_smooth, y_smooth = X, y
        if self.config.X.enabled:
            self.log("interpolate X_%s", name)
            pX = self.config.X.params
            X_smooth = _lowess_df(
                X,
                feature_name=pX.feature_name,
                index_as_feature=pX.index_as_feature,
                frac=pX.frac,
                robust=pX.robust,
                iters=pX.iters,
            )
        else:
            self.log("not interpolate X_%s", name)

        if self.config.y.enabled:
            self.log("interpolate y_%s", name)
            py = self.config.y.params
            y_smooth = _lowess_df(
                y,
                X=X,
                index_as_feature=py.index_as_feature,
                feature_name=py.feature_name,
                frac=py.frac,
                robust=py.robust,
                iters=py.iters,
            )
        else:
            self.log("not interpolate y_%s", name)

        data = data.replace(new_X=X_smooth, new_y=y_smooth)
        return data

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
