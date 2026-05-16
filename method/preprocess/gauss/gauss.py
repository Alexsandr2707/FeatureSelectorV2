import logging
import numpy as np
import pandas as pd
from typing import Literal, Self, cast

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel

from .config import GPRConfig, KernelType
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _mask_big_nan_blocks(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Return mask for NaN runs larger than max_gap"""
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.columns:
        is_na = df[col].isna()
        group_id = (is_na != is_na.shift()).cumsum()
        group_sizes = is_na.groupby(group_id).transform("sum")
        mask[col] = is_na & (group_sizes > max_gap)

    return mask


def _build_model(cfg: GPRConfig) -> GaussianProcessRegressor:
    p = cfg.params

    if p.kernel == KernelType.MATERN:
        base = Matern(nu=p.nu)
    elif p.kernel == KernelType.RBF:
        base = RBF()
    else:
        raise ValueError("Undefined kernel type", p.kernel)

    kernel = ConstantKernel(1.0) * base + WhiteKernel()

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,
        n_restarts_optimizer=p.n_restarts_optimizer,
    )


def _prepare_xy(data: Dataset, cfg: GPRConfig):
    p = cfg.params
    X = data.X.asfreq(p.freq)
    y = data.y.asfreq(p.freq)

    if p.index_as_feature:
        X = pd.DataFrame(
            {"time": np.arange(len(y), dtype=float)},
            index=y.index,
        )
    else:
        gen_index = X.index.intersection(y.index)
        X = X.loc[gen_index]
        y = y.loc[gen_index]
    return X, y


def _valid_mask(X: pd.DataFrame, y: pd.DataFrame):
    return ~y.isna().any(axis=1) & ~X.isna().any(axis=1)


class GPR(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: GPRConfig | None = None):
        super().__init__()
        self.config = config or GPRConfig()
        self.model_train: GaussianProcessRegressor | None = None
        self.model_valid: GaussianProcessRegressor | None = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.log("fitting train model")
        X_train, y_train = _prepare_xy(data.train, self.config)
        mask = _valid_mask(X_train, y_train)
        self.model_train = _build_model(self.config)
        self.model_train.fit(X_train.loc[mask], y_train.loc[mask])

        if data.valid is not None:
            self.log("fitting valid model")
            X_valid, y_valid = _prepare_xy(data.valid, self.config)
            mask = _valid_mask(X_valid, y_valid)
            self.model_valid = _build_model(self.config)
            self.model_valid.fit(X_valid.loc[mask], y_valid.loc[mask])

        return self

    def _transform_dataset(
        self, data: Dataset, dataset_type: Literal["train", "valid"]
    ) -> Dataset:
        if dataset_type == "train":
            self.log("transform train dataset")
            model = self.model_train
        elif dataset_type == "valid":
            if self.config.interp_valid:
                self.log("transform valid dataset")
                model = self.model_valid
            else:
                self.log("skip valid dataset")
                return data
        else:
            raise ValueError("Undefined dataset_type", dataset_type)

        X, y = _prepare_xy(data, self.config)
        y_res = y.copy()

        valid_X_mask = ~X.isna().any(axis=1)
        y_nan_mask = y.isna().any(axis=1)
        if self.config.params.drop_big_gap:
            big_nan_blocks = _mask_big_nan_blocks(y, self.config.params.max_gap).any(
                axis=1
            )
        else:
            big_nan_blocks = np.zeros(y.shape[0], dtype=bool)

        miss_mask = y_nan_mask & valid_X_mask & ~big_nan_blocks

        if miss_mask.any() and model is not None:
            preds, stds = cast(
                tuple[np.ndarray, np.ndarray],
                model.predict(X.loc[miss_mask], return_std=True),
            )
            threshold = self.config.params.k_confidence * stds.mean()
            keep_mask = stds < threshold
            preds = np.where(keep_mask, preds, np.nan)  # drop unstable values
            y_res.loc[miss_mask, :] = preds.reshape(-1, 1)

        return data.replace(
            new_X=data.X.asfreq(self.config.params.freq),
            new_y=y_res,
        ).dropna(how="all")

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("disabled")
            return data

        data = data.transform(
            train_fn=lambda x: self._transform_dataset(x, "train"),
            valid_fn=lambda x: self._transform_dataset(x, "valid"),
        )
        self.log_params("result stats", *data.stats())
        return data
