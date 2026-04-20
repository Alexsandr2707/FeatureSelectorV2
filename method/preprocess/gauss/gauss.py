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


def _build_model(cfg: GPRConfig) -> GaussianProcessRegressor:
    p = cfg.params

    if p.kernel == KernelType.MATERN:
        base = Matern(p.length_scale, nu=p.nu)
    elif p.kernel == KernelType.RBF:
        base = RBF(p.length_scale)
    else:
        raise ValueError("Undefined kernel type", p.kernel)

    kernel = ConstantKernel(1.0) * base + WhiteKernel(p.noise_level)

    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=p.alpha,
        normalize_y=True,
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
            self.log("transform valid dataset")
            model = self.model_valid
        else:
            raise ValueError("Undefined dataset_type", dataset_type)

        X, y = _prepare_xy(data, self.config)
        y_res = y.copy()

        valid_X_mask = ~X.isna().any(axis=1)
        miss_mask = y.isna().any(axis=1) & valid_X_mask

        if miss_mask.any() and model is not None:
            preds = model.predict(X.loc[miss_mask])
            y_res.loc[miss_mask, :] = cast(pd.Series, preds).reshape(-1, 1)

        return data.replace(new_y=y_res).dropna(how="all")

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
