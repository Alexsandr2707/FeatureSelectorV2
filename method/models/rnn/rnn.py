import pandas as pd
import logging
import torch
from typing import Any, cast, Self

from ..base import ModelResults, SplitResults
from .config import RNNConfig
from .rnn_model import RNNModel
from .vector import sliding_window
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _prepare_data(data: DatasetBundle, lag: int):
    X_train, y_train = data.train.data
    X_valid, y_valid = data.valid.data if data.valid else (None, None)

    X_train, y_train, train_index = sliding_window(X_train, y_train, lag=lag, dropna=1)  # type: ignore
    if X_valid is not None and y_valid is not None:
        X_valid, y_valid, valid_index = sliding_window(
            X_valid, y_valid, lag=lag, dropna=1
        )  # type: ignore
    else:
        X_valid, y_valid, valid_index = None, None, None

    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).float()
    X_valid_tensor = torch.tensor(X_valid).float() if X_valid is not None else None
    y_valid_tensor = torch.tensor(y_valid).float() if y_valid is not None else None
    return (
        X_train_tensor,
        y_train_tensor,
        train_index,
        X_valid_tensor,
        y_valid_tensor,
        valid_index,
    )


class RNN(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(self, config: RNNConfig | None = None):
        super().__init__()
        self.config = config or RNNConfig()
        self.is_fitted: bool = False
        self.model: RNNModel | None = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        if data.has_valid is None:
            raise ValueError("Model haven't got valid data")

        data = data.copy()
        (
            X_train_tensor,
            y_train_tensor,
            train_index,
            X_valid_tensor,
            y_valid_tensor,
            valid_index,
        ) = _prepare_data(data, self.config.model.lag)

        self.model = RNNModel(
            features_in=data.train.X.shape[-1],
            lag=self.config.model.lag,
            gru=self.config.model.gru,
            decay=self.config.model.decay,
            l2=self.config.model.l2,
            lr=self.config.model.lr,
            use_scheduler=True,
            min_lr=self.config.model.min_lr,
        )

        self.model.evaluate(
            X_train_tensor,
            y_train_tensor,
            X_valid=X_valid_tensor,
            y_valid=y_valid_tensor,
            train_index=train_index,
            valid_index=valid_index,
            verbose=True,
            batch=self.config.trainer.batch,
            epochs=self.config.trainer.epochs,
            device="cpu",
            fit_model=True,
            early_stopping_rounds=self.config.trainer.early_stoping,
        )

        self.is_fitted = True
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted, make it first")

        data = data.copy()
        (
            X_train_tensor,
            y_train_tensor,
            train_index,
            X_valid_tensor,
            y_valid_tensor,
            valid_index,
        ) = _prepare_data(data, self.config.model.lag)

        dict_result = self.model.evaluate(
            X_train_tensor,
            y_train_tensor,
            X_valid=X_valid_tensor,
            y_valid=y_valid_tensor,
            train_index=train_index,
            valid_index=valid_index,
            verbose=True,
            batch=self.config.trainer.batch,
            epochs=self.config.trainer.epochs,
            device="cpu",
            fit_model=False,
        )

        dumb = pd.DataFrame()
        tt = dict_result["train"]["true"]
        tp = dict_result["train"]["pred"]
        vt = dict_result["valid"]["true"] if dict_result["valid"] is not None else None
        vp = dict_result["valid"]["pred"] if dict_result["valid"] is not None else None

        result = ModelResults.from_df(
            train_true=tt,
            train_pred=tp,
            valid_true=vt if vt is not None else dumb,
            valid_pred=vp if vp is not None else dumb,
        )

        return result
