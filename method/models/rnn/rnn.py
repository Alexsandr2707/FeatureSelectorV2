import pandas as pd
import logging
import torch
from typing import Any

from .config import RNNConfig
from .rnn_model import RNNModel
from .vector import sliding_window
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method


logger = logging.getLogger(__name__)


class RNN(BasePipelineStep[DatasetBundle, Any], ClassLogger):
    def __init__(self, config: RNNConfig | None = None):
        super().__init__()
        self.config = config or RNNConfig()

    @log_method()
    def transform(self, data: DatasetBundle) -> Any:
        self.log("training model", level=logging.INFO)

        if data.has_valid is None:
            raise ValueError("Model haven't got valid data")

        data = data.copy()
        X_train, y_train = data.train.data
        X_valid, y_valid = data.valid.data  # type: ignore

        X_train, y_train, train_index = sliding_window(X_train, y_train, lag=self.config.model.lag, dropna=1)  # type: ignore
        if X_valid is not None and y_valid is not None:
            X_valid, y_valid, valid_index = sliding_window(
                X_valid, y_valid, lag=self.config.model.lag, dropna=1
            )  # type: ignore
        else:
            self.log(
                "no validation data provided, skipping validation",
                level=logging.WARNING,
            )
            X_valid, y_valid, valid_index = None, None, None

        X_train_tensor = torch.tensor(X_train).float()
        X_valid_tensor = torch.tensor(X_valid).float()
        y_train_tensor = torch.tensor(y_train).float()
        y_valid_tensor = torch.tensor(y_valid).float()

        model = RNNModel(
            features_in=X_train.shape[-1],
            lag=self.config.model.lag,
            gru=self.config.model.gru,
            decay=self.config.model.decay,
            l2=self.config.model.l2,
            lr=self.config.model.lr,
            use_scheduler=True,
            min_lr=self.config.model.min_lr,
        )

        result = model.evaluate(
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
            early_stopping_rounds=self.config.trainer.early_stoping,
        )

        self.log("Model trained", level=logging.INFO)
        return result
