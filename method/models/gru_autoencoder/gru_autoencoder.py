import logging
from typing import Self

import pandas as pd
import torch

from logging_tools.logging_tools import ClassLogger, log_method
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from method.models.base import ModelResults
from method.models.rnn.model import BaseData
from method.models.rnn.vector import sliding_window

from .config import GRUAutoencoderConfig
from .gru_autoencoder_model import GRUAutoencoderModel

logger = logging.getLogger(__name__)


def _reconstruction_data(dataset: Dataset, lag: int):
    features = dataset.X.dropna(how="any")
    if len(features) < lag:
        raise ValueError("Not enough complete rows to form autoencoder windows")
    windows, _, index = sliding_window(features, features, lag=lag, dropna=1)
    tensor = torch.as_tensor(windows, dtype=torch.float32)
    return tensor, tensor, index


def _make_loader(model, dataset: Dataset, lag: int, batch_size: int):
    windows, targets, index = _reconstruction_data(dataset, lag)
    loader = model.data_loader(
        windows,
        targets,
        batch_size=batch_size,
        shuffle=False,
        device="cpu",
        base_data=BaseData,
    )
    return loader, index


def _prediction_frames(model, dataset: Dataset, lag: int, batch_size: int):
    loader, index = _make_loader(model, dataset, lag, batch_size)
    true, pred = model.predict(loader)
    columns = dataset.X.columns
    return (
        pd.DataFrame(true[:, -1, :], index=index, columns=columns),
        pd.DataFrame(pred[:, -1, :], index=index, columns=columns),
    )


class GRUAutoencoder(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(self, config: GRUAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or GRUAutoencoderConfig()
        self.is_fit = False
        self.model: GRUAutoencoderModel | None = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        if data.valid is None:
            raise ValueError("Model haven't got valid data")

        self.log_params("model config:", self.config)

        self.model = GRUAutoencoderModel(
            features_in=data.train.X.shape[-1],
            lag=self.config.model.lag,
            compression=self.config.model.compression,
            gru2=self.config.model.gru2,
            decay=self.config.model.decay,
            lr=self.config.model.lr,
            use_scheduler=True,
            min_lr=self.config.model.min_lr,
            use_best_model=self.config.model.use_best_model,
        )

        train_loader, _ = _make_loader(
            self.model, data.train, self.config.model.lag, self.config.trainer.batch
        )
        valid_loader, _ = _make_loader(
            self.model, data.valid, self.config.model.lag, self.config.trainer.batch
        )
        self.model.fit(
            train_loader,
            valid_loader,
            epochs=self.config.trainer.epochs,
            verbose=True,
            early_stopping_rounds=self.config.trainer.early_stoping,
        )
        self.is_fit = True
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        if not self.is_fit or self.model is None:
            raise ValueError("Model not fitted, make it first")
        if data.valid is None:
            raise ValueError("Autoencoder evaluation requires a validation split")

        train_true, train_pred = _prediction_frames(
            self.model, data.train, self.config.model.lag, self.config.trainer.batch
        )
        valid_true, valid_pred = _prediction_frames(
            self.model, data.valid, self.config.model.lag, self.config.trainer.batch
        )
        return ModelResults.from_df(
            train_true=train_true,
            train_pred=train_pred,
            valid_true=valid_true,
            valid_pred=valid_pred,
        )
