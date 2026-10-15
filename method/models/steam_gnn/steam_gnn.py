from typing import Self

import pandas as pd
import torch
from torch.utils.data import DataLoader

from logging_tools.logging_tools import ClassLogger, log_method
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle

from ..base import ModelResults
from .steam_gnn_model import (
    Model as SteamGNNNetwork,
)
from .steam_gnn_model import (
    TimeSeriesDataset,
    _make_windows,
    train_model,
)


class SteamGNN(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(
        self,
        stack_cnt: int,
        time_step: int,
        multi_layer: int,
        horizon: int = 1,
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
        device: str = "cpu",
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        units: int | None = None,
    ) -> None:
        super().__init__()
        self.units = units
        self.stack_cnt = stack_cnt
        self.time_step = time_step
        self.multi_layer = multi_layer
        self.horizon = horizon
        self.dropout_rate = dropout_rate
        self.leaky_rate = leaky_rate
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.model: SteamGNNNetwork | None = None
        self.is_fit = False

    def _prepare_split(self, data: Dataset):
        complete = pd.concat([data.X, data.y], axis=1).dropna()
        values = complete.to_numpy(dtype="float32")
        feature_count = data.X.shape[1]
        X = values
        y = values[:, feature_count:]
        windows_X, windows_y = _make_windows(X, y, self.time_step, self.horizon)
        target_index = complete.index[self.time_step : len(complete) - self.horizon + 1]
        return windows_X, windows_y, target_index, complete

    def _make_loader(self, data: Dataset, shuffle: bool) -> DataLoader:
        windows_X, windows_y, _, _ = self._prepare_split(data)
        return DataLoader(
            TimeSeriesDataset(windows_X, windows_y),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.is_fit = False
        if data.train.y.shape[1] != 1:
            raise ValueError("SteamGNN supports exactly one target column")
        expected_units = data.train.X.shape[1] + data.train.y.shape[1]
        if self.units is None:
            self.units = expected_units
        if expected_units != self.units:
            raise ValueError(
                f"units must match input features plus target ({expected_units})"
            )

        self.model = SteamGNNNetwork(
            units=self.units,
            stack_cnt=self.stack_cnt,
            time_step=self.time_step,
            multi_layer=self.multi_layer,
            horizon=self.horizon,
            dropout_rate=self.dropout_rate,
            leaky_rate=self.leaky_rate,
            device=self.device,
        )
        train_loader = self._make_loader(data.train, shuffle=True)
        valid_loader = (
            self._make_loader(data.valid, shuffle=False)
            if data.valid is not None
            else None
        )
        self.model, _ = train_model(
            self.model,
            train_loader,
            valid_loader,
            epochs=self.epochs,
            lr=self.lr,
            weight_decay=self.weight_decay,
            device=self.device,
            patience=self.patience,
        )
        self.is_fit = True
        return self

    def _predict_split(self, data: Dataset):
        if not self.is_fit or self.model is None:
            raise ValueError("Model not fitted, make it first")

        windows_X, _, target_index, complete = self._prepare_split(data)
        self.model.eval()
        with torch.no_grad():
            forecast, _, _ = self.model(
                torch.as_tensor(windows_X, dtype=torch.float32, device=self.device)
            )
            prediction = forecast[..., -1][:, 0].cpu().numpy()

        predicted = pd.DataFrame(prediction, index=target_index, columns=data.y.columns)
        feature_count = data.X.shape[1]
        target_values = complete.iloc[
            self.time_step : len(complete) - self.horizon + 1,
            feature_count:,
        ].to_numpy()
        true = pd.DataFrame(target_values, index=target_index, columns=data.y.columns)
        return true, predicted

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        if not self.is_fit or self.model is None:
            raise ValueError("Model not fitted, make it first")

        train_true, train_pred = self._predict_split(data.train)
        if data.valid is not None:
            valid_true, valid_pred = self._predict_split(data.valid)
        else:
            valid_true = pd.DataFrame(columns=data.train.y.columns)
            valid_pred = pd.DataFrame(columns=data.train.y.columns)

        return ModelResults.from_df(
            train_true=train_true,
            train_pred=train_pred,
            valid_true=valid_true,
            valid_pred=valid_pred,
        )
