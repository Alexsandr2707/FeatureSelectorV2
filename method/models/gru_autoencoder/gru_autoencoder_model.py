from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from method.models.rnn.evaluate import Evaluate
from method.models.rnn.model import BaseModel


class GRUAutoencoderModel(BaseModel, Evaluate):
    def __init__(
        self,
        features_in: int,
        lag: int = 24,
        compression: int = 8,
        gru2: tuple[int, int] | list[int] | None = None,
        decay: float = 0.01,
        lr: float = 1e-2,
        use_scheduler: bool = True,
        min_lr: float = 1e-4,
        use_best_model: bool = True,
    ) -> None:
        if min(features_in, lag, compression) < 1 or compression > lag:
            raise ValueError("compression must be between 1 and lag")
        Evaluate.__init__(self)
        BaseModel.__init__(self)

        self.features_in = features_in
        self.lag = lag
        self.lr = lr
        self.decay = decay
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.use_best_model = use_best_model

        gru2_units, gru2_layers = gru2 if gru2 is not None else (features_in, 1)
        self.feed = nn.ModuleList(
            [
                # Encoder
                nn.GRU(
                    input_size=lag,
                    hidden_size=compression,
                    num_layers=1,
                    batch_first=True,
                ),
                # Decoder
                nn.GRU(
                    input_size=compression,
                    hidden_size=gru2_units,
                    num_layers=gru2_layers,
                    batch_first=True,
                ),
                # Reconstruction
                nn.Linear(gru2_units, lag),
            ]
        )
        self.loss = nn.MSELoss()

        with torch.no_grad():
            self._init_weights(self.feed.modules())

        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=decay,
        )

    @staticmethod
    def _init_weights(modules: Iterable[nn.Module]) -> None:
        for module in modules:
            if isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [batch, lag, features] as [batch, features, compression]."""
        if x.ndim != 3 or x.shape[1:] != (self.lag, self.features_in):
            raise ValueError(
                f"Expected input shape [batch, {self.lag}, {self.features_in}]"
            )
        encoded, _ = self.feed[0](x.transpose(1, 2))
        return encoded

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3 or encoded.shape[1:] != (
            self.features_in,
            self.feed[0].hidden_size,
        ):
            raise ValueError(
                "Expected encoded shape "
                f"[batch, {self.features_in}, {self.feed[0].hidden_size}]"
            )
        decoded, _ = self.feed[1](encoded)
        # [B, F, H] -> [B, F, T] -> [B, T, F]
        return self.feed[2](decoded).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def evaluate(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        *args: Any,
        fit_model: bool = True,
        device: str = "cpu",
        **kwargs: Any,
    ):
        return Evaluate.evaluate(
            self,
            X_train,
            y_train,
            *args,
            lag=self.lag,
            device=device,
            fit_model=fit_model,
            **kwargs,
        )

    def fit(
        self,
        train: Any,
        valid: Any = None,
        epochs: int = 1,
        verbose: bool = False,
        early_stopping_rounds: int = 300,
        **kwargs: Any,
    ):
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.min_lr,
            )

        return BaseModel.fit(
            self,
            train=train,
            valid=valid,
            epochs=epochs,
            penalty_func=None,
            restore=self.use_best_model,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
