# type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, norm

from .model import BaseModel, BaseModelEMA, GRU_state
from .lag import CorrelationLag
from .evaluate import Evaluate


class RNNModel(BaseModel, Evaluate):
    def __init__(
        self,
        features_in,
        features_out=1,
        lag=24,
        gru=(8, 1),
        decay=0.01,
        l2=0.5,
        lr=1e-2,
        use_scheduler=True,
        min_lr=1e-4,
        use_best_model=True,
    ):
        Evaluate.__init__(self)
        BaseModel.__init__(self)

        self.lag = lag
        self.l2 = l2
        self.decay = decay
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.use_best_model = use_best_model
        self.features_out = features_out

        try:
            units, num_layers = gru
        except:
            units, num_layers = gru, 1

        self.feed = nn.Sequential(
            GRU_state(
                features_in,
                units,
                num_layers=num_layers,
                batch_first=True,
                last_state=True,
            ),
            nn.LayerNorm(units),
            nn.Linear(units, features_out),
        )
        self.loss = nn.MSELoss()

        with torch.no_grad():
            self._init_weights(modules=self.feed.modules())

        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=decay,
        )

    def reset_optimizer(self):
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.decay,
        )

    def replace_head(self, features_out):
        old_head = self.feed[-1]
        if not isinstance(old_head, nn.Linear):
            raise TypeError("RNNModel head must be a Linear layer")

        device = old_head.weight.device
        new_head = nn.Linear(old_head.in_features, features_out).to(device)
        self.feed[-1] = new_head
        self.features_out = features_out

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            if isinstance(m, GRU_state):
                for name, param in m.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # def penalty(self):
    #     for W in self.feed[0].parameters():
    #         break
    #     w = W * (1 - self.spearman)
    #     l2 = self.l2 * norm(w.view(-1), 2)
    #     return l2

    def forward(self, x):
        return self.feed(x)

    def evaluate(self, X_train, y_train, *args, fit_model=True, device="cpu", **kwargs):

        return Evaluate.evaluate(
            self,
            X_train,
            y_train,
            *args,
            lag=self.lag,
            device=device,
            fit_model=fit_model,
            **kwargs
        )

    def fit(
        self, train, valid, epochs=1, verbose=False, early_stopping_rounds=300, **kwargs
    ):
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=self.min_lr
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
            **kwargs
        )
