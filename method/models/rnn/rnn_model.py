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
        lr=1e-3,
        use_scheduler=True,
        min_lr=1e-4,
        use_best_model=True,
    ):
        Evaluate.__init__(self)
        BaseModel.__init__(self)

        self.lag = lag
        self.l2 = l2
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.use_best_model = use_best_model

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

    @staticmethod
    def _init_weights(modules):
        for m in modules:
            # 1. Инициализация кастомного GRU_state
            if isinstance(m, GRU_state):
                for name, param in m.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

            # elif isinstance(m, nn.Linear):
            #     # Вместо reset_parameters() используем Xavier
            #     nn.init.xavier_uniform_(m.weight)
            #     if m.bias is not None:
            #         nn.init.zeros_(m.bias)
            #     # m.reset_parameters()

            # elif isinstance(m, nn.LayerNorm):
            #     # standard initialization
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0)

    # def penalty(self):
    #     for W in self.feed[0].parameters():
    #         break
    #     w = W * (1 - self.spearman)
    #     l2 = self.l2 * norm(w.view(-1), 2)
    #     return l2

    def forward(self, x):
        return self.feed(x)

    def evaluate(self, X_train, y_train, *args, fit_model=True, device="cpu", **kwargs):
        # if fit_model:
        #     self.corr = CorrelationLag(
        #         maxlag=self.lag, blur=False, dropna=True, corr="spearman"
        #     )
        #     self.corr.fit(X_train[:, -1], y_train[:, -1])
        #     self.spearman = Tensor(self.corr.lags.abs().max().values).to(device)

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
