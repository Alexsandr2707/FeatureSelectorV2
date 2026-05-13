from enum import StrEnum

import torch
import torch.nn as nn
import torch.optim as optim

from ..rnn.evaluate import Evaluate
from ..rnn.model import BaseModel, GRU_state


class GateType(StrEnum):
    TRAINABLE = "trainable"
    MEAN = "mean"


class MoERNNModel(BaseModel, Evaluate):
    def __init__(
        self,
        features_in,
        features_out=1,
        num_experts=3,
        gate_type: GateType | str = GateType.TRAINABLE,
        lag=24,
        gru=(8, 1),
        decay=0.01,
        lr=1e-3,
        use_scheduler=True,
        min_lr=1e-4,
        use_best_model=True,
    ):
        Evaluate.__init__(self)
        BaseModel.__init__(self)

        self.num_experts = num_experts
        self.gate_type = GateType(gate_type)
        self.lag = lag
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr
        self.use_best_model = use_best_model

        try:
            units, num_layers = gru
        except:
            units, num_layers = gru, 1

        exp_list = []
        for _ in range(self.num_experts):
            exp = nn.Sequential(
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
            with torch.no_grad():
                self._init_weights(modules=exp.modules())
            exp_list.append(exp)

        self.experts = nn.ModuleList(exp_list)
        self.loss = nn.MSELoss()

        # Decides which expert to trust based on the last step
        if self.gate_type == GateType.TRAINABLE:
            self.gate = nn.Sequential(
                nn.Linear(features_in, units),
                nn.ReLU(),
                nn.Linear(units, self.num_experts),
                nn.Softmax(dim=1),
            )
            with torch.no_grad():
                self._init_uniform_gate()
        elif self.gate_type == GateType.MEAN:
            self.gate = None
        else:
            raise ValueError("Undefined gate type", self.gate_type)

        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=decay,
        )

    @staticmethod
    def _init_weights(modules):
        # TODO: initialization of other modules leads to bad results. Why???
        for m in modules:
            if isinstance(m, GRU_state):
                for name, param in m.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def _init_uniform_gate(self):
        if self.gate is None:
            return

        final_linear = self.gate[2]
        nn.init.zeros_(final_linear.weight)  # type: ignore
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)  # type: ignore

    def forward(self, x):
        # get predictions from each expert
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(
            expert_outputs, dim=1
        )  # (batch, num_experts, features_out)

        # x[:, -1, :]: (batch, features_in)
        if self.gate_type == GateType.TRAINABLE and self.gate is not None:
            gate_weights = self.gate(x[:, -1, :])
            output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        else:
            # average all experts
            output = torch.mean(expert_outputs, dim=1)

        return output

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
        self,
        train,
        valid=None,
        epochs=100,
        early_stopping_rounds=300,
        restore=True,
        verbose=False,
        penalty_func=None,
        **kwargs
    ):
        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=self.min_lr
            )

        BaseModel.fit(
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
