from dataclasses import dataclass, field
from enum import StrEnum, Enum
from typing import Self, get_type_hints, Any

from .moe_rnn_model import GateType
from method.models.rnn.rnn import RNNConfig, RNN
from method.core.config_base import BaseConfig


@dataclass(frozen=True)
class MoERNNConfig(BaseConfig):
    n_exps: int = 3
    exp_params: RNNConfig = field(default_factory=RNNConfig)
    gate_type: GateType = GateType.TRAINABLE
    make_test: bool = False
    test_frac: float = 0.8

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()

        if "exp_params" in d:
            params = RNNConfig.from_dict(d["exp_params"])
            build_dict["exp_params"] = params

        return cls(**build_dict)
