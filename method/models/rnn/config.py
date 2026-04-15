from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, GroupConfig, XyConfig, SwitchConfig
from ..config_base import ModelConfig, ModelType


@dataclass(frozen=True)
class RNNParams(BaseConfig):
    lag: int = 48
    gru: list[int] = field(default_factory=lambda: [16, 1])
    l2: float = 0.00
    decay: float = 0.01
    lr: float = 1e-2
    min_lr: float = 1e-4


@dataclass(frozen=True)
class RNNTrainerParams(BaseConfig):
    epochs: int = 200
    batch: int = 128
    early_stoping: int = 300


@dataclass(frozen=True)
class RNNConfig(ModelConfig):
    trainer: RNNTrainerParams = field(default_factory=RNNTrainerParams)
    model: RNNParams = field(default_factory=RNNParams)
