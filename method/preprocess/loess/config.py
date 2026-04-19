from enum import StrEnum
from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig, XyConfig


class SmoothMethod(StrEnum):
    MEAN = "mean"


@dataclass(frozen=True)
class LoessParams(BaseConfig):
    frac: float = 0.1
    robust: bool = False
    iters: int = 0  # for robast only
    feature_name: str | None = None
    index_as_feature: bool = False


@dataclass(frozen=True)
class LoessGroupConfig(SwitchConfig, GroupConfig):
    params: LoessParams = field(default_factory=LoessParams)


@dataclass(frozen=True)
class LoessConfig(SwitchConfig, XyConfig):
    X: LoessGroupConfig = field(default_factory=LoessGroupConfig)
    y: LoessGroupConfig = field(default_factory=LoessGroupConfig)
