from enum import StrEnum
from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig, XyConfig


class SmoothMethod(StrEnum):
    MEAN = "mean"
    LOESS = "loess"


@dataclass(frozen=True)
class SmootherParams(BaseConfig):
    limit: int = 5
    frac: float = 0.1


@dataclass(frozen=True)
class SmootherGroup(GroupConfig, SwitchConfig):
    method: SmoothMethod = SmoothMethod.MEAN
    params: SmootherParams = field(default_factory=SmootherParams)


@dataclass(frozen=True)
class SmootherConfig(SwitchConfig, XyConfig):
    X: SmootherGroup = field(default_factory=SmootherGroup)
    y: SmootherGroup = field(default_factory=SmootherGroup)
