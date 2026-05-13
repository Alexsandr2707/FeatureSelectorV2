from dataclasses import dataclass, field
from typing import Literal
from enum import StrEnum

from method.core.config_base import SwitchConfig, XyConfig


class ScalerType(StrEnum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    STANDARD_ROLLING = "standard_rolling"
    ROBUST_ROLLING = "robust_rolling"


@dataclass(frozen=True)
class ScalerParams(SwitchConfig):
    dtype: ScalerType = ScalerType.STANDARD
    window: int = 48  # for ScalerType.WINDOW


@dataclass(frozen=True)
class ScalerConfig(XyConfig, SwitchConfig):
    X: ScalerParams = field(default_factory=ScalerParams)
    y: ScalerParams = field(default_factory=ScalerParams)
