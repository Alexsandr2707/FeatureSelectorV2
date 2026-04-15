from dataclasses import dataclass, field
from typing import Literal

from method.core.config_base import SwitchConfig, XyConfig

ScalerType = Literal["standard", "minmax", "robust"]


@dataclass(frozen=True)
class ScalerParams(SwitchConfig):
    dtype: ScalerType = "standard"


@dataclass(frozen=True)
class ScalerConfig(XyConfig, SwitchConfig):
    X: ScalerParams = field(default_factory=ScalerParams)
    y: ScalerParams = field(default_factory=ScalerParams)
