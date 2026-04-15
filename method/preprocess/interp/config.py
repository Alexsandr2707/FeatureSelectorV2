from dataclasses import dataclass, field
from typing import Literal

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig, XyConfig

InterpType = Literal["time", "linear", "nearest"]
DirectionType = Literal["forward", "backward", "both"]
AreaType = Literal["inside", "outside"]


@dataclass(frozen=True)
class InterpParams(BaseConfig):
    method: InterpType = "time"
    order: int | None = None
    limit: int = 24
    limit_direction: DirectionType = "both"
    limit_area: AreaType = "inside"


@dataclass(frozen=True)
class InterpGroupConfig(GroupConfig, SwitchConfig):
    freq: str = "1h"
    params: InterpParams = field(default_factory=InterpParams)


@dataclass(frozen=True)
class InterpConfig(XyConfig, SwitchConfig):
    X: InterpGroupConfig = field(default_factory=InterpGroupConfig)
    y: InterpGroupConfig = field(default_factory=InterpGroupConfig)
