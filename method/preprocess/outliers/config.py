from dataclasses import dataclass, field
from typing import Literal

from method.core.config_base import BaseConfig, GroupConfig, XyConfig, SwitchConfig

ScopeType = Literal["local", "global"]
DropType = Literal["drop", "clip"]


@dataclass(frozen=True)
class OutlierParams(BaseConfig):
    dtype: DropType = "clip"
    k: float = 1.5
    window: int | None = 24


@dataclass(frozen=True)
class OutlierGroupConfig(GroupConfig, SwitchConfig):
    scope: ScopeType = "global"
    params: OutlierParams = field(default_factory=OutlierParams)


@dataclass(frozen=True)
class OutlierRemoverConfig(XyConfig, SwitchConfig):
    X: OutlierGroupConfig = field(default_factory=OutlierGroupConfig)
    y: OutlierGroupConfig = field(default_factory=OutlierGroupConfig)
