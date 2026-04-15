from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, GroupConfig, XyConfig, SwitchConfig


@dataclass(frozen=True)
class FilterParams(BaseConfig):
    freq: str = "1h"
    filter_freq: str = "1W"
    max_diff: float = 20


@dataclass(frozen=True)
class FilterGroupConfig(GroupConfig, SwitchConfig):
    params: FilterParams = field(default_factory=FilterParams)


@dataclass(frozen=True)
class FilterConfig(XyConfig, SwitchConfig):
    X: FilterGroupConfig = field(default_factory=FilterGroupConfig)
    y: FilterGroupConfig = field(default_factory=FilterGroupConfig)
