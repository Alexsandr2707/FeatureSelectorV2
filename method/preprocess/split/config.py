from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, GroupConfig, SwitchConfig


@dataclass(frozen=True)
class SplitterParams(BaseConfig):
    train_size: float = 0.6


@dataclass(frozen=True)
class SplitterConfig(GroupConfig, SwitchConfig):
    params: SplitterParams = field(default_factory=SplitterParams)
