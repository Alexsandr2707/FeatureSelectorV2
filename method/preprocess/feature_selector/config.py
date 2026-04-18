from typing import Literal
from dataclasses import field, dataclass
from enum import StrEnum

from method.core.config_base import SwitchConfig, BaseConfig, GroupConfig


class SelectorType(StrEnum):
    PLS = "pls"
    STATIC = "static"


@dataclass(frozen=True)
class SelectorParams(BaseConfig):
    pls_depth: int = 3
    select_features: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SelectorConfig(GroupConfig, SwitchConfig):
    dtype: SelectorType = SelectorType.PLS
    params: SelectorParams = field(default_factory=SelectorParams)
