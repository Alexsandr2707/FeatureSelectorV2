from typing import Literal
from dataclasses import field, dataclass

from method.core.config_base import SwitchConfig, BaseConfig, GroupConfig

SelectorType = Literal["pls"]


@dataclass(frozen=True)
class SelectorParams(BaseConfig):
    pls_depth: int = 3


@dataclass(frozen=True)
class SelectorConfig(GroupConfig, SwitchConfig):
    dtype: SelectorType = "pls"
    params: SelectorParams = field(default_factory=SelectorParams)
