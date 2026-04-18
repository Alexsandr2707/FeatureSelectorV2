from typing import Literal
from enum import StrEnum
from dataclasses import dataclass, field

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig


class DiffHow(StrEnum):
    ADD = "add"
    REPLACE = "replace"


@dataclass(frozen=True)
class DifferParams(BaseConfig):
    how: DiffHow = DiffHow.ADD


@dataclass(frozen=True)
class DifferConfig(SwitchConfig, GroupConfig):
    params: DifferParams = DifferParams()
    pass
