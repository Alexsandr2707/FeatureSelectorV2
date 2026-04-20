from enum import StrEnum
from dataclasses import dataclass, field
from typing import Literal

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig, XyConfig


@dataclass(frozen=True)
class KNNParams(BaseConfig):
    freq: str = "1h"
    n_neighbors: int = 5
    index_as_feature: bool = False
    weight: Literal["uniform", "distance"] = "uniform"
    drop_big_gap: bool = False
    max_gap: int = 14 * 24


@dataclass(frozen=True)
class KNNConfig(SwitchConfig, GroupConfig):
    params: KNNParams = field(default_factory=KNNParams)
