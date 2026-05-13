from typing import Literal
from dataclasses import field, dataclass
from enum import StrEnum

from method.core.config_base import SwitchConfig, BaseConfig, GroupConfig

from .lasso import ThresholdType
from .bayes import ScoreType


class SelectorType(StrEnum):
    PLS = "pls"
    BAYES = "bayes"
    LASSO = "lasso"
    STATIC = "static"


@dataclass(frozen=True)
class SelectorParams(BaseConfig):
    # for pls and bayes selector
    depth: int = 3

    # for bayes selector
    q_count: int = 5
    max_iter: int = int(1e6)
    scoretype: ScoreType = "bic"

    # for static selector
    select_features: list[str] = field(default_factory=list)

    # for lasso selector
    l1_ratio: float = 0.99
    threshold: ThresholdType = "median"
    top_k: int | None = None


@dataclass(frozen=True)
class SelectorConfig(GroupConfig, SwitchConfig):
    dtype: SelectorType = SelectorType.PLS
    params: SelectorParams = field(default_factory=SelectorParams)
