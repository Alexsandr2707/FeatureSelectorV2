from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal
from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig


class KernelType(StrEnum):
    RBF = "rbf"
    MATERN = "matern"


@dataclass(frozen=True)
class GPRParams(BaseConfig):
    freq: str = "1h"
    index_as_feature: bool = False

    kernel: KernelType = KernelType.MATERN
    # for matern kernel
    nu: float = 1.5  # [0.5, 1.5, 2.5, float("inf")] - recomended values

    # for gpr
    n_restarts_optimizer: int = 5

    k_confidence: float = 2  # higher value - more interpvalues, can be inf

    drop_big_gap: bool = False
    max_gap: int = 14 * 24  # 2 weeks


@dataclass(frozen=True)
class GPRConfig(SwitchConfig, GroupConfig):
    interp_valid: bool = False
    params: GPRParams = field(default_factory=GPRParams)
