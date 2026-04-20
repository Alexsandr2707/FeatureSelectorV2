from dataclasses import dataclass, field
from enum import StrEnum

from method.core.config_base import BaseConfig, SwitchConfig, GroupConfig


class KernelType(StrEnum):
    RBF = "rbf"
    MATERN = "matern"


@dataclass(frozen=True)
class GPRParams(BaseConfig):
    freq: str = "1h"
    index_as_feature: bool = False

    kernel: KernelType = KernelType.MATERN
    length_scale: float = 1.0
    nu: float = 1.5

    noise_level: float = 1e-2
    alpha: float = 1e-10

    n_restarts_optimizer: int = 2


@dataclass(frozen=True)
class GPRConfig(SwitchConfig, GroupConfig):
    params: GPRParams = field(default_factory=GPRParams)
