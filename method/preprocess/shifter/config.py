from dataclasses import dataclass

from method.core.config_base import SwitchConfig


@dataclass(frozen=True)
class ShifterConfig(SwitchConfig):
    horizon: int = 1
    freq: str = "1h"
