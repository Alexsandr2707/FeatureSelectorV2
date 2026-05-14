from dataclasses import dataclass

from method.core.config_base import SwitchConfig


@dataclass(frozen=True)
class LabHistoryConfig(SwitchConfig):
    include_age: bool = False
    value_column: str = "__last_lab_value__"
    age_column: str = "__last_lab_age_hours__"
