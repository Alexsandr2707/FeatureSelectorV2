from typing import TypeVar, Generic
from dataclasses import field, dataclass
import pandas as pd

from method.core.config_base import SwitchConfig

StrDate = TypeVar("StrDate", bound=str | pd.Timestamp)
RawDateInterval = tuple[StrDate, StrDate] | list[StrDate]
DateInterval = tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class IntervalDropperConfig(SwitchConfig):
    intervals: list[RawDateInterval] = field(default_factory=list)

    def __post_init__(self):
        intervals = list(
            map(lambda x: (pd.to_datetime(x[0]), pd.to_datetime(x[1])), self.intervals)
        )
        object.__setattr__(self, "intervals", intervals)
