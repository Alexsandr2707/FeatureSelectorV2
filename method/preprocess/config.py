from dataclasses import dataclass, field
from typing import Self, get_type_hints, Any
from enum import StrEnum

from method.core.config_base import SwitchConfig, BaseConfig
from method.core.pipeline import PipelineStepProtocol

from .intervals.intervals import IntervalDropperConfig, IntervalDropper
from .filter.filter import FilterConfig, Filter
from .outliers.remover import OutlierRemoverConfig, OutlierRemover
from .interp.interp import InterpConfig, Interpolator
from .scaler.scaler import ScalerConfig, Scaler
from .feature_selector.selector import SelectorConfig, FeatureSelector
from .split.splitter import SplitterConfig, Splitter
from .differ.differ import DifferConfig, Differ
from .smoother.smoother import SmootherConfig, Smoother
from .shifter.shifter import ShifterConfig, Shifter
from .loess.loess import LoessConfig, Loess


class StepName(StrEnum):
    LOESS = "loess"
    DIFFER = "differ"
    SMOOTHER = "smoother"
    DROP_INTERVALS = "drop_intervals"
    OUTLIERS = "outliers"
    FILTER = "filter"
    INTERPOLATION = "interpolation"
    SPLITTER = "splitter"
    SCALER = "scaler"
    FEATURE_SELECTOR = "feature_selector"
    SHIFTER = "shifter"


STEP_NAMES = [s.value for s in StepName]


STEPS_CLASS: dict[StepName, Any] = {
    StepName.SMOOTHER: Smoother,
    StepName.DIFFER: Differ,
    StepName.DROP_INTERVALS: IntervalDropper,
    StepName.OUTLIERS: OutlierRemover,
    StepName.FILTER: Filter,
    StepName.INTERPOLATION: Interpolator,
    StepName.LOESS: Loess,
    StepName.SPLITTER: Splitter,
    StepName.SCALER: Scaler,
    StepName.FEATURE_SELECTOR: FeatureSelector,
    StepName.SHIFTER: Shifter,
}


DEFAULT_STEPS_ORDER: list[StepName] = [
    StepName.SHIFTER,
    StepName.DROP_INTERVALS,
    StepName.DIFFER,
    StepName.OUTLIERS,
    StepName.FILTER,
    StepName.INTERPOLATION,
    StepName.LOESS,
    StepName.SMOOTHER,
    StepName.SPLITTER,
    StepName.SCALER,
    StepName.FEATURE_SELECTOR,
]


@dataclass(frozen=True)
class StepsConfig(BaseConfig):
    drop_intervals: IntervalDropperConfig = field(default_factory=IntervalDropperConfig)
    outliers: OutlierRemoverConfig = field(default_factory=OutlierRemoverConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    interpolation: InterpConfig = field(default_factory=InterpConfig)
    scaler: ScalerConfig = field(default_factory=ScalerConfig)
    feature_selector: SelectorConfig = field(default_factory=SelectorConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    differ: DifferConfig = field(default_factory=DifferConfig)
    smoother: SmootherConfig = field(default_factory=SmootherConfig)
    shifter: ShifterConfig = field(default_factory=ShifterConfig)
    loess: LoessConfig = field(default_factory=LoessConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)

        for param in STEP_NAMES:
            if param in d:
                inter_type = hints[param]
                build_dict[param] = inter_type.from_dict(d[param])

        return cls(**build_dict)

    def step_config(self, name: StepName):
        if name == StepName.SMOOTHER:
            return self.smoother
        if name == StepName.LOESS:
            return self.loess
        elif name == StepName.DIFFER:
            return self.differ
        elif name == StepName.DROP_INTERVALS:
            return self.drop_intervals
        elif name == StepName.FEATURE_SELECTOR:
            return self.feature_selector
        elif name == StepName.FILTER:
            return self.filter
        elif name == StepName.INTERPOLATION:
            return self.interpolation
        elif name == StepName.OUTLIERS:
            return self.outliers
        elif name == StepName.SCALER:
            return self.scaler
        elif name == StepName.SPLITTER:
            return self.splitter
        elif name == StepName.SHIFTER:
            return self.shifter
        else:
            raise ValueError("Undefined step name: ", name)

    @staticmethod
    def step_class(name: StepName):
        return STEPS_CLASS[name]


@dataclass(frozen=True)
class PreprocessConfig(SwitchConfig):
    steps_order: list[StepName] = field(default_factory=lambda: DEFAULT_STEPS_ORDER)
    steps_configs: StepsConfig = field(default_factory=StepsConfig)
    steps: dict[StepName, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        conf = self.steps_configs
        create_step = lambda name: (name, conf.step_class(name)(conf.step_config(name)))
        steps = dict(map(create_step, self.steps_order))
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)
        step_config = hints["steps_configs"]
        build_dict["steps_configs"] = step_config.from_dict(d["steps_configs"])
        return cls(**build_dict)

    def step(self, name: StepName) -> PipelineStepProtocol[Any, Any]:
        return self.steps[name]
