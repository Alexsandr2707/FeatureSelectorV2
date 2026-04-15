from typing import Generic, TypeVar, Protocol, List, Self, Any

from method.core.config_base import BaseConfig

T_in = TypeVar("T_in", contravariant=True)
T_out = TypeVar("T_out", covariant=True)


class BasePipelineStep(Generic[T_in, T_out]):
    def fit(self, data: T_in) -> Self:
        return self

    def transform(self, data: T_in) -> T_out:
        return data  # type: ignore

    def fit_transform(self, data: T_in) -> T_out:
        return self.fit(data).transform(data)


class PipelineStepWithConfig(BasePipelineStep):
    def __init__(self, config: BaseConfig | None = None):
        self.config = config or BaseConfig()


class PipelineStepProtocol(Protocol[T_in, T_out]):
    def fit(self, data: T_in) -> Self: ...
    def transform(self, data: T_in) -> T_out: ...
    def fit_transform(self, data: T_in) -> T_out: ...


class Pipeline(BasePipelineStep[Any, Any]):
    def __init__(self, steps: list[tuple[str, PipelineStepProtocol[Any, Any]]]):
        self.steps = steps

    def transform(self, data: Any) -> Any:
        for name, step in self.steps:
            data = step.fit_transform(data)
        return data

    def predict(self, data: Any) -> Any:
        result = self.transform(data)
        return result
