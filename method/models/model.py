from typing import Any, Self
import logging

from .base import ModelResults
from .config import ModelConfig, get_model
from logging_tools.logging_tools import ClassLogger, log_method
from method.datasets import DatasetBundle
from method.core.pipeline import BasePipelineStep


class Model(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.model = get_model(self.config.model_type, self.config.params)

    def fit(self, data: DatasetBundle) -> Self:
        self.model.fit(data)
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        self.log("training", level=logging.INFO)
        result = self.model.transform(data)
        self.log("trained", level=logging.INFO)
        return result
