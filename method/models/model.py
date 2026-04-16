from typing import Any

from .config import ModelConfig
from logging_tools.logging_tools import ClassLogger
from method.datasets import DatasetBundle
from method.core.pipeline import BasePipelineStep


class Model(BasePipelineStep[DatasetBundle, Any], ClassLogger):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
