import logging

from .config import SplitterConfig
from method.datasets import DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class Splitter(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: SplitterConfig | None = None):
        super().__init__()
        self.config = config or SplitterConfig()

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("spliter disabled", level=logging.WARNING)
            return data

        data = data.train_test_split(ratio=self.config.params.train_size)
        self.log_params("result stats", *data.stats())
        return data
