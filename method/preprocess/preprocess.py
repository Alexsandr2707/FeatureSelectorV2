import logging
from typing import cast

from method.core.pipeline import Pipeline, BasePipelineStep
from method.datasets import DatasetBundle
from .config import PreprocessConfig
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class Preprocessor(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    config: PreprocessConfig
    pipeline: Pipeline

    def __init__(self, config: PreprocessConfig | None = None):
        super().__init__()
        self.config = config or PreprocessConfig()
        self.log("initialized")

    @log_method(level=logging.INFO)
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        self.log_params("preprocessing steps order", self.config.steps_order)
        self.log_params("start data stats", *data.stats())

        conf = self.config
        order = conf.steps_order
        get_pipeline_step = lambda name: (cast(str, name), conf.step(name))
        self.pipeline = Pipeline(list(map(get_pipeline_step, order)), make_logs=True)

        results = self.pipeline.predict(data)

        self.log_params("result stats", *results.stats())
        return results
