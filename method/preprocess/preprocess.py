import logging
from typing import Literal, Any, cast


from method.core.config_base import BaseConfig
from method.core.pipeline import Pipeline, BasePipelineStep, PipelineStepProtocol

from method.datasets import Dataset, TrainValidDataset, TrainDataset
from .config import PreprocessConfig, StepName

from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


PreprocessorInput = Dataset
PreprocessorOutput = TrainDataset | TrainValidDataset


class Preprocessor(
    BasePipelineStep[PreprocessorInput, PreprocessorOutput], ClassLogger
):
    config: PreprocessConfig
    pipeline: Pipeline

    def __init__(self, config: PreprocessConfig | None = None):
        super().__init__()
        self.config = config or PreprocessConfig()
        self.log("Initialized")

    @log_method()
    def transform(self, data: Dataset) -> PreprocessorOutput:
        self.log("Start preprocessing", level=logging.INFO)
        self.log_params(
            "Prerpocessing steps order", *self.config.steps_order, level=logging.INFO
        )
        self.log_params("Input shapes X, y", data.X.shape, data.y.shape)
        self.log_params("Not NaN X, y", data.notna_count(), level=logging.INFO)

        conf = self.config
        order = conf.steps_order
        get_pstep = lambda name: (cast(str, name), conf.step(name))
        self.pipeline = Pipeline(list(map(get_pstep, order)))

        results = self.pipeline.predict(data)

        if isinstance(results, tuple):
            data_train, data_valid = results
            self.log_params(
                "Result shapes X_train, y_train", data_train.X.shape, data_train.y.shape
            )
            self.log_params(
                "Not NaN X_train, y_train", data_train.notna_count(), level=logging.INFO
            )
            self.log_params(
                "Result shapes X_valid, y_valid", data_valid.X.shape, data_valid.y.shape
            )
            self.log_params(
                "Not NaN X_valid, y_valid", data_valid.notna_count(), level=logging.INFO
            )
        else:
            self.log_params("Result shapes X, y", results.X.shape, results.y.shape)
            self.log_params("Not NaN X, y", results.notna_count(), level=logging.INFO)

        self.log("Preprocessing completed", level=logging.INFO)
        return results
