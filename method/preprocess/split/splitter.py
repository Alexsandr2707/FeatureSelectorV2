import logging
from sklearn.model_selection import train_test_split

from .config import SplitterConfig
from method.datasets import Dataset
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)

SplitterInput = Dataset
SplitterOutput = tuple[Dataset, Dataset] | Dataset


class Splitter(BasePipelineStep[SplitterInput, SplitterOutput], ClassLogger):
    def __init__(self, config: SplitterConfig | None = None):
        super().__init__()
        self.config = config or SplitterConfig()

    @log_method()
    def transform(self, data: SplitterInput) -> SplitterOutput:
        if not self.config.enabled:
            self.log("Spliter disabled")
            return data

        df = data.join_data()
        stop = int(len(df) * self.config.params.train_size)
        stop_idx = df.index[stop]

        df_train = df[:stop]
        df_valid = df[stop:]

        data_train = data.reconstruct(df_train).dropna(how="all")
        data_valid = data.reconstruct(df_valid).dropna(how="all")

        self.log_params("Result shapes X_train, y_train", data_train.notna_count())
        self.log_params("Result shapes X_valid, y_valid", data_valid.notna_count())
        self.log("Data filtered", level=logging.INFO)
        return data_train, data_valid
