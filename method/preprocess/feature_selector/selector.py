import pandas as pd
import numpy as np
import logging
from rich.pretty import pretty_repr

from .config import SelectorConfig
from .pls import PLSTransformer
from method.datasets import Dataset
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method


logger = logging.getLogger(__name__)


def get_selector(config: SelectorConfig):
    if config.dtype == "pls":
        return PLSTransformer(config.params.pls_depth, dropna=True)
    else:
        raise ValueError(f"Unknown selector type: {config.dtype}")


SelectorInput = Dataset | tuple[Dataset, Dataset]
SelectorOutput = Dataset | tuple[Dataset, Dataset]


class FeatureSelector(BasePipelineStep[SelectorInput, SelectorOutput], ClassLogger):
    def __init__(self, config: SelectorConfig | None = None):
        super().__init__()
        self.config = config or SelectorConfig()
        self.selector = get_selector(self.config)

    @log_method()
    def transform(self, data: SelectorInput) -> SelectorOutput:
        if not self.config.enabled:
            self.log("Selector disabled")
            return data

        if isinstance(data, tuple):
            data_train, data_valid = data[0].copy(), data[1].copy()
        else:
            data_train, data_valid = data.copy(), None

        self.log_params("Selector params", self.config)

        X_train, y_train = data_train.data
        self.selector.fit(X_train, y_train)
        X_train, y_train = self.selector.transform(X_train, y_train)
        data_train = Dataset(X_train, y_train)

        self.log_params("Result shapes X_train, y_train", data_train.notna_count())
        self.log_params(
            "Selected features", *data_train.X.columns.tolist(), level=logging.INFO
        )

        if data_valid is not None:
            X_valid, y_valid = self.selector.transform(*data_valid.data)
            data_valid = Dataset(X_valid, y_valid)
            return (data_train, data_valid)
        else:
            return data_train
