from functools import partial
import logging

from .config import OutlierRemoverConfig, OutlierParams, ScopeType
from .remove_funcs import remove_global_outliers_iqr, remove_local_outliers_iqr
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset
from logging_tools.logging_tools import ClassLogger, log_method


def get_remove_func(scope: ScopeType):
    if scope == "local":
        return remove_local_outliers_iqr
    elif scope == "global":
        return remove_global_outliers_iqr
    else:
        raise ValueError(f"Unknown scope type: {scope}")


class OutlierRemover(BasePipelineStep[Dataset, Dataset], ClassLogger):
    def __init__(self, config: OutlierRemoverConfig | None = None):
        super().__init__()
        self.config = config or OutlierRemoverConfig()

    @log_method()
    def transform(self, data: Dataset) -> Dataset:
        if not self.config.enabled:
            self.log("Remover disabled")
            return data

        def build_remove_func(enabled: bool, scope: ScopeType, config: OutlierParams):
            if not enabled:
                return None

            func = get_remove_func(scope)
            return partial(func, config=config)

        remove_X = build_remove_func(
            self.config.X.enabled,
            self.config.X.scope,
            self.config.X.params,
        )

        remove_y = build_remove_func(
            self.config.y.enabled,
            self.config.y.scope,
            self.config.y.params,
        )

        X, y = data.copy().data
        if remove_X:
            self.log_params("Remove X outliers", self.config.X.params)
            X = X.apply(remove_X)
        else:
            self.log("Not remove X outliers")

        if remove_y:
            self.log_params("Remove y outliers", self.config.y.params)
            y = y.apply(remove_y) if remove_y else y
        else:
            self.log("Not remove y outliers")

        data = Dataset(X, y)
        self.log_params("Result shapes X, y", data.notna_count())
        self.log("Outliers removed", level=logging.INFO)
        return data
