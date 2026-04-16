from functools import partial
import logging

from .config import OutlierRemoverConfig, OutlierParams, ScopeType
from .remove_funcs import remove_global_outliers_iqr, remove_local_outliers_iqr
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import ClassLogger, log_method


def get_remove_func(scope: ScopeType):
    if scope == "local":
        return remove_local_outliers_iqr
    elif scope == "global":
        return remove_global_outliers_iqr
    else:
        raise ValueError(f"Unknown scope type: {scope}")


class OutlierRemover(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: OutlierRemoverConfig | None = None):
        super().__init__()
        self.config = config or OutlierRemoverConfig()

    def transform_dataset(self, data: Dataset, name: str = "train") -> Dataset:
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
            self.log_params(f"remove X_{name} outliers", self.config.X.params)
            X = X.apply(remove_X)
        else:
            self.log("not remove X outliers")

        if remove_y:
            self.log_params(f"remove y_{name} outliers", self.config.y.params)
            y = y.apply(remove_y) if remove_y else y
        else:
            self.log("not remove y outliers")

        data = data.replace(X, y, make_copy=False)
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("remover disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x, name="train")
        valid_fn = lambda x: self.transform_dataset(x, name="valid")
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        self.log_params("result stats", *data.stats())
        return data
