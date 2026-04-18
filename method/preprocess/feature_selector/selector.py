import logging
from typing import Self

from .config import SelectorConfig, SelectorType
from .pls import PLSTransformer
from .static import StaticSelector
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method


logger = logging.getLogger(__name__)


def get_selector(config: SelectorConfig):
    if config.dtype == SelectorType.PLS:
        return PLSTransformer(config.params.pls_depth, dropna=True)
    elif config.dtype == SelectorType.STATIC:
        return StaticSelector(config.params)
    else:
        raise ValueError(f"Unknown selector type: {config.dtype}")


class FeatureSelector(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: SelectorConfig | None = None):
        super().__init__()
        self.config = config or SelectorConfig()
        self.selector = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.log_params("selector params", self.config)
        self.selector = get_selector(self.config)
        self.selector.fit(*data.train.data)
        self.log_params("selected features", self.selector.get_feature_names_out())
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("selector disabled", level=logging.WARNING)
            return data

        if self.selector is None:
            raise ValueError("Selector not initialized, make fit first")

        def fn(data: Dataset) -> Dataset:
            return data.replace(
                new_X=self.selector.transform(data.X),  # type: ignore
                make_copy=False,
            )

        data = data.copy()
        data = data.transform(train_fn=fn, valid_fn=fn)

        self.log_params("result stats", *data.stats())
        return data
