import logging

import pandas as pd

from .config import LabHistoryConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset, DatasetBundle
from logging_tools.logging_tools import ClassLogger, log_method


class LabHistory(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: LabHistoryConfig | None = None):
        super().__init__()
        self.config = config or LabHistoryConfig()

    def _column_name(self, base_name: str, y_columns: pd.Index, y_col: str) -> str:
        if len(y_columns) == 1:
            return base_name
        return f"{base_name}_{y_col}"

    def _age_hours(self, y_col: pd.Series) -> pd.Series:
        index = y_col.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("LabHistory age feature requires a DatetimeIndex")

        observed_at = pd.Series(index=index, data=pd.NaT, dtype="datetime64[ns]")
        observed_at.loc[y_col.notna()] = index[y_col.notna()]
        observed_at = observed_at.ffill()

        current_at = pd.Series(index=index, data=index)
        age = current_at - observed_at
        return age.dt.total_seconds() / 3600.0

    def transform_dataset(self, data: Dataset) -> Dataset:
        X, y = data.copy().data
        new_X = X.copy()

        for y_col_name in y.columns:
            y_col = y[y_col_name]
            value_name = self._column_name(
                self.config.value_column, y.columns, y_col_name
            )
            new_X[value_name] = y_col.ffill()

            if self.config.include_age:
                age_name = self._column_name(
                    self.config.age_column, y.columns, y_col_name
                )
                new_X[age_name] = self._age_hours(y_col)

        return data.replace(new_X=new_X, new_y=y)

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("disabled", level=logging.WARNING)
            return data

        self.log_params("config:", self.config)
        data = data.transform(
            train_fn=self.transform_dataset,
            valid_fn=self.transform_dataset,
        )
        self.log_params("result stats", *data.stats())
        return data
