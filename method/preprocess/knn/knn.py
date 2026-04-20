import numpy as np
import pandas as pd
import logging
from sklearn.impute import KNNImputer
from typing import Self, Literal


from .config import KNNConfig
from method.datasets import Dataset, DatasetBundle
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


def _get_knn(config: KNNConfig):
    p = config.params
    return KNNImputer(n_neighbors=p.n_neighbors, weights=p.weight)


def _drop_large_nan_blocks(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    is_nan = df.isna().any(axis=1)

    groups = (is_nan != is_nan.shift()).cumsum()
    group_sizes = is_nan.groupby(groups).transform("sum")

    bad = is_nan & (group_sizes >= max_gap)

    return df.loc[~bad].copy()


def _get_df_from_dataset(
    data: Dataset,
    freq: str = "1h",
    index_as_feature: bool = False,
    drop_big_gap: bool = False,
    max_gap: int = 14 * 24,
) -> pd.DataFrame:

    if index_as_feature:
        y = data.y.asfreq(freq=freq)
        index = pd.DataFrame({y.columns[0] + "_INDEX": range(len(y))}, index=y.index)
        df = pd.concat([index, y], axis=1)
    else:
        df = data.join_data().asfreq(freq=freq)

    if drop_big_gap:
        df = _drop_large_nan_blocks(df, max_gap)
    return df


def _get_dataset_from_df(
    df: pd.DataFrame,
    data_raw: Dataset,
    freq: str = "1h",
    index_as_feature: bool = False,
):
    X_raw, y_raw = data_raw.copy().data
    X_res, y_res = X_raw.asfreq(freq), y_raw.asfreq(freq)
    gen_index = X_raw.index.intersection(df.index)
    if index_as_feature:
        y_new = df.iloc[:, 1].to_frame()
        X_new = data_raw.X
    else:
        X_new, y_new = data_raw.frame_reconstruct(df).data
    X_res.loc[gen_index] = X_new.loc[gen_index]
    y_res.loc[gen_index] = y_new.loc[gen_index]
    data = data_raw.replace(new_X=X_res, new_y=y_res)
    return data


class KNN(BasePipelineStep[DatasetBundle, DatasetBundle], ClassLogger):
    def __init__(self, config: KNNConfig | None = None):
        super().__init__()
        self.config = config or KNNConfig()
        self.knn_train = _get_knn(self.config)
        self.knn_valid = _get_knn(self.config)

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.log("fitting train dataset")
        p = self.config.params
        df_train = _get_df_from_dataset(
            data.train,
            index_as_feature=p.index_as_feature,
            freq=p.freq,
            drop_big_gap=p.drop_big_gap,
            max_gap=p.max_gap,
        )

        self.knn_train.fit(df_train)
        if data.valid:
            self.log("fitting valid dataset")
            df_valid = _get_df_from_dataset(
                data.valid, index_as_feature=p.index_as_feature, freq=p.freq
            )
            self.knn_valid.fit(df_valid)
        return self

    def transform_dataset(
        self, data: Dataset, data_type: Literal["train", "valid"] = "train"
    ) -> Dataset:
        p = self.config.params
        df = _get_df_from_dataset(
            data,
            index_as_feature=p.index_as_feature,
            freq=p.freq,
            drop_big_gap=p.drop_big_gap,
            max_gap=p.max_gap,
        )

        if data_type == "train":
            self.log("transform train dataset")
            knn_out = self.knn_train.transform(df)
        elif data_type == "valid":
            self.log("transform valid dataset")
            knn_out = self.knn_valid.transform(df)
        else:
            raise ValueError("Undefined dataset type", data_type)

        df.iloc[:, :] = knn_out
        data = _get_dataset_from_df(
            df, data, freq=p.freq, index_as_feature=p.index_as_feature
        ).dropna(how="all")
        return data

    @log_method()
    def transform(self, data: DatasetBundle) -> DatasetBundle:
        if not self.config.enabled:
            self.log("disabled", level=logging.WARNING)
            return data

        train_fn = lambda x: self.transform_dataset(x, data_type="train")
        valid_fn = lambda x: self.transform_dataset(x, data_type="valid")
        data = data.transform(train_fn=train_fn, valid_fn=valid_fn)

        self.log_params("result stats", *data.stats())
        return data
