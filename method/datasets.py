import pandas as pd
import copy
from pathlib import Path
from dataclasses import dataclass, InitVar
from typing import Literal, Self, Any, Callable

from .core.config_base import BaseConfig
from .core.pipeline import BasePipelineStep


# use it make explisit behavior
class UnsetType:
    pass


UNSET = UnsetType()


@dataclass(frozen=True)
class DatasetConfig(BaseConfig):
    name: str
    features_path: Path | str
    target_path: Path | str
    target_column: str
    freq: str

    def __post_init__(self):
        object.__setattr__(self, "features_path", Path(self.features_path))
        object.__setattr__(self, "target_path", Path(self.target_path))

        if not Path(self.features_path).exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        if not Path(self.target_path).exists():
            raise FileNotFoundError(f"Target file not found: {self.target_path}")


@dataclass(slots=True, frozen=True)
class Dataset:
    X: pd.DataFrame
    y: pd.DataFrame
    X_scaler: Any = None
    y_scaler: Any = None

    make_copy: InitVar[bool] = True

    def __post_init__(self, make_copy: bool):
        X, y = self.data
        X_scaler, y_scaler = self.X_scaler, self.y_scaler
        if make_copy:
            X = X.copy(deep=True)
            y = y.copy(deep=True)
            X_scaler = copy.deepcopy(X_scaler) if X_scaler is not None else X_scaler
            y_scaler = copy.deepcopy(y_scaler) if y_scaler is not None else y_scaler

        if isinstance(y, pd.Series):
            y = y.to_frame()

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is not a Dataframe")

        if not isinstance(y, pd.DataFrame):
            raise ValueError("y is not a Dataframe")

        if not X.index.equals(y.index):
            raise ValueError("X and y must same index")

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "X_scaler", X_scaler)
        object.__setattr__(self, "y_scaler", y_scaler)

    def copy(self) -> Self:
        return self.__class__(
            self.X,
            self.y,
            X_scaler=self.X_scaler,
            y_scaler=self.y_scaler,
        )

    def replace(
        self,
        new_X: pd.DataFrame | UnsetType = UNSET,
        new_y: pd.DataFrame | UnsetType = UNSET,
        new_X_scaler: Any = UNSET,
        new_y_scaler: Any = UNSET,
        make_copy: bool = True,
    ) -> Self:
        new_X = self.X if isinstance(new_X, UnsetType) else new_X
        new_y = self.y if isinstance(new_y, UnsetType) else new_y
        new_X_scaler = (
            self.X_scaler if isinstance(new_X_scaler, UnsetType) else new_X_scaler
        )
        new_y_scaler = (
            self.y_scaler if isinstance(new_y_scaler, UnsetType) else new_y_scaler
        )
        return self.__class__(
            new_X,
            new_y,
            X_scaler=new_X_scaler,
            y_scaler=new_y_scaler,
            make_copy=make_copy,
        )

    def frame_reconstruct(self, df: pd.DataFrame, make_copy=True) -> Self:
        return self.replace(
            new_X=df[self.X.columns],
            new_y=df[self.y.columns],
            make_copy=make_copy,
        )

    @property
    def data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.X, self.y

    def join_data(self) -> pd.DataFrame:
        overlap = set(self.X.columns) & set(self.y.columns)
        if overlap:
            raise ValueError(f"Overlapping columns: {overlap}")
        return pd.concat([self.X, self.y], axis=1)

    def dropna(self, how: Literal["all", "any"] = "all") -> Self:
        return self.frame_reconstruct(self.join_data().dropna(how=how))

    def transform(
        self,
        X_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        y_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        safe: bool = True,
    ) -> Self:
        X, y = self.data
        X, y = self.replace(X, y, make_copy=safe).data  # conditional copy
        new_data = self.replace(new_X=X_fn(X), new_y=y_fn(y))
        return new_data

    def notna_counts(self, how: Literal["all", "rows"] = "all") -> tuple[int, int]:
        if how == "all":
            X_stat = int(self.X.notna().sum().sum())
            y_stat = int(self.y.notna().sum().sum())
        elif how == "rows":
            X_stat = int(self.X.notna().all(axis=1).sum())
            y_stat = int(self.y.notna().all(axis=1).sum())
        else:
            raise ValueError("Undefined 'how' name", how)
        return (X_stat, y_stat)

    def fit_scalers(
        self,
        fit_X_scaler: bool = False,
        fit_y_scaler: bool = False,
    ) -> Self:
        if fit_X_scaler and self.X_scaler:
            self.X_scaler.fit(self.X)
        if fit_y_scaler and self.y_scaler:
            self.y_scaler.fit(self.y)
        return self

    def scale(
        self,
        scale_X: bool = False,
        scale_y: bool = False,
        how: Literal["straight", "inverse"] = "straight",
        safe: bool = True,
    ) -> Self:
        def get_scale_func(scaler: Any, use_scaler: bool, how: str):
            if scaler is None or not use_scaler:
                return lambda x: x
            if how == "straight":
                return scaler.transform
            elif how == "inverse":
                return scaler.inverse_transform
            else:
                raise ValueError("Undefined 'how' type", how)

        X_fn = get_scale_func(self.X_scaler, scale_X, how)
        y_fn = get_scale_func(self.y_scaler, scale_y, how)
        new_data = self.transform(X_fn=X_fn, y_fn=y_fn, safe=safe)
        return new_data

    def stats(self, name_tail: str = ""):
        X_cells, y_cells = self.notna_counts(how="all")
        X_rows, y_rows = self.notna_counts(how="rows")
        X_name, y_name = "X" + name_tail, "y" + name_tail

        result = [
            (X_name, {"cells": X_cells, "rows": X_rows, "shape": self.X.shape}),
            (y_name, {"cells": y_cells, "rows": y_rows, "shape": self.y.shape}),
        ]
        return result

    @property
    def index(self):
        return self.X.index

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: slice):
        return self.replace(self.X.iloc[idx], self.y.iloc[idx])


@dataclass(slots=True, frozen=True)
class DatasetBundle:
    train: Dataset
    valid: Dataset | None = None

    make_copy: InitVar[bool] = False

    def __post_init__(self, make_copy: bool):
        train, valid = self.train, self.valid
        if not isinstance(train, Dataset):
            raise ValueError("train is not Dataset")

        if valid is not None and not isinstance(valid, Dataset):
            raise ValueError("valid is not None and not Dataset")

        if make_copy:
            train = train.copy()
            valid = valid.copy() if valid is not None else valid
            object.__setattr__(self, "train", train)
            object.__setattr__(self, "valid", valid)

        if valid is not None:
            self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        if self.valid is None:
            return None

        if list(self.train.X.columns) != list(self.valid.X.columns):
            raise ValueError("Train and valid must have identical feature columns")

        if list(self.train.y.columns) != list(self.valid.y.columns):
            raise ValueError("Train and valid must have identical target columns")

    def replace(
        self,
        new_train: Dataset | UnsetType = UNSET,
        new_valid: Dataset | None | UnsetType = UNSET,
        make_copy=False,
    ) -> Self:
        new_train = self.train if isinstance(new_train, UnsetType) else new_train
        new_valid = self.valid if isinstance(new_valid, UnsetType) else new_valid
        return self.__class__(new_train, new_valid, make_copy=make_copy)

    def copy(self) -> Self:
        return self.__class__(self.train, self.valid, make_copy=True)

    @property
    def data(self):
        return self.train, self.valid

    @property
    def has_valid(self):
        return self.valid is not None

    def transform(
        self,
        train_fn: Callable[[Dataset], Dataset] = lambda x: x,
        valid_fn: Callable[[Dataset], Dataset] = lambda x: x,
        safe: bool = False,
    ):
        return self.replace(
            train_fn(self.train),
            valid_fn(self.valid) if self.valid else self.valid,
            make_copy=safe,
        )

    def train_test_split(self, ratio: float = 0.6):
        if self.valid:
            raise ValueError("DatasetBundle already has valid")

        data = self.train
        stop = int(len(data) * ratio)

        new_train = data[:stop]
        new_valid = data[stop:]

        return self.replace(new_train, new_valid)

    def merge_data(self):
        if self.valid is None:
            return self.copy()

        train_df = self.train.join_data()
        valid_df = self.valid.join_data()
        full_df = pd.concat([train_df, valid_df])
        ds = self.train.frame_reconstruct(full_df, make_copy=False)
        return self.replace(new_train=ds, new_valid=None)

    def fit_scalers(
        self,
        fit_X_scaler: bool = False,
        fit_y_scaler: bool = False,
    ) -> Self:
        train, valid = self.data
        train = train.fit_scalers(fit_X_scaler=fit_X_scaler, fit_y_scaler=fit_y_scaler)
        if valid is not None:
            valid = valid.replace(
                new_X_scaler=train.X_scaler,
                new_y_scaler=train.y_scaler,
            )
        return self.replace(train, valid)

    def scale(
        self,
        scale_X: bool = False,
        scale_y: bool = False,
        how: Literal["straight", "inverse"] = "straight",
        safe: bool = True,
    ) -> Self:
        train, valid = self.data
        train = train.scale(scale_X=scale_X, scale_y=scale_y, how=how, safe=safe)
        if valid is not None:
            valid = valid.scale(scale_X=scale_X, scale_y=scale_y, how=how, safe=safe)
        return self.replace(train, valid)

    def stats(self):
        if self.valid:
            stat = self.train.stats(name_tail="_train")
            stat = stat + self.valid.stats(name_tail="_valid")
        else:
            stat = self.train.stats()
        return stat


class DatasetLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def load(self) -> Dataset:
        X = pd.read_csv(self.config.features_path, index_col=0, parse_dates=True)
        Y = pd.read_csv(self.config.target_path, index_col=0, parse_dates=True)
        y = Y[self.config.target_column].to_frame()
        X = X.asfreq(self.config.freq)
        y = y.asfreq(self.config.freq)

        df = X.join(y, how="outer")
        df = df.resample(self.config.freq).first().dropna(how="all")
        df = df.sort_index()

        X = df.loc[:, X.columns]
        y = df.loc[:, y.columns]
        return Dataset(X, y)


class LoadDatasetStep(BasePipelineStep[DatasetConfig, DatasetBundle]):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.loader = DatasetLoader(self.config)
        self.data: Dataset | None = None

    def fit(self, data: Any = None) -> Self:
        self.data = self.loader.load()
        return self

    def transform(self, data: Any = None) -> DatasetBundle:
        if self.data is None:
            raise ValueError("Dataset not loaded, make fit first")
        return DatasetBundle(self.data)

    def fit_transform(self, data: Any = None) -> DatasetBundle:
        return self.fit().transform()
