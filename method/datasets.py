import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Self, Any

from .core.config_base import BaseConfig
from .core.pipeline import BasePipelineStep


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


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.DataFrame

    def __post_init__(self):
        if not self.X.index.equals(self.y.index):
            raise ValueError("Features and target must have the same index.")

        self._x_columns = list(self.X.columns)
        self._y_columns = list(self.y.columns)

    @property
    def data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.X, self.y

    def join_data(self) -> pd.DataFrame:
        return pd.concat([self.X, self.y], axis=1)

    @classmethod
    def from_joined(
        cls,
        df: pd.DataFrame,
        x_columns: list[str],
        y_columns: list[str],
    ) -> Self:
        X = df[x_columns].copy(deep=True)
        y = df[y_columns].copy(deep=True)
        return cls(X, y)

    def reconstruct(self, df: pd.DataFrame) -> Self:
        return self.__class__.from_joined(
            df,
            self._x_columns,
            self._y_columns,
        )

    def dropna(self, how: Literal["any", "all"] = "all") -> Self:
        df = self.join_data().dropna(how=how)
        return self.reconstruct(df)

    def copy(self) -> Self:
        return self.__class__(self.X.copy(deep=True), self.y.copy(deep=True))

    def notna_count(self) -> tuple[int, int]:
        return self.X.dropna().shape[0], self.y.dropna().shape[0]


class DatasetLoader:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def load(self) -> Dataset:
        X = pd.read_csv(self.config.features_path, index_col=0, parse_dates=True)
        Y = pd.read_csv(self.config.target_path, index_col=0, parse_dates=True)
        y = Y[self.config.target_column]
        y = pd.DataFrame(y)
        X = X.asfreq(self.config.freq)
        y = y.asfreq(self.config.freq)

        df = X.join(y, how="outer")
        df = df.resample(self.config.freq).first().dropna(how="all")

        X = df.loc[:, X.columns]
        y = df.loc[:, y.columns]
        return Dataset(X, y)


class LoadDatasetStep(BasePipelineStep):
    def __init__(self, config: DatasetConfig):
        self.config = config

    def transform(self, data: Any = None) -> Dataset:
        loader = DatasetLoader(self.config)
        return loader.load()


TrainDataset = Dataset
ValidDataset = Dataset
TrainValidDataset = tuple[TrainDataset, ValidDataset]
