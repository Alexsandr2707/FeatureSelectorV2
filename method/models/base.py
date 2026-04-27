import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Self, get_type_hints, Literal, Callable, cast

from method.core.config_base import BaseConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset
from method.metrics import metrics


class UnsetType:
    pass


UNSET = UnsetType()


@dataclass(frozen=True)
class BaseModelConfig(BaseConfig):
    trainer: BaseConfig = field(default_factory=BaseConfig)
    model: BaseConfig = field(default_factory=BaseConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)

        if "trainer" in d:
            trainer_type = hints["trainer"]
            build_dict["trainer"] = trainer_type.from_dict(d["trainer"])
        if "model" in d:
            model_type = hints["model"]
            build_dict["model"] = model_type.from_dict(d["model"])

        return cls(**build_dict)


@dataclass(frozen=True)
class SplitResults(BaseConfig):
    true: pd.DataFrame
    pred: pd.DataFrame

    def __post_init__(self):
        prep_data = lambda x: pd.DataFrame(x)
        object.__setattr__(self, "true", prep_data(self.true))
        object.__setattr__(self, "pred", prep_data(self.pred))

        true_shape = self.true.shape
        assert true_shape == (
            true_shape[0],
            true_shape[-1],
        ), f"bad shape {self.true.shape}"

        pred_shape = self.pred.shape
        assert pred_shape == (
            pred_shape[0],
            pred_shape[-1],
        ), f"bad shape {self.pred.shape}"

        assert (self.true.index == self.pred.index).all(), "indeces is not same"

    def replace(
        self,
        new_true: pd.DataFrame | UnsetType = UNSET,
        new_pred: pd.DataFrame | UnsetType = UNSET,
    ):
        new_instance = self.__class__(
            true=self.true if isinstance(new_true, UnsetType) else new_true,
            pred=self.pred if isinstance(new_pred, UnsetType) else new_pred,
        )
        return new_instance

    def metrics(self, cone=3.5):
        return metrics(self.true.iloc[:, 0], self.pred.iloc[:, 0], cone=cone)

    def transform(
        self,
        true_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        pred_fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
    ):
        true = true_fn(self.true)
        pred = pred_fn(self.pred)
        new_instance = self.replace(new_true=true, new_pred=pred)
        return new_instance

    def transform_all(self, fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x):
        return self.transform(true_fn=fn, pred_fn=fn)

    def asdict(self):
        return asdict(self)

    def asdict_of_series(self):
        df_dict = cast(dict[str, pd.DataFrame], asdict(self))
        series_dict = {}
        for k, v in df_dict.items():
            series_dict[k] = v.iloc[:, 0]
        return series_dict

    @property
    def index(self):
        return self.true.index


@dataclass(frozen=True)
class ModelResults(BaseConfig):
    train: SplitResults
    valid: SplitResults

    def replace(
        self,
        new_train: SplitResults | UnsetType = UNSET,
        new_valid: SplitResults | UnsetType = UNSET,
    ):
        new_instance = self.__class__(
            train=self.train if isinstance(new_train, UnsetType) else new_train,
            valid=self.valid if isinstance(new_valid, UnsetType) else new_valid,
        )
        return new_instance

    def transform(
        self,
        train_fn: Callable[[SplitResults], SplitResults] = lambda x: x,
        valid_fn: Callable[[SplitResults], SplitResults] = lambda x: x,
    ):
        new_train = train_fn(self.train)
        new_valid = valid_fn(self.valid)
        new_instance = self.replace(new_train=new_train, new_valid=new_valid)
        return new_instance

    def transform_all(self, fn: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x):
        new_train = self.train.transform_all(fn=fn)
        new_valid = self.valid.transform_all(fn=fn)
        new_instance = self.replace(new_train=new_train, new_valid=new_valid)
        return new_instance

    def summary(self):
        return {
            "train": self.train.metrics(),
            "valid": self.valid.metrics(),
        }

    def asdict(self):
        return asdict(self)

    def asdict_of_series(self):
        series_dict = {}
        series_dict["train"] = self.train.asdict_of_series()
        series_dict["valid"] = self.valid.asdict_of_series()
        return series_dict
