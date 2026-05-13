import pandas as pd
from dataclasses import dataclass, field
from enum import StrEnum, Enum
from typing import Self, get_type_hints, Protocol, Self, Any

from method.datasets import DatasetBundle
from method.core.config_base import BaseConfig
from .base import ModelResults, SplitResults
from .rnn.rnn import RNN, RNNConfig
from .ensemble.ensemble import Ensemble, EnsembleConfig
from .moe_rnn.moe_rnn import MoERNN, MoERNNConfig


class ModelType(StrEnum):
    RNN = "rnn"
    ENSEMBLE = "ensemble"
    MoERNN = "moernn"


ModelParams = RNNConfig | EnsembleConfig | MoERNNConfig


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    model_type: ModelType = ModelType.RNN
    params: RNNConfig = field(default_factory=RNNConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)

        if "model_type" in d:
            model_type = ModelType(d["model_type"])
            build_dict["model_type"] = model_type
        else:
            model_type = cls.model_type

        params_type = get_model_type(model_type)
        params = params_type.from_dict(d.get("params", {}))
        build_dict["params"] = params

        return cls(**build_dict)


def get_model_type(model_type: ModelType):
    if model_type == ModelType.RNN:
        return RNNConfig
    elif model_type == ModelType.ENSEMBLE:
        return EnsembleConfig
    elif model_type == ModelType.MoERNN:
        return MoERNNConfig
    else:
        raise ValueError("Undefined model type", model_type)


def get_model(model_type: ModelType, params: ModelParams):
    if model_type == ModelType.RNN and isinstance(params, RNNConfig):
        return RNN(params)
    elif model_type == ModelType.ENSEMBLE and isinstance(params, EnsembleConfig):
        return Ensemble(params)
    elif model_type == ModelType.MoERNN and isinstance(params, MoERNNConfig):
        return MoERNN(params)
    else:
        raise ValueError("Undefined model type or config")
