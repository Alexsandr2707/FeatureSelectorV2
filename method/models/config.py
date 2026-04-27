from dataclasses import dataclass, field
from enum import StrEnum, Enum
from typing import Self, get_type_hints

from method.core.config_base import BaseConfig
from .rnn.rnn import RNN, RNNConfig


class ModelType(StrEnum):
    RNN = "rnn"


ModelParams = RNNConfig


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

        if "params" in d:
            params_type = hints["params"]

            if model_type == ModelType.RNN:
                params = params_type.from_dict(d["params"])
            else:
                raise ValueError("Undefined model type")

            build_dict["params"] = params

        return cls(**build_dict)


def get_model(model_type: ModelType, params: ModelParams):
    if model_type == ModelType.RNN:
        return RNN(params)
    else:
        raise ValueError("Undefined model type")
