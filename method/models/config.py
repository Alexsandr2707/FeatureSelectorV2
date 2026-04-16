from dataclasses import dataclass, field
from enum import StrEnum, Enum

from method.core.config_base import BaseConfig
from .rnn.config import RNNConfig


class ModelType(StrEnum):
    RNN = "rnn"


class ModelsConfigs(Enum):
    pass


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    model_type: ModelType = ModelType.RNN
    params = None
