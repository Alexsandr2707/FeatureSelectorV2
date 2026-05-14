from dataclasses import dataclass, field

from method.core.config_base import BaseConfig
from ..base import BaseModelConfig


@dataclass(frozen=True)
class RNNParams(BaseConfig):
    lag: int = 48
    gru: list[int] = field(default_factory=lambda: [16, 1])
    l2: float = 0.00
    decay: float = 0.01
    lr: float = 1e-2
    min_lr: float = 1e-4
    use_best_model: bool = True


@dataclass(frozen=True)
class RNNTrainerParams(BaseConfig):
    epochs: int = 200
    batch: int = 128
    early_stoping: int = 300


@dataclass(frozen=True)
class RNNPretrainConfig(BaseConfig):
    enabled: bool = False
    horizon: int = 1
    trainer: RNNTrainerParams = field(default_factory=RNNTrainerParams)

    @classmethod
    def from_dict(cls, d: dict) -> "RNNPretrainConfig":
        build_dict = d.copy()
        if "trainer" in d:
            build_dict["trainer"] = RNNTrainerParams.from_dict(d["trainer"])
        return cls(**build_dict)


@dataclass(frozen=True)
class RNNConfig(BaseModelConfig):
    trainer: RNNTrainerParams = field(default_factory=RNNTrainerParams)
    model: RNNParams = field(default_factory=RNNParams)
    pretrain: RNNPretrainConfig = field(default_factory=RNNPretrainConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "RNNConfig":
        build_dict = d.copy()
        if "trainer" in d:
            build_dict["trainer"] = RNNTrainerParams.from_dict(d["trainer"])
        if "model" in d:
            build_dict["model"] = RNNParams.from_dict(d["model"])
        if "pretrain" in d:
            build_dict["pretrain"] = RNNPretrainConfig.from_dict(d["pretrain"])
        return cls(**build_dict)
