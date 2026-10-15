from dataclasses import dataclass, field

from method.core.config_base import BaseConfig
from method.models.base import BaseModelConfig
from method.models.rnn.config import RNNTrainerParams


@dataclass(frozen=True)
class GRUAutoencoderParams(BaseConfig):
    lag: int = 48
    compression: int = 16  # Latent width for each feature's encoded history
    gru2: list[int] | None = None
    decay: float = 0.01
    lr: float = 1e-2
    min_lr: float = 1e-4
    use_best_model: bool = True


@dataclass(frozen=True)
class GRUAutoencoderConfig(BaseModelConfig):
    trainer: RNNTrainerParams = field(default_factory=RNNTrainerParams)
    model: GRUAutoencoderParams = field(default_factory=GRUAutoencoderParams)

    @classmethod
    def from_dict(cls, d: dict) -> "GRUAutoencoderConfig":
        build_dict = d.copy()
        if "trainer" in d:
            build_dict["trainer"] = RNNTrainerParams.from_dict(d["trainer"])
        if "model" in d:
            build_dict["model"] = GRUAutoencoderParams.from_dict(d["model"])
        return cls(**build_dict)
