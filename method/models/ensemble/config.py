from dataclasses import dataclass, field
from enum import StrEnum, Enum
from typing import Self, get_type_hints, Any

from .meta_models import MetaModelParams, MetaModelType, meta_model, BaseEnsembleMeta
from method.models.rnn.rnn import RNNConfig, RNN
from method.core.config_base import BaseConfig


class EstTypes(StrEnum):
    RNN = "rnn"


class LearningMethodType(StrEnum):
    BASIC = "basic"
    EXPANDED_WINDOW = "expanded_window"


EstConfig = RNNConfig
Est = RNN


@dataclass(frozen=True)
class LearningMethodParams(BaseConfig):
    init_frac: float = 0.5
    end_frac: float = 0.3


def get_est_type(est_type: EstTypes):
    if est_type == EstTypes.RNN:
        return RNNConfig
    else:
        raise ValueError("Undefined model type", est_type)


def get_est(est_type: EstTypes, config: EstConfig):
    if est_type == EstTypes.RNN and isinstance(config, RNNConfig):
        return RNN(config)
    else:
        raise ValueError("Undefined model type or config")


@dataclass(frozen=True)
class EnsembleConfig(BaseConfig):
    est_type: EstTypes = EstTypes.RNN
    n_est: int = 3
    est_params: Any = field(default_factory=RNNConfig)
    meta_model: MetaModelType = MetaModelType.MEAN
    meta_model_params: MetaModelParams = field(default_factory=MetaModelParams)
    split_method: LearningMethodType = LearningMethodType.EXPANDED_WINDOW
    split_method_params: LearningMethodParams = field(
        default_factory=LearningMethodParams
    )

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)

        if "est_type" in d:
            estimator_type = EstTypes(d["est_type"])
        else:
            estimator_type = cls.est_type

        if "est_params" in d:
            params_type = get_est_type(estimator_type)
            params = params_type.from_dict(d["est_params"])
            build_dict["est_params"] = params

        if "split_method_params" in d:
            params_type = hints["split_method_params"]
            params = params_type.from_dict(d["split_method_params"])
            build_dict["split_method_params"] = params

        if "meta_model_params" in d:
            params_type = hints["meta_model_params"]
            params = params_type.from_dict(d["meta_model_params"])
            build_dict["meta_model_params"] = params

        return cls(**build_dict)
