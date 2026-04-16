import pandas as pd
from dataclasses import dataclass, field
from typing import Self, get_type_hints, Literal

from method.core.config_base import BaseConfig
from method.core.pipeline import BasePipelineStep
from method.datasets import Dataset


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
