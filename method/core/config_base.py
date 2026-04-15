from dataclasses import fields, MISSING, dataclass, field
from typing import Self, Literal, get_type_hints


@dataclass(frozen=True)
class BaseConfig:
    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**d)


@dataclass(frozen=True)
class GroupConfig(BaseConfig):
    params: BaseConfig = field(default_factory=BaseConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)
        if "params" in d:
            params_type = hints["params"]
            build_dict["params"] = params_type.from_dict(d["params"])
        return cls(**build_dict)


@dataclass(frozen=True)
class XyConfig(BaseConfig):
    X: BaseConfig = field(default_factory=BaseConfig)
    y: BaseConfig = field(default_factory=BaseConfig)

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        build_dict = d.copy()
        hints = get_type_hints(cls)

        if "X" in d:
            X_type = hints["X"]
            build_dict["X"] = X_type.from_dict(d["X"])
        if "y" in d:
            y_type = hints["y"]
            build_dict["y"] = y_type.from_dict(d["y"])

        return cls(**build_dict)


@dataclass(frozen=True)
class SwitchConfig(BaseConfig):
    enabled: bool = True
