import logging
from copy import deepcopy
from typing import Any, Literal

import pandas as pd

from method.core.config_base import BaseConfig
from method.models.model import ModelResults

logger = logging.getLogger(__name__)


def make_logs(
    make_logs: bool, logger: logging.Logger, level: int, msg: str, *args, **kwargs
) -> None:
    if make_logs:
        logger.log(level, msg, *args, **kwargs)


def set_seeds(seed: int = 0) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_exec_params(
    name: str, exec_config: dict, default_value: Any = {}, make_logs: bool = True
) -> Any:
    if name in exec_config:
        return exec_config[name]

    if make_logs:
        logger.warning("config name '%s' not found, using default config", name)
    return default_value


def replace_exec_param(
    path: list[str],
    new_value: Any,
    config: BaseConfig | dict,
    output: Literal["same", "dict"] = "same",
) -> dict | BaseConfig:
    dict_config = (
        config.asdict() if isinstance(config, BaseConfig) else deepcopy(config)
    )
    param_name = path[-1]
    parent_path = path[:-1]
    current_config = dict_config

    try:
        for name in parent_path:
            current_config = current_config[name]
        current_config[param_name] = new_value
    except KeyError as exc:
        raise ValueError("Undefined path in config", path) from exc

    if output == "same":
        return (
            config.from_dict(dict_config)
            if isinstance(config, BaseConfig)
            else dict_config
        )
    if output == "dict":
        return dict_config

    raise ValueError("Undefined output type", output)


def inverse_transform_results(results: ModelResults, scaler: Any) -> ModelResults:
    def scale_func(x, index):
        res = scaler.inverse_transform(x)
        res = pd.DataFrame(res, index=index)
        return res

    train_fn = lambda x: scale_func(x, results.train.index)
    valid_fn = lambda x: scale_func(x, results.valid.index)

    new_train = results.train.transform_all(fn=train_fn)
    new_valid = results.valid.transform_all(fn=valid_fn)
    return results.replace(new_train=new_train, new_valid=new_valid)
