import pandas as pd
from pathlib import Path
import logging
from typing import Any, cast, Literal
import time
from copy import deepcopy

from logging_tools.config import setup_logging
from method.datasets import DatasetConfig, LoadDatasetStep

from method.viz import plot_results, plot_prep_data
from method.preprocess.config import PreprocessConfig
from method.preprocess.preprocess import Preprocessor
import data_configs
from data_configs.base_raw_data import EXECUTE_CONFIG
from method.core.config_base import BaseConfig
from method.core.pipeline import Pipeline

from method.models.model import Model, ModelConfig, ModelResults
from method.metrics import metrics

logger = logging.getLogger(__name__)
DATASET_CONFIG = DatasetConfig.from_dict(EXECUTE_CONFIG["dataset"])


def set_seeds(seed: int = 0):
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
    name: str, exec_config: dict, default_value: Any = {}, make_logs=True
):
    if name in exec_config:
        return exec_config[name]
    else:
        logger.warning("config name '%s' not found, using default config", name)
        return default_value


def replace_exec_param(
    path: list,
    new_value: Any,
    config: BaseConfig | dict,
    output: Literal["same", "dict"] = "same",
) -> dict | BaseConfig:
    dict_config = (
        config.asdict() if isinstance(config, BaseConfig) else deepcopy(config)
    )
    param_name = path[-1]
    path = path[:-1]

    try:
        for name in path:
            dict_config = dict_config[name]
        dict_config[param_name] = new_value
    except:
        raise ValueError("Undefined path in conifg", path)

    if output == "same":
        config = (
            config.from_dict(dict_config)
            if isinstance(config, BaseConfig)
            else dict_config
        )
    elif output == "dict":
        config = dict_config
    else:
        raise ValueError("Undefined output type", output)
    return config


def inverse_transform_results(results: ModelResults, scaler: Any):
    def scale_func(x, index):
        res = scaler.inverse_transform(x)
        res = pd.DataFrame(res, index=index)
        return res

    train_fn = lambda x: scale_func(x, results.train.index)
    valid_fn = lambda x: scale_func(x, results.valid.index)

    new_train = results.train.transform_all(fn=train_fn)
    new_valid = results.valid.transform_all(fn=valid_fn)
    results = results.replace(new_train=new_train, new_valid=new_valid)
    return results


def basic_experiment(
    make_plot_prep_data: bool = False,
    make_plot_results: bool = True,
    make_logs: bool = True,
    exec_config: dict | None = None,
    cone: float | None = None,
):
    if exec_config is None:
        if make_logs:
            logger.warning("exec_config is not defined, using default config")
        exec_config = EXECUTE_CONFIG

    # configurate seeds
    seed = get_exec_params(
        "random_seed", exec_config, default_value=0, make_logs=make_logs
    )
    set_seeds(seed)

    # configurate all steps
    loader_config = DatasetConfig.from_dict(exec_config["dataset"])
    loader = LoadDatasetStep(loader_config)
    preprocessor_config = PreprocessConfig.from_dict(exec_config["preprocess"])
    preprocessor = Preprocessor(preprocessor_config)
    model_config = ModelConfig.from_dict(exec_config["model"])
    model = Model(model_config)

    # execute pipeline
    start = time.perf_counter()

    # download data
    data_raw = loader.fit_transform()

    # preprocess data
    data_prep = preprocessor.fit_transform(data_raw)

    # plot prep data
    if make_plot_prep_data:
        plot_prep_data(data_raw, data_prep)

    # train model
    result = model.fit_transform(data_prep)

    duration = time.perf_counter() - start
    if make_logs:
        logger.info(f"Full pipeline execution complited in ({duration:.2f})s")

    # inverse scaling for results
    try:
        y_scaler = preprocessor.get_step("scaler").scaler_y  # type: ignore
        result = inverse_transform_results(result, y_scaler)
        cone = 3.5 if cone is None else cone
    except:
        if make_logs:
            logger.warning("Bad scaling")
        cone = 0.2 if cone is None else cone

    # plot results
    if make_plot_results:
        plot_results(result, cone=cone)

    train_metrics = result.train.metrics(cone=cone)
    valid_metrics = result.valid.metrics(cone=cone)

    # logging results
    if make_logs:
        logger.debug("Metrics for Train:")
        logger.debug("\n%s", train_metrics)
        logger.info("Metrics for Valid:")
        logger.info("\n%s", valid_metrics)

    return valid_metrics


def seeds_experiment(
    seeds: list[int] | tuple[int, int] = (0, 42),
    exec_config: dict | None = None,
    make_logs: bool = True,
):
    seeds = seeds if isinstance(seeds, list) else list(range(*seeds))
    exec_config = EXECUTE_CONFIG if exec_config is None else exec_config
    results: dict[int, pd.Series] = {}

    for seed in seeds:
        logger.info("New Seed: %s", seed)
        exec_config = cast(dict, replace_exec_param(["random_seed"], seed, exec_config))
        metrics = basic_experiment(
            exec_config=exec_config,
            make_plot_prep_data=False,
            make_plot_results=False,
            make_logs=False,
        )
        results[seed] = metrics
        logger.info("Result Pearson: %.4f", metrics["Pearson"])
        logger.debug("All metrics: \n%s", metrics)

    comaparator = lambda x: results[x]["Pearson"]
    best_seed = max(results, key=comaparator)

    if make_logs:
        logger.info("Best seed: %s", best_seed)
        logger.info("Best metrics: \n%s", results[best_seed])

    return results, best_seed


if __name__ == "__main__":
    setup_logging(root_only=False)
    logger = logging.getLogger(__name__)
    basic_experiment(exec_config=EXECUTE_CONFIG, make_plot_prep_data=False)
