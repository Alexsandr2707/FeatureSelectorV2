import logging
from typing import cast

import pandas as pd

from data_configs.base_configs.base_raw_data import EXECUTE_CONFIG

from .basic import basic_experiment
from .utils import replace_exec_param

logger = logging.getLogger(__name__)


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
        run_metrics = basic_experiment(
            exec_config=exec_config,
            make_plot_results=False,
            make_logs=False,
        )
        results[seed] = run_metrics
        logger.info("Result Pearson: %.4f", run_metrics["Pearson"])
        logger.debug("All metrics: \n%s", run_metrics)

    comparator = lambda x: results[x]["Pearson"]
    best_seed = max(results, key=comparator)

    if make_logs:
        logger.info("Best seed: %s", best_seed)
        logger.info("Best metrics: \n%s", results[best_seed])

    return results, best_seed
