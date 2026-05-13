import logging

from data_configs.base_raw_data import EXECUTE_CONFIG
from experiments import (
    ParamsGrid,
    basic_experiment,
    get_diff,
    get_exec_params,
    grid_search,
    inverse_transform_results,
    replace_exec_param,
    seeds_experiment,
    set_seeds,
    metrics_summary,
)
from experiments.utils import make_logs as _make_logs
from logging_tools.config import setup_logging
from method.datasets import DatasetConfig

logger = logging.getLogger(__name__)
DATASET_CONFIG = DatasetConfig.from_dict(EXECUTE_CONFIG["dataset"])

__all__ = [
    "DATASET_CONFIG",
    "EXECUTE_CONFIG",
    "setup_logging",
    "ParamsGrid",
    "_make_logs",
    "basic_experiment",
    "get_diff",
    "get_exec_params",
    "grid_search",
    "inverse_transform_results",
    "replace_exec_param",
    "seeds_experiment",
    "set_seeds",
    "metrics_summary",
]


if __name__ == "__main__":
    setup_logging(root_only=False)
    logger = logging.getLogger(__name__)

    basic_experiment(
        exec_config=EXECUTE_CONFIG, make_plot_bn_dag=False, make_plot_prep_data=True
    )
