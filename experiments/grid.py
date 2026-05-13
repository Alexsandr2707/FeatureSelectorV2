import copy
import logging
import time
from itertools import product
from typing import Any

from .basic import basic_experiment
from .utils import make_logs as make_logs_fn

logger = logging.getLogger(__name__)


class ParamsGrid:
    def __init__(self, exec_config: dict):
        self.base_config = exec_config
        self.param_grid: dict[tuple[str, ...], list[Any]] = {}

    def add_param(
        self, path: list[str] | tuple[str, ...], values: list[Any] | tuple[Any, ...]
    ) -> None:
        if isinstance(path, list):
            path = tuple(path)
        if isinstance(values, tuple):
            values = list(values)

        if path in self.param_grid:
            self.param_grid[path].extend(values)
        else:
            self.param_grid[path] = values

    def __iter__(self):
        paths = list(self.param_grid.keys())
        values_lists = [self.param_grid[p] for p in paths]

        for combination in product(*values_lists):
            current_config = copy.deepcopy(self.base_config)

            for path, value in zip(paths, combination):
                self._set_by_path(current_config, path, value)

            yield current_config

    def _set_by_path(self, config: dict, path: tuple[str, ...], value: Any) -> None:
        d = config
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = value


def get_diff(grid: ParamsGrid, current_config: dict) -> dict:
    """Return only changed params."""
    diff = {}
    for path in grid.param_grid.keys():
        val = current_config
        for key in path:
            val = val[key]
        diff[".".join(path)] = val
    return diff


def grid_search(
    params_grid: ParamsGrid,
    make_logs: bool = True,
):
    results = []

    total_combinations = 1
    for values in params_grid.param_grid.values():
        total_combinations *= len(values)

    make_logs_fn(
        make_logs,
        logger,
        logging.INFO,
        "Starting Grid Search: %s combinations found",
        total_combinations,
    )

    task_fn = lambda config: basic_experiment(
        make_plot_prep_data=False,
        make_plot_bn_dag=False,
        make_plot_results=False,
        exec_config=config,
        make_logs=False,
    )

    start_time = time.time()

    for i, config in enumerate(params_grid, 1):
        make_logs_fn(
            make_logs,
            logger,
            logging.INFO,
            "[%s/%s] Starting iteration %s: %s",
            i,
            total_combinations,
            i,
            get_diff(params_grid, config),
        )

        try:
            run_metrics = task_fn(config)
            make_logs_fn(
                make_logs,
                logger,
                logging.INFO,
                "Result Pearson: %.4f",
                run_metrics["Pearson"],
            )
            make_logs_fn(make_logs, logger, logging.DEBUG, "All metrics: \n%s", run_metrics)

        except Exception as e:
            make_logs_fn(
                make_logs, logger, logging.ERROR, "Iteration %s failed: %s", i, e
            )
            run_metrics = None

        results.append({"iteration": i, "config": config, "metrics": run_metrics})

    end_time = time.time()

    make_logs_fn(
        make_logs,
        logger,
        logging.INFO,
        "Grid Search Completed in %s seconds",
        end_time - start_time,
    )

    best_result = max(
        results,
        key=lambda x: x["metrics"]["Pearson"] if x["metrics"] is not None else -1,
    )
    best_config = best_result["config"]
    best_metrics = best_result["metrics"]

    make_logs_fn(
        make_logs,
        logger,
        logging.INFO,
        "Best result Pearson: %.4f",
        best_metrics["Pearson"] if best_metrics is not None else -1,
    )
    make_logs_fn(
        make_logs,
        logger,
        logging.INFO,
        "Best config: %s",
        get_diff(params_grid, best_config),
    )

    return best_result, results
