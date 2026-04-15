import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import cast
from rich.pretty import pretty_repr
import random
import torch


from logging_tools.config import setup_logging
from method.datasets import Dataset, DatasetConfig, DatasetLoader
from method.viz import plot_data, plot_results
from method.preprocess.config import PreprocessConfig
from method.preprocess.preprocess import Preprocessor
from data_configs.base_raw_data import EXECUTE_CONFIG
from method.core.pipeline import Pipeline
from method.models.rnn.rnn import RNNConfig, RNN
from method.metrics import metrics

logger = logging.getLogger(__name__)

CONFIGS_PATH = Path("configs")
DATASET_CONFIG = DatasetConfig.from_dict(EXECUTE_CONFIG["dataset"])

# Determimate seeds

torch.manual_seed(0)
np.random.seed(42)
random.seed(42)


if __name__ == "__main__":
    setup_logging()
    logging.basicConfig(level=logging.INFO)

    data = DatasetLoader(DATASET_CONFIG).load()

    # EXECUTE_CONFIG["preprocess"]["splitter"]["enabled"] = False
    preprocess_config = PreprocessConfig.from_dict(EXECUTE_CONFIG["preprocess"])
    preprocessor = Preprocessor(preprocess_config)

    # data = preprocessor.fit_transform(data)
    # data = cast(Dataset, data)
    # plot_data(data.X.dropna(), plot_type="plot")
    # plot_data(data.y.dropna(), plot_type="plot")

    rnn_config = RNNConfig.from_dict(EXECUTE_CONFIG["model"])
    rnn = RNN(rnn_config)

    steps = [
        ("preprocess", preprocessor),
        ("rnn", rnn),
    ]
    full_pipeline = Pipeline(steps)

    result = full_pipeline.fit_transform(data)
    # result_score = metrics(**result["valid"], cone=0)["MAE"]

    logger.info("\nMetrics for Train:")
    logger.debug(metrics(**result["train"], cone=0.0))
    logger.info("\nMetrics for Valid:")
    logger.info(metrics(**result["valid"], cone=0))

    # plot_results(result)

    # param_grid = {
    #     "filter": {
    #         "y": {
    #             "enabled": [True, False],
    #         },
    #     },
    #     "outliers": {
    #         "X": {
    #             "enabled": True,
    #             "scope": "local",
    #             "params": {"dtype": "clip", "window": [64, 256, 1024], "k": 1.5},
    #         },
    #         "y": {
    #             "enabled": True,
    #             "scope": "local",
    #             "params": {"dtype": "clip", "window": [64, 256, 1024], "k": 1.5},
    #         },
    #     },
    #     "interpolation": {
    #         "X": {
    #             "enabled": True,
    #             "freq": "1h",
    #             "params": {
    #                 "method": "spline",
    #                 "order": 3,
    #                 "limit": 6,
    #                 "limit_area": "inside",
    #                 "limit_direction": "both",
    #             },
    #         },
    #         "y": {
    #             "enabled": True,
    #             "freq": "1h",
    #             "params": {
    #                 "method": "spline",
    #                 "order": 3,
    #                 "limit": [3, 6, 12, 24, 48],
    #                 "limit_area": "inside",
    #                 "limit_direction": "both",
    #             },
    #         },
    #     },
    #     "feature_selector": {
    #         "params": {"pls_depth": [3, 4, 5]},
    #     },
    # }

    # from method.gridsearch import PreprocessGridSearch

    # gs = PreprocessGridSearch(data, preprocess_config, rnn_config, param_grid)
    # best_config, best_result = gs.fit()

    # results_df = gs.get_results_dataframe()
    # results_df.to_csv("gridsearch_result.csv", float_format="%.6f")
    # logger.info(results_df)  # See all results sorted by score
