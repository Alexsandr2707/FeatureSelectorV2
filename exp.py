import numpy as np
from pathlib import Path
import logging
import random
import torch
import time

from logging_tools.config import setup_logging
from method.datasets import DatasetConfig, LoadDatasetStep

from method.viz import plot_results
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

    loader = LoadDatasetStep(DATASET_CONFIG)
    preprocessor_config = PreprocessConfig.from_dict(EXECUTE_CONFIG["preprocess"])
    preprocessor = Preprocessor(preprocessor_config)
    rnn_config = RNNConfig.from_dict(EXECUTE_CONFIG["model"])
    rnn = RNN(rnn_config)

    steps = [("data_loader", loader), ("preprocessor", preprocessor), ("rnn", rnn)]
    full_pipeline = Pipeline(steps=steps, make_logs=False)

    start = time.perf_counter()
    result = full_pipeline.fit_transform(None)
    duration = time.perf_counter() - start
    logger.info(f"Full pipeline execution complited in ({duration:.2f})s")

    # plot_results(result)

    logger.debug("Metrics for Train:")
    logger.debug("\n%s", metrics(**result["train"], cone=0.1))
    logger.info("Metrics for Valid:")
    logger.info("\n%s", metrics(**result["valid"], cone=0.1))
