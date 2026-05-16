import logging
import time
from typing import cast

from data_configs.base_configs.base_raw_data import EXECUTE_CONFIG
from method.datasets import DatasetConfig, LoadDatasetStep
from method.models.model import Model, ModelConfig
from method.preprocess.config import PreprocessConfig
from method.preprocess.preprocess import Preprocessor
from method.viz import plot_prep_data, plot_results

from .utils import (
    get_exec_params,
    inverse_transform_results,
    make_logs as make_logs_fn,
    set_seeds,
)

logger = logging.getLogger(__name__)


def basic_experiment(
    make_plot_prep_data: bool = False,
    inverse_prep_data_scale: bool = False,
    make_plot_bn_tree: bool = False,
    make_plot_bn_dag: bool = False,
    make_plot_results: bool = True,
    inverse_results_scale: bool = True,
    make_logs: bool = True,
    exec_config: dict | None = None,
    cone: float | None = None,
):
    if exec_config is None:
        make_logs_fn(
            make_logs,
            logger,
            logging.WARNING,
            "exec_config is not defined, using default config",
        )
        exec_config = EXECUTE_CONFIG

    seed = get_exec_params(
        "random_seed", exec_config, default_value=0, make_logs=make_logs
    )
    set_seeds(seed)

    loader_config = DatasetConfig.from_dict(exec_config["dataset"])
    loader = LoadDatasetStep(loader_config)
    preprocessor_config = PreprocessConfig.from_dict(exec_config["preprocess"])
    preprocessor = Preprocessor(preprocessor_config)
    model_config = ModelConfig.from_dict(exec_config["model"])
    model = Model(model_config)

    start = time.perf_counter()

    data_raw = loader.fit_transform()
    data_prep = preprocessor.fit_transform(data_raw)

    if make_plot_prep_data:
        plot_prep_data(data_raw, data_prep, inverse_scale=inverse_prep_data_scale)

    if make_plot_bn_tree or make_plot_bn_dag:
        from method.preprocess.feature_selector.bayes import BayesTransformer
        from method.preprocess.feature_selector.selector import FeatureSelector

        selector = cast(FeatureSelector, preprocessor.get_step("feature_selector"))
        bn_model = selector.selector

        if isinstance(bn_model, BayesTransformer):
            if make_plot_bn_tree:
                bn_model.plot_tree()
            if make_plot_bn_dag:
                bn_model.plot_DAG()

    result = model.fit_transform(data_prep)
    duration = time.perf_counter() - start

    make_logs_fn(
        make_logs,
        logger,
        logging.INFO,
        "Full pipeline execution complited in (%.2f)s",
        duration,
    )

    if inverse_results_scale:
        try:
            train_scaler = data_prep.train.y_scaler
            if data_prep.valid is None:
                valid_scaler = lambda x: x
            else:
                valid_scaler = data_prep.valid.y_scaler
            result = inverse_transform_results(result, train_scaler, valid_scaler)
            cone = 3.5 if cone is None else cone
        except Exception as e:
            make_logs_fn(make_logs, logger, logging.WARNING, "Bad scaling", e)
            cone = 0.2 if cone is None else cone
    else:
        cone = 0.2 if cone is None else cone

    if make_plot_results:
        plot_results(result, cone=cone)

    train_metrics = result.train.metrics(cone=cone)
    valid_metrics = result.valid.metrics(cone=cone)

    make_logs_fn(make_logs, logger, logging.DEBUG, "Metrics for Train:")
    make_logs_fn(make_logs, logger, logging.DEBUG, "\n%s", train_metrics)
    make_logs_fn(make_logs, logger, logging.INFO, "Metrics for Valid:")
    make_logs_fn(make_logs, logger, logging.INFO, "\n%s", valid_metrics)

    return valid_metrics
