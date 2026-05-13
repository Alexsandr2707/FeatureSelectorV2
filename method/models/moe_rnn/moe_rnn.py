import pandas as pd
import logging
from typing import Self

from ..base import ModelResults
from ..rnn.rnn import _prepare_data
from .config import MoERNNConfig
from .moe_rnn_model import MoERNNModel
from method.datasets import Dataset, DatasetBundle, ExpandedWindowDB
from method.core.pipeline import BasePipelineStep
from logging_tools.logging_tools import ClassLogger, log_method

logger = logging.getLogger(__name__)


class MoERNN(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(self, config: MoERNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoERNNConfig()
        self.is_fit: bool = False
        self.model: MoERNNModel | None = None

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        if data.has_valid is None:
            raise ValueError("Model haven't got valid data")

        self.log_params("params", self.config)

        # split data for traing
        data = data.copy()
        if self.config.make_test:
            self.log("making test")
            split = self.config.test_frac
            win = ExpandedWindowDB.from_db(
                data,
                nwin=1,
                init_frac=split,
                end_frac=1 - split,
            )
            data = win.get_window(0)

        self.log_params("expert input:", *data.stats())

        # prepare data for model
        (
            X_train_tensor,
            y_train_tensor,
            train_index,
            X_valid_tensor,
            y_valid_tensor,
            valid_index,
        ) = _prepare_data(data, self.config.exp_params.model.lag)

        # train model
        exp_conf = self.config.exp_params

        self.model = MoERNNModel(
            features_in=data.train.X.shape[-1],
            num_experts=self.config.n_exps,
            gate_type=self.config.gate_type,
            lag=exp_conf.model.lag,
            gru=exp_conf.model.gru,
            decay=exp_conf.model.decay,
            lr=exp_conf.model.lr,
            use_scheduler=True,
            min_lr=exp_conf.model.min_lr,
            use_best_model=exp_conf.model.use_best_model,
        )

        self.model.evaluate(
            X_train_tensor,
            y_train_tensor,
            X_valid=X_valid_tensor,
            y_valid=y_valid_tensor,
            train_index=train_index,
            valid_index=valid_index,
            verbose=True,
            batch=exp_conf.trainer.batch,
            epochs=exp_conf.trainer.epochs,
            device="cpu",
            fit_model=True,
            early_stopping_rounds=exp_conf.trainer.early_stoping,
        )

        self.is_fit = True
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        if not self.is_fit or self.model is None:
            raise ValueError("Model not fitted, make it first")

        self.log_params("prediction input:", *data.stats())

        # prepare data for model
        exp_conf = self.config.exp_params
        data = data.copy()
        (
            X_train_tensor,
            y_train_tensor,
            train_index,
            X_valid_tensor,
            y_valid_tensor,
            valid_index,
        ) = _prepare_data(data, self.config.exp_params.model.lag)

        # make predictions
        dict_result = self.model.evaluate(
            X_train_tensor,
            y_train_tensor,
            X_valid=X_valid_tensor,
            y_valid=y_valid_tensor,
            train_index=train_index,
            valid_index=valid_index,
            verbose=True,
            batch=exp_conf.trainer.batch,
            epochs=exp_conf.trainer.epochs,
            device="cpu",
            fit_model=False,
        )

        dumb = pd.DataFrame()
        tt = dict_result["train"]["true"]
        tp = dict_result["train"]["pred"]
        vt = dict_result["valid"]["true"] if dict_result["valid"] is not None else None
        vp = dict_result["valid"]["pred"] if dict_result["valid"] is not None else None

        result = ModelResults.from_df(
            train_true=tt,
            train_pred=tp,
            valid_true=vt if vt is not None else dumb,
            valid_pred=vp if vp is not None else dumb,
        )

        return result
