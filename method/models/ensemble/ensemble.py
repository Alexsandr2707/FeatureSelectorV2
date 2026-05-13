from typing import Any, Self
import logging
import numpy as np
import pandas as pd

from ..base import ModelResults, SplitResults, ModelProtocol
from .config import (
    EnsembleConfig,
    BaseEnsembleMeta,
    get_est,
    meta_model,
    LearningMethodType,
)
from logging_tools.logging_tools import ClassLogger, log_method
from method.datasets import DatasetBundle, ExpandedWindowDB
from method.core.pipeline import BasePipelineStep


def _transform_df_func(
    preds: list[pd.DataFrame],
    y: pd.DataFrame,
    model: Any,
    fit_model: bool = False,
) -> pd.DataFrame:
    gen_index = y.index
    for p in preds:
        gen_index = gen_index.intersection(p.index)

    preds_np = [p.reindex(gen_index).to_numpy().squeeze() for p in preds]
    y = y.reindex(gen_index)

    P_np = np.column_stack(preds_np)
    y_np = y.to_numpy().squeeze()
    if fit_model:
        np_res = model.fit_transform(P_np, y_np)
    else:
        np_res = model.transform(P_np)
    pd_res = pd.DataFrame(np_res, index=y.index)
    return pd_res


class Ensemble(BasePipelineStep[DatasetBundle, ModelResults], ClassLogger):
    def __init__(self, config: EnsembleConfig | None = None) -> None:
        super().__init__()
        self.config = config or EnsembleConfig()
        self.estimators: list[ModelProtocol] = []
        self.meta: BaseEnsembleMeta
        self.is_fit: bool = False

    @log_method()
    def fit(self, data: DatasetBundle) -> Self:
        self.log_params("params", self.config)

        wins: list[DatasetBundle] = []

        # split data for traing
        init_frac = self.config.split_method_params.init_frac
        end_frac = self.config.split_method_params.end_frac
        if self.config.split_method == LearningMethodType.BASIC:
            nwins = 1
            exp_win = ExpandedWindowDB.from_db(
                data, nwin=nwins, init_frac=init_frac, end_frac=end_frac
            )
            wins = [exp_win.get_window(0)] * self.config.n_est
        elif self.config.split_method == LearningMethodType.EXPANDED_WINDOW:
            nwins = self.config.n_est
            exp_win = ExpandedWindowDB.from_db(
                data, nwin=nwins, init_frac=init_frac, end_frac=end_frac
            )
            wins = exp_win.get_windows()
        else:
            raise ValueError("Undefined LearningMethodType", self.config.split_method)

        # train estimators
        est_results: list[SplitResults] = []
        for num in range(self.config.n_est):
            self.log("traing %s estimator", num, level=logging.INFO)
            self.log_params("estimator input", wins[num].stats())
            est = get_est(self.config.est_type, self.config.est_params)
            res = est.fit_transform(wins[num]).join_split()
            self.estimators.append(est)
            est_results.append(res)

        # train meta model
        self.meta = meta_model(self.config.meta_model, self.config.meta_model_params)

        gen_size = min(len(res.true) for res in est_results)
        self.log_params("using general window size", gen_size)

        valid_preds = [r.pred for r in est_results]
        valid_true = est_results[-1].true
        valid_res = _transform_df_func(
            valid_preds, valid_true, self.meta, fit_model=True
        )

        self.is_fit = True
        return self

    @log_method()
    def transform(self, data: DatasetBundle) -> ModelResults:
        if not self.is_fit:
            raise RuntimeError("Model not fitted, make fit first")

        result: ModelResults | None = None
        est_results: list[ModelResults] = []

        for est in self.estimators:
            res = est.transform(data)
            est_results.append(res)

        valid_preds = [r.valid.pred for r in est_results]
        valid_true = est_results[0].valid.true
        valid_res = _transform_df_func(
            valid_preds, valid_true, self.meta, fit_model=False
        )
        train_preds = [r.train.pred for r in est_results]
        train_true = est_results[0].train.true
        train_res = _transform_df_func(
            train_preds, train_true, self.meta, fit_model=False
        )

        result = ModelResults.from_df(
            train_pred=train_res,
            train_true=train_true,
            valid_pred=valid_res,
            valid_true=valid_true,
        )
        return result
