# TODO: maybe require bug fix with correct Feature selection
# TODO: require install graphviz, xdg-utils to run plot_DAG
# TODO: undefined logs, created by bnlearn/graphviz

import numpy as np
import pandas as pd
import bnlearn as bn
from sklearn.base import TransformerMixin
from contextlib import contextmanager
import os
import sys
import copy
import logging
from typing import Literal

logger = logging.getLogger(__name__)

ScoreType = Literal["bic", "k2", "bdeu", "bds", "aic"]


def has_visual_environment():
    ip = None
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
    except ImportError:
        pass

    if ip is None:
        return False

    return True


@contextmanager
def suppress_stdout_stderr():
    """Контекстный менеджер для полного подавления вывода в консоль."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_depth(matrix, node, depth, out=True, good=None):
    if out:
        goto = matrix.loc[node]
    else:
        goto = matrix[node]

    goto = goto[goto != 0].index

    if good is None:
        good = set()

    if depth == 1:
        return dict(zip(goto, [node for _ in goto]))

    result = {}
    good = good | {node}

    goto = [x for x in goto if x not in good]

    for next_node in goto:
        child = get_depth(matrix, next_node, depth - 1, out, good=good)

        for k, v in child.items():
            if k not in result:
                result[k] = v

    return result


def select_features(model, var, d=4):
    matrix = model["adjmat"]
    store = {}
    for depth in range(1, d + 1):
        elems = list(get_depth(matrix, var, depth).keys())
        store = dict(**store, **dict(zip(elems, np.ones(len(elems)) * depth)))
    return pd.DataFrame(pd.Series(store).sort_values())


def qcut(x, q=4):
    x = x.copy()
    for c in x.columns:
        x.loc[:, c] = pd.qcut(x[c], q, labels=False, duplicates="drop")
    return x


class BayesTransformer(TransformerMixin):
    def __init__(
        self,
        depth: int = 4,
        max_iter: int = 10,
        q: int = 4,
        scoretype: ScoreType = "bic",
    ):
        self.depth = depth
        self.max_iter = max_iter
        self.is_fit = False
        self.q = q
        self.scoretype = scoretype

    def fit(self, X, y):
        assert len(y) == y.size, "y must be 1-dimensional!"
        if type(y) == pd.DataFrame:
            self.var = y.columns[0]
        elif type(y) == pd.Series and hasattr(y, "name"):
            self.var = y.name
        else:
            raise ValueError("Undefined y type")

        df = X.join(y).dropna()
        df = qcut(df, q=(self.q or 4))

        with suppress_stdout_stderr():
            model = bn.structure_learning.fit(
                df,
                methodtype="chow-liu",
                root_node=self.var,
                white_list=None,
                black_list=None,
                bw_list_method="links",
                scoretype=self.scoretype,
                max_iter=self.max_iter,
                n_jobs=-1,
            )
            model_pruned = bn.independence_test(model, df, alpha=0.05, prune=True)

        self.model = model
        self.model_pruned = model_pruned
        self._features = select_features(self.model_pruned, self.var, d=self.depth)
        self._feature_names_out = self._features.index.to_list()
        self.is_fit = True
        return self

    def features(self):
        assert self.is_fit
        return self._features

    def transform(self, X, y=None):
        assert self.is_fit
        cols = [c for c in self._feature_names_out if c in X.columns]
        if y is None:
            return X[cols]
        else:
            return X[cols], y

    def get_feature_names_out(self):
        return self._feature_names_out

    def _get_sub_model(self):
        assert self.is_fit
        model_pruned = self.model_pruned
        features = self.get_feature_names_out() + [self.var]

        sub_model = copy.deepcopy(model_pruned)
        nodes_to_remove = [n for n in sub_model["model"].nodes() if n not in features]
        sub_model["model"].remove_nodes_from(nodes_to_remove)
        sub_model["adjmat"] = sub_model["adjmat"].loc[features, features]

        df_ind = sub_model["independence_test"]
        sub_model["independence_test"] = df_ind[
            df_ind["source"].isin(features) & df_ind["target"].isin(features)
        ].reset_index(drop=True)
        return sub_model

    def plot_DAG(self):
        """Plot DAG of selected features"""
        assert self.is_fit
        sub_model = self._get_sub_model()
        with suppress_stdout_stderr():
            bn.plot(sub_model, edge_labels="pvalue")

    def plot_tree(self):

        if not has_visual_environment():
            logger.log(logging.WARNING, "Can't plot tree, no visual environment")
            return

        sub_model = self._get_sub_model()

        with suppress_stdout_stderr():
            graph = bn.plot_graphviz(sub_model, edge_labels="pvalue")

            from IPython.display import display

            display(graph)
