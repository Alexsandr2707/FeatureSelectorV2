import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RollingStandardScaler:
    def __init__(self, window: int, eps: float = 1e-8):
        self.window = window
        self.eps = eps
        self.means_ = None
        self.stds_ = None

    def fit(self, X: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        if self.means_ is not None or self.stds_ is not None:
            logger.warning(
                "Transforming with fitted RollingStandardScaler, it is rewrite statistics"
            )

        df = pd.DataFrame(X)
        means = df.rolling(self.window).mean().fillna(0)
        means.iloc[: self.window] = df.iloc[: self.window].mean()
        stds = df.rolling(self.window).std().fillna(1)
        stds.iloc[: self.window] = df.iloc[: self.window].std()
        stds = stds.where(stds >= self.eps, 1.0)

        self.means_ = means
        self.stds_ = stds
        res = (df - self.means_) / self.stds_
        return res

    def inverse_transform(self, X_scaled: pd.DataFrame):
        if self.means_ is None or self.stds_ is None:
            raise ValueError("Call fit first")
        if not X_scaled.index.equals(self.means_.index):
            raise ValueError("Index mismatch between X and fitted statistics")

        return X_scaled * self.stds_ + self.means_

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def unfitted_copy(self):
        return self.__class__(window=self.window, eps=self.eps)


class RollingRobustScaler:
    def __init__(self, window: int, eps: float = 1e-8):
        self.window = window
        self.eps = eps

        self.q1_ = None
        self.q3_ = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X: pd.DataFrame):

        return self

    def transform(self, X: pd.DataFrame):
        if (
            self.q1_ is not None
            or self.q3_ is not None
            or self.center_ is not None
            or self.scale_ is not None
        ):
            logger.warning(
                "Transforming with fitted rolling scaler, it is rewrite statistics"
            )

        df = pd.DataFrame(X)

        q1 = df.rolling(self.window).quantile(0.25)
        q1.iloc[: self.window] = df.iloc[: self.window].quantile(0.25)
        q1 = q1.ffill()

        q3 = df.rolling(self.window).quantile(0.75)
        q3.iloc[: self.window] = df.iloc[: self.window].quantile(0.75)
        q3 = q3.ffill()

        iqr = q3 - q1

        center = df.rolling(self.window).median()
        center.iloc[: self.window] = df.iloc[: self.window].median()
        center = center.ffill()

        iqr = iqr.where(iqr >= self.eps, 1.0)

        self.q1_ = q1
        self.q3_ = q3
        self.center_ = center
        self.scale_ = iqr

        return (df - self.center_) / self.scale_

    def inverse_transform(self, X_scaled: pd.DataFrame):
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Call fit first")
        if not X_scaled.index.equals(self.center_.index):
            raise ValueError("Index mismatch between X and fitted statistics")
        return X_scaled * self.scale_ + self.center_

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    def unfitted_copy(self):
        return self.__class__(window=self.window, eps=self.eps)
