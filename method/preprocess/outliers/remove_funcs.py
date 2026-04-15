import pandas as pd
from .config import OutlierParams


def remove_global_outliers_iqr(
    s: pd.Series,
    config: OutlierParams,
):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    IQR = q3 - q1
    low = q1 - config.k * IQR
    high = q3 + config.k * IQR
    if config.dtype == "drop":
        s = s.where((low <= s) & (s <= high))
    elif config.dtype == "clip":
        s = s.clip(lower=low, upper=high)
    else:
        raise ValueError("Value of 'remove_type' must be in ['clip', 'drop']")
    return s


def remove_local_outliers_iqr(
    s: pd.Series,
    config: OutlierParams,
) -> pd.Series:
    if config.window is None:
        raise ValueError("Window size must be specified for local outlier removal")

    window = config.window
    q1 = s.rolling(window, center=True).quantile(0.25)
    q3 = s.rolling(window, center=True).quantile(0.75)
    iqr = q3 - q1

    low = q1 - config.k * iqr
    high = q3 + config.k * iqr

    if config.dtype == "drop":
        s = s.where((low <= s) & (s <= high))
    elif config.dtype == "clip":
        s = s.clip(lower=low, upper=high)
    else:
        raise ValueError("Value of 'remove_type' must be in ['clip', 'drop']")

    return s
