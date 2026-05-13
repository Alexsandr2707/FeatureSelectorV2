import pandas as pd

HIGHER_IS_BETTER_METRICS: frozenset[str] = frozenset({"Pearson", "R2"})
LOWER_IS_BETTER_METRICS: frozenset[str] = frozenset(
    {
        "MAE",
        "rMSE",
        "MAPE",
        "Pearson (p-value)",
        "Hinge",
    }
)


def metrics_summary(
    metrics: dict[int, pd.Series] | pd.DataFrame,
    stability_penalty: float = 0.5,
) -> pd.DataFrame:
    """Aggregate validation metrics over seeds with a stability-aware score."""
    if isinstance(metrics, dict):
        metrics = pd.DataFrame.from_dict(metrics, orient="index")

    summary = pd.DataFrame(
        {
            "mean": metrics.mean(axis=0),
            "std": metrics.std(axis=0, ddof=0),
        }
    )

    summary["score"] = summary["mean"]

    higher_is_better = [
        metric for metric in HIGHER_IS_BETTER_METRICS if metric in summary.index
    ]
    lower_is_better = [
        metric for metric in LOWER_IS_BETTER_METRICS if metric in summary.index
    ]

    summary.loc[higher_is_better, "score"] = (
        summary.loc[higher_is_better, "mean"]
        - stability_penalty * summary.loc[higher_is_better, "std"]
    )
    summary.loc[lower_is_better, "score"] = (
        summary.loc[lower_is_better, "mean"]
        + stability_penalty * summary.loc[lower_is_better, "std"]
    )

    return summary.loc[:, ["mean", "std", "score"]]
