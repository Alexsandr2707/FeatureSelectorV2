import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from typing import Literal
import scipy.stats as stats

PlotType = Literal["plot", "scatter", "hist", "plot_scatter"]
DEFAULT_FIG_SIZE = (12, 10)


def _plot(
    ax,
    x,
    y,
    plot_type: PlotType,
    color: str = "blue",
    lw=0.7,
    alpha=0.7,
    s=10,
    **kwargs,
):
    if plot_type == "plot":
        ax.plot(x, y, color=color, lw=lw, **kwargs)
    elif plot_type == "scatter":
        ax.scatter(x, y, color=color, s=s, alpha=alpha, **kwargs)
    elif plot_type == "hist":
        ax.hist(y, color=color, **kwargs)
    elif plot_type == "plot_scatter":
        ax.plot(x, y, color=color, lw=lw, **kwargs)
        ax.scatter(x, y, color=color, s=s, alpha=0.5)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")


def plot_data(
    data: pd.DataFrame,
    names: list[str] | None = None,
    index_name: str | None = None,
    plot_type: PlotType = "plot_scatter",
    title: str | None = None,
):
    names = names if names is not None else data.columns.to_list()
    index = data[index_name] if index_name is not None else data.index

    # Calculate fig sizes
    ax_count = len(names)
    ncols = math.ceil(ax_count**0.5)
    nrows = int(ax_count**0.5)
    if ncols * nrows < ax_count:
        nrows += 1

    # Plot all features
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=DEFAULT_FIG_SIZE)
    fig.subplots_adjust(hspace=0.5)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    if ax_count == 1:
        axs = np.array([axs]).reshape((-1, 1))

    for num, name in enumerate(names):
        i, j = divmod(num, ncols)
        x, y = index, data[name]
        _plot(axs[i, j], x, y, plot_type, color="blue")

        if ax_count <= 4:
            axs[i, j].set_title(name)
            axs[i, j].grid(which="major")
            axs[i, j].tick_params("x", rotation=20)
        else:
            short_name = name if len(name) <= 10 else "..." + name[-10:]
            axs[i, j].set_title(short_name)
            axs[i, j].grid(which="major")
            axs[i, j].tick_params("x", rotation=20)

    for num in range(ax_count, nrows * ncols):
        i, j = divmod(num, ncols)
        axs[i, j].set_visible(False)

    plt.show()


def plot_data_compare(
    data_raw: pd.DataFrame,
    data_prep: pd.DataFrame,
    title: str | None = None,
):
    # Plot all features
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    _plot(ax, data_prep.index, data_prep, plot_type="plot", color="blue")
    _plot(ax, data_raw.index, data_raw, plot_type="scatter", color="red", s=5)
    ax.grid(which="major")
    plt.show()


def plot_results(result: dict[str, pd.DataFrame]):
    fig = plt.figure(figsize=DEFAULT_FIG_SIZE)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2, fig=fig)
    ax2 = plt.subplot2grid((2, 2), (1, 0), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (1, 1), fig=fig)

    result["train"]["true"] = result["train"]["true"].dropna()
    result["train"]["pred"] = result["train"]["pred"].dropna()
    result["valid"]["true"] = result["valid"]["true"].dropna()
    result["valid"]["pred"] = result["valid"]["pred"].dropna()

    labels = ["Train True", "Train Pred", "Valid True", "Valid Pred"]
    colors = ["blue", "orange", "green", "red"]

    for label, color in zip(labels, colors):
        l1 = label.split()[0].lower()
        l2 = label.split()[1].lower()

        ax1.plot(result[l1][l2], label=label, color=color, lw=0.7)

    ax1.legend()
    ax1.grid(True)
    ax1.set_title("RNN Predictions vs True Values")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Target Value")

    train_residuals = result["train"]["true"] - result["train"]["pred"]
    valid_residuals = result["valid"]["true"] - result["valid"]["pred"]

    _plot(ax2, result["valid"]["true"], valid_residuals, "scatter", alpha=0.5)
    ax2.axhline(0)  # zero line
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residuals vs Predictions")

    stats.probplot(valid_residuals, dist="norm", plot=ax3)
    ax3.set_title("QQ-Plot of Residuals")

    plt.tight_layout()
    plt.show()
