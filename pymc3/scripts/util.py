from typing import Dict
import numpy as np
from matplotlib import pyplot as plt


def plot_samples(
    x: np.ndarray,
    samples: Dict[str, np.ndarray],
    fig=None,
    ax=None,
):
    fig, ax = (fig, ax) if fig and ax else plt.subplots(figsize=(14, 11))
    print(f"{x.shape=}, {samples['y'].shape=}")
    for y in samples["y"]:
        ax.scatter(x, y, color="blue", alpha=0.04)
    return fig, ax


def plot_pred_distribution(
    x: np.ndarray,
    samples: Dict[str, np.ndarray],
    fig=None,
    ax=None,
):
    fig, ax = (fig, ax) if fig and ax else plt.subplots(figsize=(14, 11))
    y = samples["y"]
    mean = y.mean(0)
    std = y.std(0)
    ax.plot(x, mean, label="Mean prediction")
    ax.fill_between(
        x,
        mean + 2 * std,
        mean - 2 * std,
        alpha=0.4,
        label="95% credible interval",
    )
    return fig, ax
