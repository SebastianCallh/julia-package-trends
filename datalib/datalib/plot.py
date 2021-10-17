import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({'font.size': 22})

def observations_per_region(df: pd.DataFrame, log_scale: bool = False) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.scatterplot(
        data=df,
        x="Date",
        y="Request count (log10)" if log_scale else "Request count",
        hue="Region",
        style="Client type",
        s=100,
        ax=ax
    )
    ax.set_title("Package requests over time per region and client type")
    ax.set_xticklabels(ax.get_xticks(), rotation = 50, fontsize=16)
    return fig
