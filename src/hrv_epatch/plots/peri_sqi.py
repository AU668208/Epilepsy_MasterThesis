import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

def plot_peri_sqi_score(
    df,
    role_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize=(6,4),
    show=True,
):
    if role_order is None:
        role_order = ["baseline_far", "baseline_near", "ictal", "post_near", "post_far"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x="role",
        y="sqi_score",
        order=role_order,
        showfliers=False,
        ax=ax
    )
    ax.set_title("Peri-ictal SQI Score")
    ax.set_ylabel("SQI score")
    ax.set_xlabel("Role")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
