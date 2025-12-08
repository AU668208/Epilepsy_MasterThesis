import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_peri_psd_feature(
    df,
    feature: str,
    role_order: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize=(6, 4),
    show=True,
):
    """
    Plot a peri-ictal PSD-related feature (e.g. QRS_rel, HF_rel, Line_rel) across roles.
    """
    if role_order is None:
        role_order = ["baseline_far", "baseline_near", "ictal", "post_near", "post_far"]

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x="role",
        y=feature,
        order=role_order,
        showfliers=False,
        ax=ax,
    )

    ax.set_title(f"Peri-ictal {feature}")
    ax.set_ylabel(feature)
    ax.set_xlabel("Role")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
