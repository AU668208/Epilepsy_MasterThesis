import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_peri_summary(df, features, role_order=None, figsize=(14, 10), save_path=None):
    if role_order is None:
        role_order = ["baseline_far", "baseline_near", "ictal", "post_near", "post_far"]

    # Filtrér kun features der faktisk findes
    present_features = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print("Advarsel - følgende features findes ikke i df og bliver ignoreret:", missing)
    if not present_features:
        raise ValueError("Ingen af de angivne features findes i df.")

    fig, axes = plt.subplots(len(present_features), 1, figsize=figsize, sharex=True)

    # Hvis kun én feature → axes er ikke en liste
    if len(present_features) == 1:
        axes = [axes]

    for ax, feat in zip(axes, present_features):
        sns.boxplot(
            data=df,
            x="role",
            y=feat,
            order=role_order,
            showfliers=False,
            ax=ax,
        )
        ax.set_title(feat)
        ax.tick_params(axis='x', rotation=25)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, axes


def plot_peri_median_curves(
    df,
    features,
    role_order=None,
    figsize=(12, 6),
    linewidth=2.0,
    markersize=7,
    grid=True,
    normalize=True,          # <--- vigtig: plot relativt til baseline_far
):
    if role_order is None:
        role_order = ["baseline_far", "baseline_near", "ictal", "post_near", "post_far"]

    plt.figure(figsize=figsize)

    # Farve- og markør-cyklus
    colors = sns.color_palette("tab10", n_colors=len(features))
    markers = ["o", "s", "D", "^", "v", "<", ">", "P", "*", "X"]

    for i, feat in enumerate(features):
        med = df.groupby("role")[feat].median().reindex(role_order)

        # Normalisér ift. baseline_far (rolle 0)
        if normalize:
            baseline = med.iloc[0]
            if baseline != 0 and not np.isnan(baseline):
                med = med / baseline

        # Brug fast grøn til rms, ellers standardpalette
        if feat == "rms":
            color = "tab:green"   # eller din egen myGreenDark, hvis du har den
        else:
            color = colors[i]

        plt.plot(
            role_order,
            med.values,
            label=feat,
            color=color,
            marker=markers[i % len(markers)],
            markersize=markersize,
            linewidth=linewidth,
        )

    if grid:
        plt.grid(alpha=0.3)

    plt.xticks(rotation=25)
    plt.title("Median peri-ictal dynamics of selected features", fontsize=14)

    if normalize:
        plt.ylabel("Median (relative to baseline_far)")
    else:
        plt.ylabel("Median value")

    # Markér ictal diskret
    plt.axvline("ictal", color="grey", linestyle="--", alpha=0.4)

    plt.legend(title="Features", fontsize=9)
    plt.tight_layout()
    plt.show()
