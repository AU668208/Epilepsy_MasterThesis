from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_recording_clusters(
    df_quality_clustered: pd.DataFrame,
    cluster_col: str = "quality_cluster_kmeans",
    feat_x: str = "win_std_median",
    feat_y: str = "frac_noiseburst_windows",
    feat_z: str = "qrs_power_ratio",
    palette: Mapping[str, str] | None = None,
    outdir: Path | None = None,
) -> None:
    """
    Visualiser recording-level klynger i feature-space (3D + 2D projektioner).

    Klynger sorteres efter "signal-kvalitet":
    - høj median qrs_power_ratio
    - lav median frac_noiseburst_windows

    Parametre
    ---------
    df_quality_clustered : DataFrame
        Skal indeholde kolonnerne `cluster_col`, feat_x, feat_y, feat_z.
    cluster_col : str
        Navnet på kolonnen med klynge-ID (fx 'quality_cluster_kmeans').
    feat_x, feat_y, feat_z : str
        Navne på de tre features der plottes.
    palette : dict, optional
        Farve-dict, fx {"primary": ..., "secondary": ..., "tertiary": ...}.
        Hvis None, bruges Matplotlibs standardfarver.
    outdir : Path, optional
        Hvis angivet, gemmes figurerne som PNG-filer i denne mappe.
    """

    if cluster_col not in df_quality_clustered.columns:
        raise KeyError(f"{cluster_col!r} findes ikke i df_quality_clustered")

    if palette is None:
        # fallback, hvis du kalder funktionen uden palette
        palette = {
            "primary": "C0",
            "secondary": "C1",
            "tertiary": "C2",
        }

    # --- 1) Klynge-"kvalitet" og sortering ---
    cluster_stats = (
        df_quality_clustered
        .groupby(cluster_col)
        .agg(
            qrs_med=("qrs_power_ratio", "median"),
            noise_med=("frac_noiseburst_windows", "median"),
        )
    )

    ordered_clusters = (
        cluster_stats
        .sort_values(["qrs_med", "noise_med"], ascending=[False, True])
        .index.tolist()
    )

    # navne og farver i kvalitets-rækkefølge
    labels_ordered = [
        "Cluster A (high signal quality)",
        "Cluster B (intermediate quality)",
        "Cluster C (higher noise level)",
    ]
    color_ordered = [
        palette["primary"],   # bedste
        palette["secondary"],  # mellem
        'gray', # mest støj
    ]

    name_map: dict[int, str] = {}
    color_map: dict[int, str] = {}
    for k, name, col in zip(ordered_clusters, labels_ordered, color_ordered):
        name_map[k] = name
        color_map[k] = col

    # gem navne i DF hvis du vil bruge dem senere
    df_quality_clustered["cluster_named"] = df_quality_clustered[cluster_col].map(name_map)

    # ---------- 3D-plot ----------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    for k in ordered_clusters:
        df_k = df_quality_clustered[df_quality_clustered[cluster_col] == k]
        ax.scatter(
            df_k[feat_x],
            df_k[feat_y],
            df_k[feat_z],
            s=40,
            alpha=0.8,
            label=name_map[k],
            color=color_map[k],
        )

    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.set_zlabel(feat_z)
    ax.set_title("Recording-level clusters in feature space")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if outdir is not None:
        outpath3d = outdir / "recording_clusters_3d_quality_sorted.png"
        fig.savefig(outpath3d, dpi=300, bbox_inches="tight")
        print("Saved:", outpath3d)

    plt.show()

    # ---------- 2D-projektioner ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    pairs = [
        (feat_x, feat_z),
        (feat_x, feat_y),
        (feat_z, feat_y),
    ]
    titles = [
        f"{feat_x} vs {feat_z}",
        f"{feat_x} vs {feat_y}",
        f"{feat_z} vs {feat_y}",
    ]

    for ax, (fx, fy), title in zip(axes, pairs, titles):
        for k in ordered_clusters:
            df_k = df_quality_clustered[df_quality_clustered[cluster_col] == k]
            ax.scatter(
                df_k[fx],
                df_k[fy],
                s=35,
                alpha=0.85,
                color=color_map[k],
                label=name_map[k]
            )
        ax.set_title(title)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)

    # --- legend right under plots ---
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),  # ← bring legend closer
        ncol=3,
        frameon=True
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))  # ← reduce bottom padding
    plt.show()