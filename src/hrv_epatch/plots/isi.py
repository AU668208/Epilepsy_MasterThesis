import numpy as np
import matplotlib.pyplot as plt
from src.hrv_epatch.plots.plotstyle import palette
import pandas as pd

def plot_isi_per_patient(df_isi: pd.DataFrame, use_log: bool = True):
    """
    Visualiser within-recording ISI per patient som punkter.
    Y-akse i timer. Log-skala med fornuftige ticks baseret på data.
    """
    if df_isi.empty:
        print("No ISI data to plot.")
        return

    patients = sorted(df_isi["patient_id"].unique())
    x_pos = {pid: i for i, pid in enumerate(patients)}

    # Sørg for at have en kolonne i timer
    if "isi_hours" not in df_isi.columns:
        df_isi = df_isi.copy()
        df_isi["isi_hours"] = df_isi["isi_seconds"] / 3600.0

    y = df_isi["isi_hours"].values
    y_pos = y[y > 0]  # sikkerhed ift. log

    y_min = y_pos.min()
    y_max = y_pos.max()

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    # ét punkt per ISI, med lidt jitter i x
    for pid in patients:
        d = df_isi[df_isi["patient_id"] == pid]
        x = np.full(len(d), x_pos[pid], dtype=float)
        x += (np.random.rand(len(d)) - 0.5) * 0.3  # jitter
        ax.scatter(
            x,
            d["isi_hours"],
            s=25,
            alpha=1,
            color=palette["primary"],
        )

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(patients)
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Inter-seizure interval (hours)")

    if use_log:
        ax.set_yscale("log")

        # sæt grænser tættere på data
        ax.set_ylim(y_min * 0.7, y_max * 1.3)

        # lav pæne, håndplukkede ticks i timer
        candidate_ticks = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96])
        ticks = candidate_ticks[(candidate_ticks >= y_min*0.7) & (candidate_ticks <= y_max*1.3)]
        if len(ticks) > 0:
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])

    ax.set_title("Within-enrollment inter-seizure intervals per patient")
    plt.tight_layout()
    plt.show()