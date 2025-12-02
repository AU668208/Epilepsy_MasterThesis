import numpy as np
import matplotlib.pyplot as plt
import pywt   # wavelets
from typing import Optional

def cwt_single_segment(ecg, fs, scales=None, wavelet='morl'):
    """
    Compute CWT coefficients for a 1D ECG segment.
    """
    if scales is None:
        # scale range: works well for 512 Hz ECG
        scales = np.arange(1, 256)

    coefficients, frequencies = pywt.cwt(ecg, scales, wavelet, sampling_period=1/fs)
    return coefficients, frequencies


def plot_cwt(ecg, fs, title="", cmap="viridis", vmin=None, vmax=None):
    """
    Plot a single CWT heatmap.
    """
    coef, freqs = cwt_single_segment(ecg, fs)

    plt.imshow(
        np.abs(coef),
        extent=[0, len(ecg)/fs, freqs.min(), freqs.max()],
        cmap=cmap,
        aspect='auto',
        origin='lower',
        vmin=vmin, vmax=vmax,
    )
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.colorbar(label="|CWT Coefficient|")
    plt.tight_layout()


def plot_peri_cwt_triplet(df_peri, fs, patient_id, seizure_id,
                          figsize=(14, 10), save_path=None):
    """
    Plot baseline_near, ictal, and post_near CWT heatmaps for a given seizure.

    df_peri must contain:
        recording_uid, patient_id, seizure_id, role, signal
    """
    # Extract 3 windows
    win_baseline = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "baseline_near")
    ]

    win_ictal = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "ictal")
    ]

    win_post = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "post_near")
    ]

    if len(win_baseline)==0 or len(win_ictal)==0 or len(win_post)==0:
        raise ValueError("Missing one or more peri-ictal windows for this patient/seizure.")

    ecg_base = win_baseline.iloc[0]["signal"]
    ecg_ict  = win_ictal.iloc[0]["signal"]
    ecg_post = win_post.iloc[0]["signal"]

    plt.figure(figsize=figsize)

    # Panel 1: baseline
    plt.subplot(3,1,1)
    plot_cwt(ecg_base, fs, title=f"Patient {patient_id}, Seizure {seizure_id} — Baseline_Near")

    # Panel 2: ictal
    plt.subplot(3,1,2)
    plot_cwt(ecg_ict, fs, title="Ictal")

    # Panel 3: post_near
    plt.subplot(3,1,3)
    plot_cwt(ecg_post, fs, title="Post_Near")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

