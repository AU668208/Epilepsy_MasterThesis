from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison_window(
    ecg_raw: np.ndarray,
    fs: float,
    nk_peaks: Sequence[int],
    lab_peaks_refined: Sequence[int],
    lab_peaks_base: Optional[Sequence[int]] = None,
    *,
    start_s: float,
    dur_s: float = 20.0,
    ecg_clean: Optional[np.ndarray] = None,
    clean_method: str = "neurokit",
    title: Optional[str] = None,
    figsize=(14, 4)
) -> None:
    """
    Plot a comparison window of raw ECG, optionally cleaned ECG, and detected R-peaks.

    Parameters
    ----------
    ecg_raw : array-like
        Raw ECG signal.
    fs : float
        Sampling rate (Hz).
    nk_peaks : sequence of int
        R-peaks detected on the cleaned ECG (e.g., NeuroKit peaks).
    lab_peaks_refined : sequence of int
        LabVIEW R-peaks (time-aligned refined).
    lab_peaks_base : sequence of int, optional
        Original LabVIEW R-peaks (before refinement), optional overlay for diagnostics.
    start_s : float
        Start of the plotting window (in seconds).
    dur_s : float, default 20.0
        Duration of the window (in seconds).
    ecg_clean : array-like, optional
        Cleaned ECG signal to plot. If None, only raw ECG is shown.
    clean_method : str, default "neurokit"
        Label to display for the cleaned signal.
    title : str, optional
        Custom plot title. If None, generated from start/duration.
    figsize : tuple, default (14, 4)
        Matplotlib figure size.
    """

    start = int(start_s * fs)
    end = int((start_s + dur_s) * fs)

    x = np.arange(start, end) / fs
    raw_win = ecg_raw[start:end]

    # Prepare peaks within the window
    nk_loc = [p for p in nk_peaks if start <= p < end]
    nk_loc = np.array(nk_loc) - start

    lab_refined_loc = [p for p in lab_peaks_refined if start <= p < end]
    lab_refined_loc = np.array(lab_refined_loc) - start

    lab_base_loc = None
    if lab_peaks_base is not None:
        lab_base_loc = [p for p in lab_peaks_base if start <= p < end]
        lab_base_loc = np.array(lab_base_loc) - start

    fig, ax = plt.subplots(figsize=figsize)

    # Raw ECG
    ax.plot(x, raw_win, color="gray", alpha=0.75, label="ECG (raw)")

    # Cleaned ECG (optional)
    if ecg_clean is not None:
        clean_win = ecg_clean[start:end]
        ax.plot(x, clean_win, color="steelblue", alpha=0.9,
                label=f"ECG (cleaned: {clean_method})")

    # NK peaks
    ax.scatter(x[nk_loc], raw_win[nk_loc], s=30, color="orange", zorder=4,
               label="NK peaks (cleaned)")

    # LabVIEW refined peaks
    ax.scatter(x[lab_refined_loc], raw_win[lab_refined_loc], 
               s=40, color="green", marker="^", zorder=5,
               label="LabVIEW peaks (refined)")

    # LabVIEW base peaks (optional)
    if lab_base_loc is not None:
        ax.scatter(x[lab_base_loc], raw_win[lab_base_loc], 
                   s=30, color="purple", marker="x", zorder=5,
                   label="LabVIEW peaks (base)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [µV]")

    if title is None:
        title = f"Comparison window {start_s:.1f}–{start_s + dur_s:.1f}s"
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
