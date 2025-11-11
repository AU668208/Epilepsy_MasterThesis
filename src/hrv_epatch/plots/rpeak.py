# src/hrv_epatch/plots/rpeak.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Mapping
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Window:
    """Time window in seconds."""
    start_s: float
    dur_s: float

def _slice_for_window(n: int, fs: float, start_s: float, dur_s: float) -> slice:
    i0 = max(0, int(round(start_s * fs)))
    i1 = min(n, i0 + int(round(dur_s * fs)))
    return slice(i0, i1)

def _time_axis(n: int, fs: float, offset_s: float) -> np.ndarray:
    return (np.arange(n) / fs) + offset_s

def _stems(ax, t_idx: np.ndarray, y: np.ndarray, label: str, marker: str):
    """Draw stem markers at given time coordinates using y-values from the raw signal."""
    if t_idx.size == 0:
        return
    ax.stem(
        t_idx, y[t_idx],
        linefmt='-', markerfmt=marker, basefmt=' ',
        label=label, use_line_collection=True
    )

def plot_raw_vs_clean_with_peaks(
    ecg_raw: np.ndarray,
    ecg_clean: np.ndarray,
    fs: float,
    window: Window,
    peaks: Mapping[str, np.ndarray],
    title: Optional[str] = None,
    show: bool = True,
    out_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot raw & cleaned ECG with overlaid peak markers inside a time window.

    Args:
        ecg_raw: 1D raw ECG in microvolts.
        ecg_clean: 1D cleaned ECG (same length as ecg_raw).
        fs: sampling rate in Hz.
        window: Window(start_s, dur_s) specifying the time interval to plot.
        peaks: dict mapping a legend name -> *global* peak sample indices.
               e.g. {
                   "NK peaks (raw)": nk_idx_raw,
                   "NK peaks (cleaned)": nk_idx_clean,
                   "LabVIEW base (raw)": lab_base_idx,
                   "LabVIEW refined (raw)": lab_refined_idx,
               }
        title: optional title.
        show: plt.show() if True.
        out_path: save figure if provided.

    Returns:
        Matplotlib Figure.
    """
    n = len(ecg_raw)
    s = _slice_for_window(n, fs, window.start_s, window.dur_s)
    offset_s = s.start / fs
    t = _time_axis(s.stop - s.start, fs, offset_s)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, ecg_raw[s], label="ECG (raw)")
    ax.plot(t, ecg_clean[s], label="ECG (cleaned: neurokit)")

    # Overlay peaks as stems (clip to window)
    for name, idx in peaks.items():
        idx = np.asarray(idx, dtype=int)
        mask = (idx >= s.start) & (idx < s.stop)
        loc = idx[mask] - s.start
        # time coordinates for stems
        t_loc = t[loc]
        # for stems we use y from the *raw* signal so amplitudes make sense
        _stems(ax, t_idx=t_loc, y=ecg_raw[s], label=name,
               marker='o' if 'NK' in name else ('^' if 'refined' in name else 'x'))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [µV]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", ncol=2)
    if title:
        ax.set_title(title)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


def plot_comparison_window(
    ecg: np.ndarray,
    fs: float,
    nk_idx: Optional[Sequence[int]] = None,
    lab_idx_refined: Optional[Sequence[int]] = None,
    lab_idx_base: Optional[Sequence[int]] = None,
    start_s: float = 0.0,
    dur_s: float = 30.0,
    cleaned: Optional[np.ndarray] = None,
    clean_label: str = "ECG (cleaned: neurokit)",
    title: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Convenience wrapper mimicking the comparison plots from your latest notebook.

    - Plots raw ECG (and optional cleaned).
    - Adds stem markers for NK and LabVIEW peaks (base/refined).
    """
    n = len(ecg)
    s = _slice_for_window(n, fs, start_s, dur_s)
    t = _time_axis(s.stop - s.start, fs, s.start / fs)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ecg[s], label="ECG (raw)")
    if cleaned is not None:
        ax.plot(t, cleaned[s], label=clean_label)

    def mark(idx, name, marker):
        if idx is None:
            return
        idx = np.asarray(idx, dtype=int)
        mask = (idx >= s.start) & (idx < s.stop)
        loc = idx[mask] - s.start
        _stems(ax, t[loc], ecg[s], name, marker)

    mark(nk_idx, "NK peaks", "o")
    mark(lab_idx_refined, "LabVIEW peaks (refined)", "^")
    mark(lab_idx_base, "LabVIEW peaks (base)", "x")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude [µV]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig
