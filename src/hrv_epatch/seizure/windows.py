
# --- seizure_window_analysis.py ---
# Analyze ±5 min windows around seizure onset, shifted by given offsets (e.g., -2h, -1h, 0, +1h, +2h).
# Works with structured annotations aligned to samples (from ekg_tdms_pipeline_v5_fix).
#
# Usage in your notebook:
#   from seizure_window_analysis import analyze_shifted_windows
#   df = analyze_shifted_windows(sig, meta, ann_aligned, use="eeg_onset_idx")
#
# Returns a pandas DataFrame with stats for each seizure and each shift window.
#
# Notes:
# - No specific matplotlib colors/styles are set.
# - Each window is plotted in its own figure if plot=True.

# src/hrv_epatch/seizure/windows.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class RecordingMeta:
    fs: float
    n_samples: int
    units: str | None = None

def _extract_window(sig: np.ndarray, fs: float, n_samples: int, center_idx: int, half_window_sec: int):
    win = int(fs * half_window_sec)
    i0  = max(0, center_idx - win)
    i1  = min(n_samples, center_idx + win)
    segment = sig[i0:i1]
    t_rel   = (np.arange(i0, i1) - center_idx) / fs
    trunc_l = (center_idx - win) < 0
    trunc_r = (center_idx + win) > n_samples
    return segment, t_rel, i0, i1, trunc_l, trunc_r

def _segment_stats(segment: np.ndarray) -> Dict[str, float]:
    if segment.size == 0:
        return {k: float("nan") for k in ("mean","min","max","std","rms","ptp","median")}
    return {
        "mean": float(np.mean(segment)),
        "min":  float(np.min(segment)),
        "max":  float(np.max(segment)),
        "std":  float(np.std(segment, ddof=0)),
        "rms":  float(np.sqrt(np.mean(segment**2))),
        "ptp":  float(np.ptp(segment)),
        "median": float(np.median(segment)),
    }

def analyze_shifted_windows(
    sig: np.ndarray,
    meta: RecordingMeta,
    ann_aligned: pd.DataFrame,
    use: str = "eeg_onset_idx",
    half_window_sec: int = 300,
    shifts_sec: Optional[List[int]] = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    For each seizure: compute stats for a ±half_window_sec window centered at (onset + shift).
    'use' chooses onset column, fallback to the other if missing.
    """
    if shifts_sec is None:
        shifts_sec = [-7200, -3600, 0, 3600, 7200]

    rows: List[Dict] = []
    for _, row in ann_aligned.iterrows():
        center_idx = row.get(use)
        if pd.isna(center_idx):
            alt = "klin_onset_idx" if use == "eeg_onset_idx" else "eeg_onset_idx"
            center_idx = row.get(alt)
        if pd.isna(center_idx):
            continue
        center_idx = int(center_idx)

        for shift in shifts_sec:
            c_shift = int(round(center_idx + shift * meta.fs))
            seg, t_rel, i0, i1, trunc_l, trunc_r = _extract_window(sig, meta.fs, meta.n_samples, c_shift, half_window_sec)
            stats = _segment_stats(seg)
            rows.append({
                "seizure_id": row.get("seizure_id"),
                "onset_source": use if pd.notna(row.get(use)) else ("klin_onset_idx" if use == "eeg_onset_idx" else "eeg_onset_idx"),
                "shift_sec": int(shift),
                "window_label": f"center={shift//3600:+d}h, ±{half_window_sec//60}min",
                "center_sample": int(c_shift),
                "start_sample": int(i0),
                "end_sample": int(i1),
                "truncated_left": bool(trunc_l),
                "truncated_right": bool(trunc_r),
                **stats,
            })

            if plot:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(t_rel, seg)
                ax.axvline(0.0)
                ax.set_xlabel("Time relative to center (s)")
                ax.set_ylabel(meta.units or "Amplitude")
                sid = row.get("seizure_id")
                ax.set_title(f"Seizure {sid} – shift {shift//3600:+d}h (±{half_window_sec//60}min)")
                plt.show()

    return pd.DataFrame(rows)

