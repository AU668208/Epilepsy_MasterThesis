
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

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _extract_window(sig: np.ndarray, fs: float, n_samples: int, center_idx: int, half_window_sec: int):
    win = int(fs * half_window_sec)
    i0 = max(0, center_idx - win)
    i1 = min(n_samples, center_idx + win)
    segment = sig[i0:i1]
    t = (np.arange(i0, i1) - center_idx) / fs  # time relative to window center (seconds)
    truncated_left = (center_idx - win) < 0
    truncated_right = (center_idx + win) > n_samples
    return segment, t, i0, i1, truncated_left, truncated_right

def _segment_stats(segment: np.ndarray) -> Dict[str, float]:
    if segment.size == 0:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan, "rms": np.nan, "ptp": np.nan, "median": np.nan}
    return {
        "mean": float(np.mean(segment)),
        "min": float(np.min(segment)),
        "max": float(np.max(segment)),
        "std": float(np.std(segment, ddof=0)),
        "rms": float(np.sqrt(np.mean(segment**2))),
        "ptp": float(np.ptp(segment)),            # peak-to-peak = max - min
        "median": float(np.median(segment))
    }

def analyze_shifted_windows(sig: np.ndarray,
                            meta,
                            ann_aligned: pd.DataFrame,
                            use: str = "eeg_onset_idx",
                            half_window_sec: int = 300,
                            shifts_sec: Optional[List[int]] = None,
                            plot: bool = True) -> pd.DataFrame:
    """
    For each seizure, compute stats for a ±half_window_sec window centered at (onset + shift).
    shifts_sec: list of second offsets relative to onset (default: [-7200,-3600,0,3600,7200]).
    use: which onset index column to use ("eeg_onset_idx" or "klin_onset_idx").
    Returns a DataFrame with stats rows.
    """
    if shifts_sec is None:
        shifts_sec = [-7200, -3600, 0, 3600, 7200]

    rows = []
    for _, row in ann_aligned.iterrows():
        # choose onset
        center_idx = row.get(use)
        # fallback if requested onset is missing
        if pd.isna(center_idx):
            alt = "klin_onset_idx" if use == "eeg_onset_idx" else "eeg_onset_idx"
            center_idx = row.get(alt)
        if pd.isna(center_idx):
            continue
        center_idx = int(center_idx)

        for shift in shifts_sec:
            center_shifted = int(round(center_idx + shift * meta.fs))
            segment, t_rel, i0, i1, trunc_l, trunc_r = _extract_window(sig, meta.fs, meta.n_samples, center_shifted, half_window_sec)
            stats = _segment_stats(segment)
            rows.append({
                "seizure_id": row.get("seizure_id"),
                "onset_source": use if pd.notna(row.get(use)) else ("klin_onset_idx" if use=="eeg_onset_idx" else "eeg_onset_idx"),
                "shift_sec": int(shift),
                "window_label": f"center={shift//3600:+d}h, ±{half_window_sec//60}min",
                "center_sample": int(center_shifted),
                "start_sample": int(i0),
                "end_sample": int(i1),
                "truncated_left": bool(trunc_l),
                "truncated_right": bool(trunc_r),
                **stats
            })

            if plot:
                # One plot per window
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(t_rel, segment)
                ax.axvline(0.0)   # visual marker at the window center
                ax.set_xlabel("Tid relativt til center (s)")
                ax.set_ylabel(meta.units or "Amplitude")
                sid = row.get("seizure_id")
                title = f"Anfald {sid} – {('EEG' if use=='eeg_onset_idx' else 'Klinisk')} onset shift {shift//3600:+d}h (±{half_window_sec//60}min)"
                ax.set_title(title)
                plt.show()

    df = pd.DataFrame(rows)
    # Add convenience: group means across shifts per seizure if needed by user later
    return df
