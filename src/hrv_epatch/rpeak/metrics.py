# src/hrv_epatch/rpeak/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.signal import convolve

@dataclass
class MatchSummary:
    TP: int
    FP: int
    FN: int
    Sensitivity: float
    PPV: float
    F1: float
    lag_samples: int
    tol_samples: int
    N_overlap: int
    lo: int
    hi: int

def make_binary_series(peak_samples: np.ndarray, n_samples: int | None,
                       left_pad: int = 0, right_pad: int = 0) -> np.ndarray:
    """
    Build 0/1 series with ones at 'peak_samples'. Negative or out-of-range peaks are ignored.
    If n_samples is None, length becomes max(peak)+1 plus padding.
    """
    peaks = np.asarray(peak_samples, dtype=int)
    if n_samples is None:
        length = (int(peaks.max()) + 1 if peaks.size else 0) + left_pad + right_pad
    else:
        length = int(n_samples) + left_pad + right_pad
    x = np.zeros(length, dtype=np.uint8)
    shifted = peaks + left_pad
    valid = shifted[(shifted >= 0) & (shifted < length)]
    x[valid] = 1
    return x

def event_metrics_overlap_lag(gold_idx: np.ndarray, test_idx: np.ndarray, fs: float,
                              tol_ms: float = 40.0, max_lag_ms: float = 150.0) -> MatchSummary:
    """
    Crop to overlap, search best small lag (±max_lag_ms), apply ±tol_ms window around 'gold' peaks,
    and compute TP/FP/FN and derived metrics.
    """
    gold_idx = np.asarray(gold_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    lo = max(gold_idx.min(), test_idx.min())
    hi = min(gold_idx.max(), test_idx.max())
    if hi <= lo:
        raise ValueError("No temporal overlap between peak sequences.")

    g = gold_idx[(gold_idx >= lo) & (gold_idx < hi)] - lo
    t = test_idx[(test_idx >= lo) & (test_idx < hi)] - lo
    N = int(hi - lo)
    a = np.zeros(N, dtype=np.uint8); a[g] = 1
    b = np.zeros(N, dtype=np.uint8); b[t] = 1

    # brute-force small integer lags
    maxlag = int(round(max_lag_ms / 1000.0 * fs))
    best, bestlag = -1, 0
    for lag in range(-maxlag, maxlag + 1):
        if lag < 0:
            score = int((a[:lag] & b[-lag:]).sum())
        elif lag > 0:
            score = int((a[lag:] & b[:-lag]).sum())
        else:
            score = int((a & b).sum())
        if score > best:
            best, bestlag = score, lag

    # align after best lag
    if bestlag > 0:
        b2 = b[bestlag:]; a2 = a[:len(b2)]
    elif bestlag < 0:
        a2 = a[-bestlag:]; b2 = b[:len(a2)]
    else:
        a2, b2 = a, b

    tol = int(round(tol_ms / 1000.0 * fs))
    win = np.ones(2 * tol + 1, dtype=int)
    TP = int((convolve(a2, win, mode="same") * b2 > 0).sum())
    FP = int(int(b2.sum()) - TP)
    FN = int(int(a2.sum()) - TP)

    sens = TP / (TP + FN) if (TP + FN) else float("nan")
    ppv  = TP / (TP + FP) if (TP + FP) else float("nan")
    f1   = 2 * sens * ppv / (sens + ppv) if (sens > 0 and ppv > 0) else float("nan")

    return MatchSummary(TP, FP, FN, sens, ppv, f1, bestlag, tol, len(a2), int(lo), int(lo + len(a2)))

def overlay_with_padding(len_signal: int, lab_samples: np.ndarray, nk_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build zero-padded binary trains for LabVIEW and NeuroKit so they share a common axis.
    Left-pad whichever starts later; choose a safe common length.
    """
    L = int(len_signal)
    lab = np.asarray(lab_samples, dtype=int)
    nk  = np.asarray(nk_samples, dtype=int)
    first_lab = int(lab.min()) if lab.size else 0
    first_nk  = int(nk.min()) if nk.size else 0

    left_pad_lab = max(first_lab - first_nk, 0)
    left_pad_nk  = max(first_nk  - first_lab, 0)

    end_lab = int(lab.max()) + left_pad_lab if lab.size else 0
    end_nk  = int(nk.max())  + left_pad_nk  if nk.size else 0
    length  = max(L, end_lab + 1, end_nk + 1)

    lab_bin = make_binary_series(lab, n_samples=length - left_pad_lab, left_pad=left_pad_lab)
    nk_bin  = make_binary_series(nk,  n_samples=length - left_pad_nk,  left_pad=left_pad_nk)

    N = min(len(lab_bin), len(nk_bin))
    return lab_bin[:N], nk_bin[:N]
