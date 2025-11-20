"""
Utilities for comparing R-peaks between LabVIEW (gold standard)
and NeuroKit2 (test detector). Includes:
- pairwise matching with tolerance
- verbose matching (TP/FP/FN extraction)
- FN/FP overview plots
- local region inspection
- FN vs seizure relations
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt


# ----------------------------------------------------------
#   1.  Pairwise matching (base + verbose)
# ----------------------------------------------------------

def match_rpeaks_pairwise(
    gold_idx: np.ndarray,
    test_idx: np.ndarray,
    fs: float,
    tol_ms: float = 40.0,
    known_offset_s: Optional[float] = None,
) -> Dict[str, float]:
    """
    Basic matching: only TP/FP/FN metrics.
    """
    gold = np.sort(np.asarray(gold_idx, dtype=int))
    test = np.sort(np.asarray(test_idx, dtype=int))

    # Apply known offset
    if known_offset_s is not None:
        shift = int(round(known_offset_s * fs))
        test = test - shift

    # Restrict to common region
    lo = max(gold.min(), test.min())
    hi = min(gold.max(), test.max())
    gold = gold[(gold >= lo) & (gold <= hi)]
    test = test[(test >= lo) & (test <= hi)]

    tol = int(round(tol_ms / 1000.0 * fs))

    i = 0
    j = 0
    TP = 0
    FN = 0
    matched_test = np.zeros(len(test), dtype=bool)

    while i < len(gold) and j < len(test):
        dt = test[j] - gold[i]
        if abs(dt) <= tol:
            TP += 1
            matched_test[j] = True
            i += 1
            j += 1
        elif test[j] < gold[i] - tol:
            j += 1
        else:
            FN += 1
            i += 1

    if i < len(gold):
        FN += (len(gold) - i)

    FP = int((~matched_test).sum())

    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1 = 2 * sens * ppv / (sens + ppv) if sens > 0 and ppv > 0 else np.nan

    return dict(TP=TP, FP=FP, FN=FN, Sensitivity=sens, PPV=ppv, F1=f1)


def match_rpeaks_pairwise_verbose(
    gold_idx: np.ndarray,
    test_idx: np.ndarray,
    fs: float,
    tol_ms: float = 40.0,
    known_offset_s: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Same as match_rpeaks_pairwise, but returns actual TP/FN/FP peak indices.
    """
    gold = np.sort(np.asarray(gold_idx, dtype=int))
    test = np.sort(np.asarray(test_idx, dtype=int))

    if known_offset_s is not None:
        shift = int(round(known_offset_s * fs))
        test = test - shift

    lo = max(gold.min(), test.min())
    hi = min(gold.max(), test.max())
    gold = gold[(gold >= lo) & (gold <= hi)]
    test = test[(test >= lo) & (test <= hi)]

    tol = int(round(tol_ms / 1000.0 * fs))

    i = 0
    j = 0
    matched_test = np.zeros(len(test), dtype=bool)
    matched_gold = np.zeros(len(gold), dtype=bool)

    while i < len(gold) and j < len(test):
        dt = test[j] - gold[i]
        if abs(dt) <= tol:
            matched_test[j] = True
            matched_gold[i] = True
            i += 1
            j += 1
        elif test[j] < gold[i] - tol:
            j += 1
        else:
            i += 1

    TP_idx_gold = gold[matched_gold]
    FN_idx_gold = gold[~matched_gold]
    TP_idx_test = test[matched_test]
    FP_idx_test = test[~matched_test]

    TP = len(TP_idx_test)
    FN = len(FN_idx_gold)
    FP = len(FP_idx_test)

    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    ppv = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    f1 = 2 * sens * ppv / (sens + ppv) if sens > 0 and ppv > 0 else np.nan

    return dict(
        TP=TP, FP=FP, FN=FN,
        Sensitivity=sens, PPV=ppv, F1=f1,
        TP_idx_gold=TP_idx_gold,
        FN_idx_gold=FN_idx_gold,
        TP_idx_test=TP_idx_test,
        FP_idx_test=FP_idx_test,
    )


# ----------------------------------------------------------
#   2. FN/FP vs. seizures
# ----------------------------------------------------------

def FN_in_seizures(FN_idx: np.ndarray, seizures: List[Tuple[int, int]]):
    inside = []
    outside = []
    for fn in FN_idx:
        hit = any(s <= fn <= e for (s, e) in seizures)
        (inside if hit else outside).append(fn)
    return np.asarray(inside), np.asarray(outside)


def FN_near_seizures(FN_idx: np.ndarray, seizures, fs, buffer_s=5):
    buf = int(buffer_s * fs)
    inside = []
    far = []
    for fn in FN_idx:
        hit = any((s - buf) <= fn <= (e + buf) for (s, e) in seizures)
        (inside if hit else far).append(fn)
    return np.asarray(inside), np.asarray(far)


# ----------------------------------------------------------
#   3. Plot utilities
# ----------------------------------------------------------

def plot_FP_FN_overview(res, fs, seizures=None):
    FN_idx = np.asarray(res["FN_idx_gold"])
    FP_idx = np.asarray(res["FP_idx_test"])

    t_FN = FN_idx / fs
    t_FP = FP_idx / fs

    plt.figure(figsize=(15, 3))

    plt.vlines(t_FP, 0.0, 0.5, lw=0.5, color="orange", alpha=0.8,
               label=f"FP (N={len(t_FP)})")
    plt.vlines(t_FN, 0.0, 1.0, lw=0.5, color="red", alpha=0.8,
               label=f"FN (N={len(t_FN)})")

    if seizures is not None:
        for s, e in seizures:
            plt.axvspan(s / fs, e / fs, color="blue", alpha=0.2)

    plt.yticks([])
    plt.xlabel("Time [s]")
    plt.title("Overview of FP/FN vs seizures")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_regions(ecg, fs, res, window_s=3.0, max_plots=6, min_time_s=0.0):
    FN_idx = np.asarray(res["FN_idx_gold"])
    TP_idx_gold = np.asarray(res["TP_idx_gold"])
    TP_idx_test = np.asarray(res["TP_idx_test"])
    FP_idx_test = np.asarray(res["FP_idx_test"])

    if min_time_s > 0:
        FN_idx = FN_idx[FN_idx / fs >= min_time_s]

    win = int(window_s * fs)
    n = min(max_plots, len(FN_idx))

    for k in range(n):
        fn = FN_idx[k]
        lo = max(0, fn - win)
        hi = min(len(ecg), fn + win)
        t = np.arange(lo, hi) / fs

        plt.figure(figsize=(12, 4))
        plt.plot(t, ecg[lo:hi], "k", lw=1)

        gtp = TP_idx_gold[(TP_idx_gold >= lo) & (TP_idx_gold < hi)]
        gfn = FN_idx[(FN_idx >= lo) & (FN_idx < hi)]
        ttp = TP_idx_test[(TP_idx_test >= lo) & (TP_idx_test < hi)]
        tfp = FP_idx_test[(FP_idx_test >= lo) & (FP_idx_test < hi)]

        plt.scatter(gtp / fs, ecg[gtp], c="blue", s=30, label="Gold TP")
        plt.scatter(gfn / fs, ecg[gfn], c="red", s=60, marker="o", label="Gold FN")
        plt.scatter(ttp / fs, ecg[ttp], c="green", s=30, marker="x", label="Test TP")
        plt.scatter(tfp / fs, ecg[tfp], c="orange", s=40, marker="x", label="Test FP")

        plt.title(f"Region {k+1}/{n} – FN at t={fn/fs:.2f}s")
        plt.legend()
        plt.tight_layout()
        plt.show()
