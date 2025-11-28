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
import matplotlib.ticker as mticker


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


def add_ecg_grid(
    ax,
    t_start: float,
    t_end: float,
    y_min: float,
    y_max: float,
    big_time: float = 0.20,
    small_time: float = 0.04,
    big_amp: Optional[float] = None,
    small_amp: Optional[float] = None,
    label_step_s: float = 1.0,
):
    """
    Tegn EKG-papirgrid:
      - store lodrette linjer hver big_time (0.20 s)
      - små lodrette linjer hver small_time (0.04 s)
      - tilsvarende i amplituderetningen
      - kun labels på udvalgte x-ticks (label_step_s)
    """

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(y_min, y_max)

    # ---- tid / x-akse ----
    # minor: små kasser (0.04 s)
    minor_locator_x = mticker.MultipleLocator(small_time)
    ax.xaxis.set_minor_locator(minor_locator_x)

    # major: store kasser (0.20 s)
    major_locator_x = mticker.MultipleLocator(big_time)
    ax.xaxis.set_major_locator(major_locator_x)

    # ---- amplitude / y-akse ----
    if big_amp is None:
        big_amp = (y_max - y_min) / 8.0
    if small_amp is None:
        small_amp = big_amp / 5.0

    minor_locator_y = mticker.MultipleLocator(small_amp)
    major_locator_y = mticker.MultipleLocator(big_amp)
    ax.yaxis.set_minor_locator(minor_locator_y)
    ax.yaxis.set_major_locator(major_locator_y)

    # Grid: både major og minor
    ax.grid(which="major", linestyle="-", linewidth=0.7, color="red", alpha=0.6)
    ax.grid(which="minor", linestyle="-", linewidth=0.3, color="red", alpha=0.3)


    # Kun x-labels hver label_step_s
    if label_step_s is not None:
        def format_x(value, pos):
            # value i sekunder; vis kun labels ved multipla af label_step_s
            if abs((value / label_step_s) - round(value / label_step_s)) < 1e-3:
                return f"{value:.1f}"
            return ""

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_x))

    ax.set_axisbelow(False)




def plot_ecg_window_with_algorithms(
    ecg: np.ndarray,
    fs: float,
    center_s: float,
    window_s: float,
    peaks_alg1: np.ndarray,
    peaks_alg2: np.ndarray,
    label_alg1: str = "Algoritme 1",
    label_alg2: str = "Algoritme 2",
    cleaned_ecg: Optional[np.ndarray] = None,
    big_time: float = 0.20,
    small_time: float = 0.04,
    big_amp: Optional[float] = None,
    small_amp: Optional[float] = None,
    offset_alg1_s: float = 0.0,
    offset_alg2_s: float = 0.0,
    label_step_s: float = 1.0,
    units_per_mV: Optional[float] = None,
    mv_range: float = 2.0,
    show_absolute_time: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
    sample_id: Optional[str] = None,  # <--- NYT
):
    """
    Plot et EKG-udsnit på grid med to algoritmers R-peaks.

    peaks_alg1/2 er sample-indekser i deres egen tidsakse.
    offset_alg1_s/offset_alg2_s kan bruges til at flytte peaks horisontalt,
    så de aligner med ecg-signalet (positiv værdi = senere i tid).
    """

    # ---- justér peaks for evt. offset (uden at ændre original-arrays) ----
    shift1 = int(round(offset_alg1_s * fs))
    shift2 = int(round(offset_alg2_s * fs))
    peaks1 = np.asarray(peaks_alg1, dtype=int) + shift1
    peaks2 = np.asarray(peaks_alg2, dtype=int) + shift2

    center_sample = int(round(center_s * fs))
    half_win = int(round(window_s * fs))
    lo = max(0, center_sample - half_win)
    hi = min(len(ecg), center_sample + half_win)

    # Absolut tid eller relativ tid fra vinduets start
    if show_absolute_time:
        t = np.arange(lo, hi) / fs
    else:
        t = (np.arange(lo, hi) - lo) / fs   # 0 → window_s*2

    seg = ecg[lo:hi]

    if units_per_mV is not None:
        # Fast EKG-lignende skala:
        # center omkring median (så baseline wander ikke “flytter” vinduet)
        y_center = float(np.median(seg))
        half_span = units_per_mV * mv_range     # fx 2.0 mV → ±2 mV
        y_min = y_center - half_span
        y_max = y_center + half_span

        # 0.5 mV per stor rude, 0.1 mV per lille rude
        big_amp = units_per_mV * 0.5
        small_amp = units_per_mV * 0.1
    else:
        # gammel auto-skalering
        y_min = float(seg.min())
        y_max = float(seg.max())
        y_pad = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
        y_min -= y_pad
        y_max += y_pad


    fig, ax = plt.subplots(figsize=(14, 3.5))

    # Grid først
    add_ecg_grid(
        ax,
        t_start=t[0],
        t_end=t[-1],
        y_min=y_min,
        y_max=y_max,
        big_time=big_time,
        small_time=small_time,
        big_amp=big_amp,
        small_amp=small_amp,
        label_step_s=label_step_s,   # ny parameter, se nedenfor
    )

    # selve signalet
    ax.plot(t, seg, color="black", linewidth=1.0, label="ECG")

    if cleaned_ecg is not None:
        seg_clean = cleaned_ecg[lo:hi]
        ax.plot(t, seg_clean, color="gray", linewidth=0.8, alpha=0.7, label="Cleaned ECG")

    # Algoritme 1 – blå cirkel
    p1 = peaks1[(peaks1 >= lo) & (peaks1 < hi)]
    if len(p1) > 0:
        if show_absolute_time:
            x1 = p1 / fs
        else:
            x1 = (p1 - lo) / fs   # relativ tid

        ax.scatter(
            x1,
            ecg[p1],
            s=90,
            marker="o",
            facecolors="cyan",
            edgecolors="black",
            linewidths=1.5,
            zorder=6,
            label=label_alg1,
        )

    # Algoritme 2 – orange kryds
    p2 = peaks2[(peaks2 >= lo) & (peaks2 < hi)]
    if len(p2) > 0:
        if show_absolute_time:
            x2 = p2 / fs
        else:
            x2 = (p2 - lo) / fs

        ax.scatter(
            x2,
            ecg[p2],
            s=110,
            marker="x",
            color="orange",
            linewidths=2.0,
            zorder=7,
            label=label_alg2,
        )



    ax.set_xlabel("Tid [s]")
    ax.set_ylabel("Amplitude")
    if show_absolute_time:
        title = f"ECG-udsnit omkring t = {center_s:.2f} s (±{window_s:.1f} s)"
    else:
        title = f"ECG-udsnit (0–{2*window_s:.1f} s)"

    if sample_id is not None:
        title = f"{title} – {sample_id}"   # <-- TILFØJ SAMPLE-ID

    ax.set_title(title)

    leg = ax.legend(
        loc="lower right",        # <-- NY PLACERING
        frameon=True,
        fontsize=10,
    )

    # Gør baggrunden let transparent så signalet kan ses igennem
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Interaktiv visning
    if show:
        plt.show()

    # Håndtering af fig-livscyklus
    if return_fig:
        return fig
    else:
        # Vi er færdige med figuren her
        plt.close(fig)
        return None



