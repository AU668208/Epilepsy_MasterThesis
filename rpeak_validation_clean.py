#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R-peak validation pipeline (clean version)
-----------------------------------------
This script consolidates the essential pieces to:
  1) Load ECG from TDMS
  2) Parse LabVIEW RR (.lvm) with comma or dot decimals
  3) (Optionally) normalize/convert timestamps using Arrow + IANA tz
  4) Detect ECG R-peaks (NeuroKit2) and align with LabVIEW R-peaks
  5) Create binary peak trains (0/1), zero-pad as needed to overlay
  6) Compute metrics with lag & tolerance (TP/FP/FN/Sensitivity/PPV/F1)
  7) Plot a quick comparison window

Fill in the PATH variables in __main__ and run.
Requires: pandas, numpy, neurokit2, nptdms, arrow, scipy, matplotlib
"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import arrow
import nptdms
from datetime import datetime, date, time
from typing import Tuple, Dict, Iterable, Optional

import neurokit2 as nk
from scipy.signal import correlate, convolve
import matplotlib.pyplot as plt


# -----------------------------
#         I/O helpers
# -----------------------------

def read_labview_rr(path: str, skiprows: int = 22) -> np.ndarray:
    """
    Read RR intervals (seconds) from LabVIEW .lvm (or text) file.

    - Handles tab-separated with decimal comma (classic LVM) or autodetected separators.
    - Picks the most plausible RR column (median ~0.2–3 s).
    - Converts from ms/us if needed.
    """
    try:
        df = pd.read_csv(path, sep="\t", engine="python", skiprows=skiprows, header=0, decimal=",")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", skiprows=skiprows, header=0)
        # force comma -> dot numeric
        df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce"))

    # choose the RR column
    cols = [c for c in df.columns if str(c).lower() not in ("x_value", "xvalue", "comment")]
    rr = None
    if "Untitled" in df.columns:
        rr = pd.to_numeric(df["Untitled"], errors="coerce").dropna().to_numpy(float)

    if rr is None:
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
            if v.size:
                med = float(np.nanmedian(v))
                if 0.1 < med < 5.0:  # seconds
                    rr = v
                    break
        if rr is None and cols:
            rr = pd.to_numeric(df[cols[0]], errors="coerce").dropna().to_numpy(float)

    if rr is None or rr.size == 0:
        raise ValueError("Could not find an RR column in the provided LabVIEW file.")

    # unit normalization
    med = float(np.nanmedian(rr))
    if med > 10000:  # microseconds
        rr = rr / 1_000_000.0
    elif med > 5:    # milliseconds
        rr = rr / 1_000.0
    return rr


def rr_to_peak_samples(rr_seconds: Iterable[float], fs: float, t0_s: float = 0.0) -> np.ndarray:
    """
    Convert RR (sec) to absolute peak sample indices (int), assuming first peak at t0_s.

    peak_times = [t0, t0 + rr[0], t0 + rr[0]+rr[1], ...]
    sample_idx = round(peak_times * fs)
    """
    rr = np.asarray(rr_seconds, dtype=float).ravel()
    t_peaks = t0_s + np.cumsum(np.insert(rr, 0, 0.0))
    return np.rint(t_peaks * fs).astype(np.int64)


def make_binary_series(peak_samples: np.ndarray, n_samples: Optional[int], left_pad: int = 0, right_pad: int = 0) -> np.ndarray:
    """
    Create a binary series with ones at peak positions, with optional left/right zero-padding.
    If n_samples is None: length will be max(peak)+1 plus padding.
    Negative peaks are ignored; peaks >= length are dropped.
    """
    peaks = np.asarray(peak_samples, dtype=int)
    if n_samples is None:
        length = (int(peaks.max()) + 1 if peaks.size else 0) + left_pad + right_pad
    else:
        length = int(n_samples) + left_pad + right_pad

    x = np.zeros(length, dtype=np.uint8)
    # shift peaks by left_pad for insertion
    shifted = peaks + left_pad
    valid = shifted[(shifted >= 0) & (shifted < length)]
    x[valid] = 1
    return x


# -----------------------------
#   Time parsing & alignment
# -----------------------------

def read_header_datetime_lvm(path: str, default_date_fmt: str = "%Y/%m/%d") -> Optional[datetime]:
    """
    Parse 'Date' and 'Time' from a LabVIEW .lvm header before ***End_of_Header***.
    Returns a naive datetime (no tz) if found; otherwise None.

    'Time' examples: '13:06:19,1816539465369...' or '13:06:19.1816...' or '13:06:19'.
    """
    date_val = None
    time_val = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("***End_of_Header***"):
                break
            low = ls.lower()
            if low.startswith("date\t") and date_val is None:
                date_val = ls.split("\t", 1)[1].strip()
            elif "time" in low and "time_pref" not in low and low.startswith("time\t") and time_val is None:
                time_val = ls.split("\t", 1)[1].strip()

    if not time_val:
        return None

    m = re.match(r"^(\d{2}:\d{2}:\d{2})[,.](\d+)$", time_val)
    if m:
        hhmmss, frac = m.group(1), m.group(2)
        # normalize to microseconds with rounding on the 7th digit
        if len(frac) > 6 and int(frac[6]) >= 5:
            frac6 = str(int(frac[:6]) + 1).zfill(6)
        else:
            frac6 = frac[:6].ljust(6, "0")
        time_norm = f"{hhmmss}.{frac6}"
        t_dt = datetime.strptime(time_norm, "%H:%M:%S.%f")
    else:
        # HH:MM:SS only
        m2 = re.match(r"^\d{2}:\d{2}:\d{2}$", time_val)
        if not m2:
            return None
        t_dt = datetime.strptime(time_val, "%H:%M:%S")

    if date_val:
        fmts = [default_date_fmt, "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in fmts:
            try:
                d = datetime.strptime(date_val, fmt).date()
                return datetime.combine(d, t_dt.time())
            except ValueError:
                continue
        # fallback to time only
        return t_dt
    return t_dt


def extract_tdms_channel(tdms_path: str, group_name: Optional[str] = None, channel_name: Optional[str] = None) -> Tuple[pd.Series, float, Optional[arrow.Arrow]]:
    """
    Load a single ECG channel from TDMS. Returns (signal_series, fs, tdms_start_time_arrow_or_None)

    - If group/channel are None, picks the first numeric channel found.
    - Tries to read sampling rate from channel properties, else falls back to 512 Hz.
    - Attempts to fetch a 'time' property and returns it as an Arrow object (tz-aware if possible).
    """
    tdms = nptdms.TdmsFile.read(tdms_path)
    chosen = None
    fs = None
    tdms_time = None

    # try to find time property at file or channel level
    def find_time(props: dict) -> Optional[arrow.Arrow]:
        if not isinstance(props, dict):
            return None
        for k, v in props.items():
            if "time" in str(k).lower() and v is not None:
                try:
                    a = arrow.get(pd.to_datetime(v))
                    # assume UTC if tz-naive; convert to local later
                    if a.tzinfo is None:
                        a = a.replace(tzinfo="UTC")
                    return a
                except Exception:
                    continue
        return None

    if hasattr(tdms, "properties"):
        tdms_time = find_time(tdms.properties)

    channels = []
    for g in tdms.groups():
        for ch in g.channels():
            try:
                arr = ch[:]
            except Exception:
                arr = ch.data
            if arr is None:
                continue
            if np.asarray(arr).dtype.kind in ("i", "u", "f"):
                channels.append((g.name, ch.name, pd.Series(arr, name=f"{g.name}.{ch.name}"), ch.properties))

    if not channels:
        raise RuntimeError("No numeric channels found in TDMS file.")

    if group_name and channel_name:
        candidates = [t for t in channels if t[0] == group_name and t[1] == channel_name]
        if not candidates:
            raise RuntimeError(f"Channel {group_name}.{channel_name} not found in TDMS.")
        chosen = candidates[0]
    else:
        # pick the first channel named like EKG if present; else first numeric
        ekg_like = [t for t in channels if "ekg" in t[1].lower() or "ecg" in t[1].lower()]
        chosen = ekg_like[0] if ekg_like else channels[0]

    gname, cname, series, props = chosen

    # sampling rate
    srate = None
    for key in ("wf_sample_rate", "sample_rate", "sampling_rate", "Rate", "rate"):
        if key in props and props[key] not in (None, ""):
            srate = props[key]
            break
    try:
        if isinstance(srate, (bytes, bytearray)):
            srate = float(srate.decode())
        elif srate is not None:
            srate = float(srate)
    except Exception:
        srate = None
    fs = srate if srate is not None else 512.0

    # channel-level time if file-level missing
    if tdms_time is None:
        tdms_time = find_time(props)

    return series.rename("ECG"), float(fs), tdms_time


# -----------------------------
#      Peak detection & lag
# -----------------------------

def detect_ecg_peaks(signal: pd.Series, fs: float) -> np.ndarray:
    """Clean ECG and return R-peak indices (global sample indices)."""
    cleaned = nk.ecg_clean(signal.to_numpy(), sampling_rate=int(fs))
    peaks_dict, _ = nk.ecg_peaks(cleaned, sampling_rate=int(fs))
    # NeuroKit returns a binary vector under 'ECG_R_Peaks': 1 at peaks, else 0
    binary = peaks_dict.get("ECG_R_Peaks")
    if binary is None:
        # Older NK versions may return indices elsewhere
        # Fallback: take first array-like found
        for v in peaks_dict.values():
            try:
                binary = np.asarray(v)
                break
            except Exception:
                continue
    idx = np.flatnonzero(np.asarray(binary, dtype=np.uint8))
    return idx


def estimate_lag_samples(x: np.ndarray, y: np.ndarray, fs: float, start_s: float = 100.0, dur_s: float = 10.0) -> int:
    """
    Estimate lag between two signals (y relative to x) via cross-correlation on a window.
    Returns lag in samples (>0 means y is delayed vs x).
    """
    i0 = int(start_s * fs)
    n = int(dur_s * fs)
    xa = x[i0:i0+n].astype(float) - float(np.mean(x[i0:i0+n]))
    ya = y[i0:i0+n].astype(float) - float(np.mean(y[i0:i0+n]))
    c = correlate(ya, xa, mode="full")
    lags = np.arange(-len(xa)+1, len(xa))
    return int(lags[int(np.argmax(c))])


def align_indices(idx: np.ndarray, lag_samples: int) -> np.ndarray:
    """Shift indices by lag (negative to advance, positive to delay), clip to >= 0."""
    out = idx - int(lag_samples)
    return out[out >= 0]


# -----------------------------
#   Event metrics (with tol)
# -----------------------------

def event_metrics_overlap_lag(gold_idx: np.ndarray,
                              test_idx: np.ndarray,
                              fs: float,
                              tol_ms: float = 40.0,
                              max_lag_ms: float = 150.0) -> Dict[str, float]:
    """
    Compute TP/FP/FN/Sensitivity/PPV/F1 after cropping to overlap, estimating a small lag
    (±max_lag_ms) and applying a symmetric tolerance window (±tol_ms) around gold peaks.
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

    # find best small lag
    maxlag = int(round(max_lag_ms/1000.0*fs))
    bestlag = 0; best = -1
    for lag in range(-maxlag, maxlag+1):
        if lag < 0:
            score = int((a[:lag] & b[-lag:]).sum())
        elif lag > 0:
            score = int((a[lag:] & b[:-lag]).sum())
        else:
            score = int((a & b).sum())
        if score > best:
            best, bestlag = score, lag

    # shift after best lag
    if bestlag > 0:
        b2 = b[bestlag:]; a2 = a[:len(b2)]
    elif bestlag < 0:
        a2 = a[-bestlag:]; b2 = b[:len(a2)]
    else:
        a2, b2 = a, b

    tol = int(round(tol_ms/1000.0*fs))
    win = np.ones(2*tol+1, dtype=int)
    TP = int((convolve(a2, win, mode='same') * b2 > 0).sum())
    FP = int(int(b2.sum()) - TP)
    FN = int(int(a2.sum()) - TP)

    sens = TP/(TP+FN) if (TP+FN) > 0 else np.nan
    ppv  = TP/(TP+FP) if (TP+FP) > 0 else np.nan
    f1   = 2*sens*ppv/(sens+ppv) if (sens>0 and ppv>0) else np.nan

    return dict(TP=TP, FP=FP, FN=FN, Sensitivity=sens, PPV=ppv, F1=f1,
                lag_samples=bestlag, tol_samples=tol, N_overlap=len(a2), lo=lo, hi=lo+len(a2))


# -----------------------------
#       Overlay utilities
# -----------------------------

def overlay_with_padding(len_signal: int,
                         lab_samples: np.ndarray,
                         nk_samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build zero-padded binary trains for LabVIEW and NK so they share a common axis.
    - If Lab starts later than ECG, we left-pad Lab with zeros.
    - If Lab starts earlier than ECG, we shift Lab to the right by trimming negatives and left-pad NK if needed.
    - Length is chosen as the max of (len_signal, last peak + a small buffer).

    Returns (lab_bin, nk_bin) as same-length arrays.
    """
    L = int(len_signal)
    lab = np.asarray(lab_samples, dtype=int)
    nk  = np.asarray(nk_samples, dtype=int)

    first_lab = int(lab.min()) if lab.size else 0
    first_nk  = int(nk.min()) if nk.size else 0

    left_pad_lab = 0
    left_pad_nk  = 0

    if first_lab > first_nk:
        # Lab starts later → left-pad Lab
        left_pad_lab = first_lab - first_nk
    elif first_nk > first_lab:
        # NK starts later → left-pad NK
        left_pad_nk = first_nk - first_lab

    # Build tentative length
    end_lab = int(lab.max()) + left_pad_lab if lab.size else 0
    end_nk  = int(nk.max())  + left_pad_nk  if nk.size else 0
    length = max(L, end_lab+1, end_nk+1)

    lab_bin = make_binary_series(lab, n_samples=length - left_pad_lab, left_pad=left_pad_lab, right_pad=0)
    nk_bin  = make_binary_series(nk,  n_samples=length - left_pad_nk,  left_pad=left_pad_nk,  right_pad=0)

    # Crop to same length just in case
    N = min(len(lab_bin), len(nk_bin))
    return lab_bin[:N], nk_bin[:N]


# -----------------------------
#             Main
# -----------------------------

def main():
    # ========== 1) CONFIG ==========
    TDMS_PATH = r"REPLACE_WITH_TDMS_FILE"
    LABVIEW_LVM_PATH = r"REPLACE_WITH_LABVIEW_LVM"
    TDMS_PATH = r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Patients ePatch data\Patient 5\recording 1\Patient 5_1.tdms"
    LABVIEW_LVM_PATH = r"E:\ML algoritme tl anfaldsdetektion vha HRV\LabView-Results\Patient5_1-corrected-rr.lvm"
    FS_FALLBACK = 512.0
    LOCAL_TZ = "Europe/Copenhagen"

    # If the LabVIEW stream (or seizure export) starts e.g. 60s after TDMS start, you can
    # reflect that here. If None, we try to infer from header times if available.
    KNOWN_LAB_INITIAL_SKIP_S: Optional[float] = None  # e.g., 60.0 or None

    # Matching / windowing
    TOL_MS = 40.0
    MAX_LAG_MS = 150.0

    # Plot window (global seconds)
    PLOT_START_S = 10000.0
    PLOT_DUR_S   = 20.0

    # ========== 2) LOAD ECG + TIME ==========
    ecg, fs, tdms_time = extract_tdms_channel(TDMS_PATH)
    if fs is None:
        fs = FS_FALLBACK
    fs = float(fs)

    # Convert TDMS time to local tz for logging
    if tdms_time is not None:
        tdms_local = tdms_time.to(LOCAL_TZ)
        print(f"[TDMS] Start time (local): {tdms_local}")
    else:
        print("[TDMS] No start time property found.")

    # ========== 3) DETECT R-PEAKS FROM ECG ==========
    nk_idx = detect_ecg_peaks(ecg, fs=fs)
    print(f"[NK] Detected {len(nk_idx)} R-peaks from ECG.")

    # (Optional) account for filter lag between raw and cleaned if you're using cleaned for display only.
    # Here we work directly on global sample indices, so we skip waveform alignment.

    # ========== 4) LABVIEW RR -> PEAK SAMPLES ==========
    rr = read_labview_rr(LABVIEW_LVM_PATH)
    # Infer offset between TDMS start and LabVIEW first peak, if possible
    lab_header_dt = read_header_datetime_lvm(LABVIEW_LVM_PATH)
    offset_sec = 0.0

    if KNOWN_LAB_INITIAL_SKIP_S is not None:
        offset_sec += float(KNOWN_LAB_INITIAL_SKIP_S)

    if lab_header_dt is not None and tdms_time is not None:
        # Assume Lab header time is wall-clock local time; if naive, localize to LOCAL_TZ
        a_lab = arrow.get(pd.to_datetime(lab_header_dt))
        if a_lab.tzinfo is None:
            a_lab = a_lab.replace(tzinfo=LOCAL_TZ)
        # tdms_time is Arrow already (UTC localized earlier)
        # Compute difference in seconds: Lab - TDMS
        delta_s = (a_lab - tdms_time.to(LOCAL_TZ)).total_seconds()
        offset_sec += float(delta_s)
        print(f"[ALIGN] Estimated offset from headers: {delta_s:.3f} s (accumulated offset now {offset_sec:.3f} s)")
    else:
        print("[ALIGN] Header times missing; using KNOWN_LAB_INITIAL_SKIP_S only." if KNOWN_LAB_INITIAL_SKIP_S else
              "[ALIGN] No header times and no known skip; assuming offset_sec=0.")

    lab_idx = rr_to_peak_samples(rr, fs=fs, t0_s=offset_sec)
    print(f"[LAB] Built {len(lab_idx)} LabVIEW peak samples.")

    # ========== 5) METRICS WITH LAG + TOL ==========
    metrics = event_metrics_overlap_lag(lab_idx, nk_idx, fs=fs, tol_ms=TOL_MS, max_lag_ms=MAX_LAG_MS)
    print("[METRICS]", metrics)

    # ========== 6) BUILD ZERO-PADDED BINARY TRAINS FOR OVERLAY ==========
    lab_bin, nk_bin = overlay_with_padding(len_signal=len(ecg), lab_samples=lab_idx, nk_samples=nk_idx)
    print(f"[BIN] Built binary trains: lab={lab_bin.shape}, nk={nk_bin.shape}")

    # ========== 7) PLOT A WINDOW ==========
    s = int(PLOT_START_S * fs); e = int((PLOT_START_S + PLOT_DUR_S) * fs)
    t = np.arange(s, e) / fs

    # Windowed peaks (relative to s)
    lab_loc = lab_idx[(lab_idx >= s) & (lab_idx < e)] - s
    nk_loc  = nk_idx [(nk_idx  >= s) & (nk_idx  < e)] - s

    # Quick plot (no specific colors)
    plt.figure(figsize=(14, 5))
    plt.plot(t, ecg.to_numpy()[s:e], label="ECG (raw)")
    plt.scatter(t[lab_loc], ecg.to_numpy()[s:e][lab_loc], s=14, label="R (LabVIEW)")
    plt.scatter(t[nk_loc],  ecg.to_numpy()[s:e][nk_loc],  s=14, label="R (NeuroKit)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.title(f"ECG with R-peaks, window {PLOT_START_S}s–{PLOT_START_S+PLOT_DUR_S}s")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # ========== 8) SUMMARY TABLE ==========
    # Percent agreement within tolerance (based on metrics)
    tp = metrics["TP"]; fp = metrics["FP"]; fn = metrics["FN"]
    sens = metrics["Sensitivity"]; ppv = metrics["PPV"]; f1 = metrics["F1"]
    print("\n=== SUMMARY ===")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    print(f"Sensitivity={sens:.4f} | PPV={ppv:.4f} | F1={f1:.4f}")
    print("================\n")


if __name__ == "__main__":
    TDMS_PATH = r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Patients ePatch data\Patient 5\recording 1\Patient 5_1.tdms"
    LABVIEW_LVM_PATH = r"E:\ML algoritme tl anfaldsdetektion vha HRV\LabView-Results\Patient5_1-corrected-rr.lvm"
    print("Edit TDMS_PATH and LABVIEW_LVM_PATH inside main() before running this script.")
    
    main()
