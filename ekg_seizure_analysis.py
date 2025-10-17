
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

from scipy.signal import butter, filtfilt, find_peaks
import pywt
import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------
def bandpass_filter(x, fs, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def _ensure_1d(x):
    x = np.asarray(x).squeeze()
    assert x.ndim == 1, "Signal must be 1D"
    return x

# ----------------------------
# R-peak detection (simple, robust default)
# ----------------------------
def detect_r_peaks_simple(x, fs, distance_ms=250, prominence=0.3):
    """
    Very simple peak detector for ECG after bandpass.
    Returns sample indices of detected R-peaks.
    """
    x = _ensure_1d(x)
    xf = bandpass_filter(x, fs, low=5, high=20, order=3)
    distance = int((distance_ms/1000.0) * fs)
    # normalize
    s = (xf - np.median(xf)) / (np.std(xf) + 1e-8)
    peaks, _ = find_peaks(s, distance=distance, prominence=prominence)
    return peaks

# ----------------------------
# RR and HRV helpers
# ----------------------------
def rr_from_peaks(r_peaks, fs):
    r_peaks = np.asarray(r_peaks)
    rr_s = np.diff(r_peaks) / fs
    t_rr = r_peaks[1:] / fs  # timestamp at each RR (aligned to the second beat)
    return t_rr, rr_s

def sliding_poincare(rr, win_beats=60, step_beats=1):
    """
    Compute SD1, SD2 over sliding windows on RR series (seconds).
    Returns center index for each window and arrays SD1, SD2, ratio.
    """
    rr = np.asarray(rr, float)
    n = len(rr)
    centers, sd1, sd2, ratio = [], [], [], []
    for start in range(0, n - win_beats + 1, step_beats):
        seg = rr[start:start+win_beats]
        if len(seg) < 2:
            continue
        rr1 = seg[:-1]; rr2 = seg[1:]
        # Poincaré descriptors
        diff = (rr2 - rr1) / np.sqrt(2.0)
        sd1_val = np.std(diff, ddof=1)
        mean_rr = np.mean(seg)
        sdnn = np.std(seg, ddof=1)
        sd2_val = np.sqrt(max(0.0, 2*sdnn*sdnn - sd1_val*sd1_val))
        ratio_val = sd2_val / (sd1_val + 1e-8)
        centers.append(start + win_beats//2)
        sd1.append(sd1_val); sd2.append(sd2_val); ratio.append(ratio_val)
    return np.array(centers), np.array(sd1), np.array(sd2), np.array(ratio)

def rolling_slope(y, x=None, win=60, step=1):
    """
    Rolling OLS slope (per window). If x is None, x = np.arange(len(y)).
    Returns centers and slopes.
    """
    y = np.asarray(y, float)
    if x is None:
        x = np.arange(len(y), dtype=float)
    out_idx, slopes = [], []
    for start in range(0, len(y) - win + 1, step):
        xx = x[start:start+win]
        yy = y[start:start+win]
        # simple OLS slope
        xm = xx.mean(); ym = yy.mean()
        num = np.sum((xx - xm)*(yy - ym))
        den = np.sum((xx - xm)**2) + 1e-12
        slopes.append(num/den)
        out_idx.append(start + win//2)
    return np.array(out_idx), np.array(slopes)

# ----------------------------
# CWT (scalogram) for time-frequency insight
# ----------------------------
def cwt_scalogram(x, fs, fmin=1.0, fmax=60.0, voices_per_oct=12, wavelet='morl'):
    """
    Compute CWT scalogram using PyWavelets.
    Returns (freqs, times, power).
    """
    x = _ensure_1d(x)
    w = pywt.ContinuousWavelet(wavelet)
    fc = pywt.scale2frequency(w, 1)
    freqs = np.linspace(fmin, fmax, int((fmax - fmin) + 1))
    scales = (fc * fs) / (freqs + 1e-8)
    coefs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0/fs)
    power = np.abs(coefs)**2
    times = np.arange(len(x))/fs
    return freqs, times, power

def plot_scalogram(freqs, times, power, seizure_t0=None, seizure_t1=None, title="CWT scalogram"):
    plt.figure(figsize=(10, 4))
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    plt.imshow(power, extent=extent, origin='lower', aspect='auto')
    if seizure_t0 is not None:
        plt.axvline(seizure_t0, linestyle='--')
    if seizure_t1 is not None:
        plt.axvline(seizure_t1, linestyle='--')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()

# ----------------------------
# DWT band-energy features
# ----------------------------
def dwt_band_energy(x, fs, wavelet='db4', level=6, win_s=10.0, step_s=2.0):
    """
    Sliding-window DWT energy features. Returns dict with:
      - 't_mid': time center per window (s)
      - 'bands': list of (name, f_low, f_high)
      - 'energy': 2D array [n_windows, n_bands]
    Band mapping here is approximate (dyadic). Interpret qualitatively.
    """
    x = _ensure_1d(x)
    N = len(x)
    win = int(win_s * fs)
    step = int(step_s * fs)
    w = pywt.Wavelet(wavelet)

    bands = []
    for k in range(1, level+1):
        f_high = fs / (2**k)
        f_low  = fs / (2**(k+1))
        bands.append((f"D{k}", f_low, f_high))
    bands.append((f"A{level}", 0.0, fs/(2**(level+1))))

    t_mid = []
    energy_rows = []
    for start in range(0, N - win + 1, step):
        seg = x[start:start+win]
        coeffs = pywt.wavedec(seg, w, level=level, mode='symmetric')
        A = coeffs[0]
        Ds = coeffs[1:][::-1]  # D1..D_level
        row = []
        for d in Ds:
            row.append(float(np.sum(d**2)))
        row.append(float(np.sum(A**2)))
        energy_rows.append(row)
        mid = (start + win//2)/fs
        t_mid.append(mid)

    return {
        "t_mid": np.array(t_mid),
        "bands": bands,
        "energy": np.array(energy_rows)
    }

def plot_dwt_energy(dwt_dict, seizure_t0=None, seizure_t1=None, title="DWT band energy (sliding)"):
    t = dwt_dict["t_mid"]
    bands = dwt_dict["bands"]
    E = dwt_dict["energy"]
    plt.figure(figsize=(10, 4))
    En = (E - E.min(axis=0, keepdims=True)) / (E.ptp(axis=0, keepdims=True) + 1e-9)
    plt.imshow(En.T, extent=[t[0], t[-1], 0, len(bands)], origin='lower', aspect='auto')
    if seizure_t0 is not None:
        plt.axvline(seizure_t0, linestyle='--')
    if seizure_t1 is not None:
        plt.axvline(seizure_t1, linestyle='--')
    yticks = np.arange(len(bands)) + 0.5
    ylabels = [b[0] for b in bands]
    plt.yticks(yticks, ylabels)
    plt.xlabel("Time (s)")
    plt.ylabel("Wavelet bands")
    plt.title(title)
    plt.tight_layout()

# ----------------------------
# Tachogram + Poincaré metrics
# ----------------------------
def plot_tachogram(t_rr, rr_s, seizure_t0=None, seizure_t1=None, title="Tachogram (RR) + rolling slope"):
    plt.figure(figsize=(10,4))
    plt.plot(t_rr, rr_s, linewidth=1.0)
    if seizure_t0 is not None:
        plt.axvline(seizure_t0, linestyle='--')
    if seizure_t1 is not None:
        plt.axvline(seizure_t1, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("RR (s)")
    plt.title(title)
    plt.tight_layout()

def plot_poincare(rr_s, seizure_state=None, title="Poincaré plot (RR_{n} vs RR_{n+1})"):
    rr1 = rr_s[:-1]; rr2 = rr_s[1:]
    plt.figure(figsize=(4,4))
    plt.scatter(rr1, rr2, s=4, alpha=0.6)
    plt.xlabel("RR_n (s)")
    plt.ylabel("RR_{n+1} (s)")
    plt.title(title)
    plt.tight_layout()

def sliding_poincare_ratio_series(rr_s, t_rr, win_beats=60, step_beats=5,
                                  seizure_t0=None, seizure_t1=None,
                                  title="Poincaré SD2/SD1 (sliding)"):
    centers, sd1, sd2, ratio = sliding_poincare(rr_s, win_beats=win_beats, step_beats=step_beats)
    t_idx = np.clip(centers, 0, len(t_rr)-1)
    t_series = t_rr[t_idx]
    plt.figure(figsize=(10,4))
    plt.plot(t_series, ratio, linewidth=1.0)
    if seizure_t0 is not None:
        plt.axvline(seizure_t0, linestyle='--')
    if seizure_t1 is not None:
        plt.axvline(seizure_t1, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("SD2/SD1")
    plt.title(title)
    plt.tight_layout()
    return t_series, ratio

# ----------------------------
# Orchestrator for a single around-seizure segment
# ----------------------------
def analyze_segment(x, fs, seizure_t0, seizure_t1, view_pad_s=120.0,
                    wavelet='db4', dwt_level=6,
                    rr_win_beats=60, rr_step_beats=5):
    freqs, times, power = cwt_scalogram(x, fs, fmin=1.0, fmax=min(60.0, fs/2 - 1), wavelet='morl')
    plot_scalogram(freqs, times, power,
                   seizure_t0=seizure_t0, seizure_t1=seizure_t1,
                   title="CWT scalogram (1–60 Hz)")

    dwt_dict = dwt_band_energy(x, fs, wavelet=wavelet, level=dwt_level, win_s=10.0, step_s=2.0)
    plot_dwt_energy(dwt_dict, seizure_t0=seizure_t0, seizure_t1=seizure_t1)

    r_peaks = detect_r_peaks_simple(x, fs)
    t_rr, rr_s = rr_from_peaks(r_peaks, fs)
    plot_tachogram(t_rr, rr_s, seizure_t0=seizure_t0, seizure_t1=seizure_t1)
    t_ratio, ratio = sliding_poincare_ratio_series(rr_s, t_rr,
                                                   win_beats=rr_win_beats, step_beats=rr_step_beats,
                                                   seizure_t0=seizure_t0, seizure_t1=seizure_t1)

    return {
        "cwt": {"freqs": freqs, "times": times, "power_shape": power.shape},
        "dwt": dwt_dict,
        "rr": {"t_rr": t_rr, "rr_s": rr_s},
        "poincare_ratio": {"t": t_ratio, "ratio": ratio},
    }

# ============================
# Memory-safe, windowed analysis
# ============================
def extract_window(x_all, fs, seizure_t0, seizure_t1, pad_pre_s=300.0, pad_post_s=300.0):
    """
    Return a clipped window x_win centered around [seizure_t0, seizure_t1] with padding.
    Also returns: t_start (absolute start time of window), t_end.
    """
    N = len(x_all)
    total_s = N / fs
    t_start = max(0.0, seizure_t0 - pad_pre_s)
    t_end = min(total_s, seizure_t1 + pad_post_s)
    i0 = int(t_start * fs)
    i1 = int(t_end * fs)
    x_win = _ensure_1d(x_all[i0:i1])
    return x_win, t_start, t_end

def cwt_scalogram_downsampled(x, fs, fmin=1.0, fmax=60.0, wavelet='morl',
                              target_fs=120.0, n_freqs=40):
    """
    Memory-safe CWT: 1) decimate to ~target_fs (integer stride), 2) limit #freqs.
    Returns freqs (Hz), times (s), power, and decimation factor.
    """
    decim = max(1, int(np.floor(fs / target_fs)))
    x_ds = x[::decim]
    fs_ds = fs / decim

    fmax_eff = min(fmax, fs_ds/2 - 1)
    freqs = np.linspace(fmin, fmax_eff, int(n_freqs))

    w = pywt.ContinuousWavelet(wavelet)
    fc = pywt.scale2frequency(w, 1)
    scales = (fc * fs_ds) / (freqs + 1e-8)

    coefs, _ = pywt.cwt(x_ds, scales, wavelet, sampling_period=1.0/fs_ds)
    power = np.abs(coefs)**2
    times = np.arange(len(x_ds))/fs_ds
    return freqs, times, power, decim

def analyze_segment_window(x_all, fs, seizure_t0, seizure_t1,
                           pad_pre_s=300.0, pad_post_s=300.0,
                           wavelet='db4', dwt_level=6,
                           rr_win_beats=60, rr_step_beats=5,
                           do_cwt=True, cwt_target_fs=120.0, cwt_n_freqs=40):
    """
    Windowed analysis around seizure to avoid huge memory usage.
    Returns result dict similar to analyze_segment, but for the window only.
    """
    x_win, t_start, t_end = extract_window(x_all, fs, seizure_t0, seizure_t1, pad_pre_s, pad_post_s)
    seiz0_loc = seizure_t0 - t_start
    seiz1_loc = seizure_t1 - t_start

    out = {}

    if do_cwt:
        freqs, times, power, decim = cwt_scalogram_downsampled(
            x_win, fs, fmin=1.0, fmax=60.0, wavelet='morl',
            target_fs=cwt_target_fs, n_freqs=cwt_n_freqs
        )
        plot_scalogram(freqs, times, power,
                       seizure_t0=seiz0_loc, seizure_t1=seiz1_loc,
                       title=f"CWT scalogram (~{int(fs/decim)} Hz, {cwt_n_freqs} freqs)")
        out["cwt"] = {"freqs": freqs, "times": times, "power_shape": power.shape, "decimation": decim}
    else:
        out["cwt"] = None

    dwt_dict = dwt_band_energy(x_win, fs, wavelet=wavelet, level=dwt_level, win_s=10.0, step_s=2.0)
    plot_dwt_energy(dwt_dict, seizure_t0=seiz0_loc, seizure_t1=seiz1_loc)
    out["dwt"] = dwt_dict

    r_peaks = detect_r_peaks_simple(x_win, fs)
    t_rr, rr_s = rr_from_peaks(r_peaks, fs)
    plot_tachogram(t_rr, rr_s, seizure_t0=seiz0_loc, seizure_t1=seiz1_loc)
    t_ratio, ratio = sliding_poincare_ratio_series(rr_s, t_rr,
                                                   win_beats=rr_win_beats, step_beats=rr_step_beats,
                                                   seizure_t0=seiz0_loc, seizure_t1=seiz1_loc)
    out["rr"] = {"t_rr": t_rr, "rr_s": rr_s}
    out["poincare_ratio"] = {"t": t_ratio, "ratio": ratio}
    out["window"] = {"t_start_abs": t_start, "t_end_abs": t_end, "duration_s": t_end - t_start}
    return out
