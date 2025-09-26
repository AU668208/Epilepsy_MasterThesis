#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeppesen ECG Seizure Pipeline — Notebook-friendly API (TDMS-ready)
==================================================================
Public API (importér i din notebook):
- run_pipeline_from_tdms(...)        # læs TDMS via navngiven group/channel
- run_pipeline_from_tdms_auto(...)   # læs TDMS automatisk (første kanal)
- run_pipeline_from_ecg(...)         # kør på rå ECG-array + fs
- run_pipeline_from_rr(...)          # kør kun feature/detektion på RR i sekunder
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
import matplotlib.pyplot as plt

# TDMS er valgfri afhængighed (kun nødvendig for TDMS-funktionerne)
try:
    from nptdms import TdmsFile  # type: ignore
    _HAVE_TDMS = True
except Exception:
    _HAVE_TDMS = False

# -----------------------------
# Filtre & hjælpefunktioner
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5*fs
    from scipy.signal import butter
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(x, fs, low=0.5, high=32.0, order=4):
    b, a = butter_bandpass(low, high, fs, order)
    return filtfilt(b, a, x)

def derivative_filter(x, fs):
    kern = np.array([-1, -2, 0, 2, 1]) * (fs/8.0)
    return np.convolve(x, kern, mode='same')

def moving_average(x, win):
    if win <= 1: return x.copy()
    return np.convolve(x, np.ones(win)/win, mode='same')

def sd1_sd2(rr):
    if len(rr) < 2: return np.nan, np.nan
    d = (rr[:-1] - rr[1:]) / np.sqrt(2.0)
    s = (rr[:-1] + rr[1:]) / np.sqrt(2.0)
    sd1 = np.std(d, ddof=1) if len(d) > 1 else np.std(d)
    sd2 = np.std(s, ddof=1) if len(s) > 1 else np.std(s)
    return sd1, sd2

def lorenz_features(rr):
    sd1, sd2 = sd1_sd2(rr)
    if not np.isfinite(sd1) or not np.isfinite(sd2) or sd1 == 0:
        return np.nan, np.nan, np.nan, np.nan
    T = 4.0 * sd1
    L = 4.0 * sd2
    CSI = L / T
    ModCSI = (L**2) / T
    return T, L, CSI, ModCSI

def rr_from_peaks(peaks, fs):
    if peaks is None or len(peaks) < 2: return np.array([]), np.array([])
    rr = np.diff(peaks) / float(fs)
    t_rr = peaks[1:] / float(fs)
    return rr, t_rr

def median_filter_beats(rr_sec, win_beats=7):
    if len(rr_sec) == 0: return rr_sec.copy()
    pad = win_beats // 2
    rr_pad = np.pad(rr_sec, (pad, pad), mode='edge')
    out = np.zeros_like(rr_sec)
    for i in range(len(rr_sec)):
        out[i] = np.median(rr_pad[i:i+win_beats])
    return out

def sliding_windows_idx(n, w, step=1):
    i = 0
    while i + w <= n:
        yield i, i + w
        i += step

def tachogram_slope(hr_bpm, t_s):
    if len(hr_bpm) < 2: return np.nan
    a, _ = np.polyfit(np.asarray(t_s), np.asarray(hr_bpm), 1)
    return a

def compute_features(rr_sec, t_rr, window_beats=100):
    med_rr = median_filter_beats(rr_sec, 7)
    hr = 60.0 / np.maximum(rr_sec, 1e-6)
    hr_med = 60.0 / np.maximum(med_rr, 1e-6)
    t_cum = t_rr.copy()

    slopes_med = []
    T_list = []; L_list = []; CSI_list = []; ModCSI_list = []; CSI_raw_list = []; idx_end = []
    for s, e in sliding_windows_idx(len(rr_sec), window_beats, 1):
        slopes_med.append(tachogram_slope(hr_med[s:e], t_cum[s:e]))
        T, L, csi, modcsi = lorenz_features(med_rr[s:e])
        T_list.append(T); L_list.append(L); CSI_list.append(csi); ModCSI_list.append(modcsi)
        Tr, Lr, csir, _ = lorenz_features(rr_sec[s:e])
        CSI_raw_list.append(csir)
        idx_end.append(e-1)

    t_end = t_rr[np.array(idx_end, dtype=int)]
    slopes_med = np.asarray(slopes_med, float)
    T_arr = np.asarray(T_list, float); L_arr = np.asarray(L_list, float)
    CSI = np.asarray(CSI_list, float); ModCSI = np.asarray(ModCSI_list, float)
    CSI_raw = np.asarray(CSI_raw_list, float)
    modcsi_filt_times_slope = ModCSI * slopes_med
    csi_times_slope = CSI_raw * slopes_med
    df = pd.DataFrame({
        "beat_idx_end": idx_end,
        "t_end_s": t_end,
        "slope_bpm_per_s": slopes_med,
        "T": T_arr, "L": L_arr,
        "CSI": CSI, "ModCSI": ModCSI, "CSI_raw": CSI_raw,
        "modcsi_filt_times_slope": modcsi_filt_times_slope,
        "csi_times_slope": csi_times_slope,
    })
    return df

def auto_threshold(series: np.ndarray, pct: float = 99.5):
    series = series[np.isfinite(series)]
    return float(np.percentile(series, pct)) if series.size > 0 else np.nan

def detect_events(feature_series, times, thr, min_separation_beats=50):
    mask = np.isfinite(feature_series) & (feature_series >= thr)
    events = []; i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        st = i
        while i < n and mask[i]: i += 1
        en = i
        events.append([st, en])
    merged = []
    for ev in events:
        if not merged: merged.append(ev); continue
        if ev[0] - merged[-1][1] <= min_separation_beats:
            merged[-1][1] = max(merged[-1][1], ev[1])
        else:
            merged.append(ev)
    return merged

# -----------------------------
# R-peak detektorer
# -----------------------------
def rpeaks_simple(ecg, fs):
    bp = bandpass_filter(ecg, fs, 0.5, 32.0, 4)
    der = derivative_filter(bp, fs)
    sqr = der**2
    mwi = moving_average(sqr, max(1, int(round(0.150*fs))))
    thr = np.percentile(mwi, 95) * 0.5
    peaks = []; last = -10**9; refr = int(round(0.250 * fs))
    alpha = 0.01
    for i in range(1, len(mwi) - 1):
        thr = (1 - alpha) * thr + alpha * mwi[i]
        if mwi[i] > thr and mwi[i] > mwi[i-1] and mwi[i] >= mwi[i+1]:
            if i - last >= refr:
                peaks.append(i); last = i
    return np.array(peaks, dtype=int)

@dataclass
class LVParams:
    fs: float = 256.0
    win_s: float = 2.0
    thigh_alpha: float = 0.75
    fwd_bwd_radius_samp: int = 15
    refractory_s: float = 0.25
    delta_thresh_samples: int = 35
    rrshort_n: int = 8
    rrlong_n: int = 34
    rmax_clip_low_s: float = 0.4
    rmax_clip_high_s: float = 1.2

class LabVIEWRpeak:
    """Jeppesen/LabVIEW-agtig R-peak detektor: 2s vinduer, T(high)/T(low), Rmax, DELTA, ±15 lokalisering, 0.25s refr."""
    def __init__(self, params: LVParams):
        self.p = params

    def _window_max_series(self, sig):
        fs = self.p.fs; L = int(round(self.p.win_s * fs))
        rect = np.maximum(sig, 0.0)
        m = []; edges = []
        for st in range(0, len(rect), L):
            en = min(len(rect), st + L)
            m.append(np.max(rect[st:en]) if en > st else 0.0)
            edges.append((st, en))
        return np.array(m), edges

    def _thigh_series(self, window_max):
        alpha = self.p.thigh_alpha
        N = len(window_max); thigh = np.zeros(N)
        for k in range(N):
            if k == 0:
                thigh[k] = alpha * window_max[0]
            else:
                lo = max(0, k - 8)
                ref = window_max[lo:k] if k > 0 else window_max[:1]
                med = np.median(ref) if ref.size > 0 else window_max[k]
                thigh[k] = alpha * med
        return thigh

    def _fwd_bwd_localise(self, x, idx):
        r = self.p.fwd_bwd_radius_samp
        st = max(0, idx - r); en = min(len(x), idx + r + 1)
        if en <= st: return idx
        seg = x[st:en]; off = np.argmax(seg)
        return st + off

    def detect(self, ecg):
        fs = self.p.fs
        bp = bandpass_filter(ecg, fs, 0.5, 32.0, 4)

        wmax, edges = self._window_max_series(bp)
        thigh_series = self._thigh_series(wmax)

        thigh = np.zeros(len(bp))
        for k, (st, en) in enumerate(edges):
            thigh[st:en] = thigh_series[k]
        tlow = 0.4 * thigh

        peaks = []; rr_list = []
        refr = int(round(self.p.refractory_s * fs))
        Lwin = int(round(self.p.win_s * fs))
        last_window_det = False; next_window_edge = Lwin

        i = 0
        while i < len(bp):
            if peaks and i - peaks[-1] < refr:
                i += 1; continue

            thr = thigh[i]; use_tlow = False
            if len(peaks) > 0:
                dt = (i - peaks[-1]) / fs
                rlong = np.median(rr_list[-self.p.rrlong_n:]) if len(rr_list) >= 2 else 0.8
                rmax = np.clip(rlong, self.p.rmax_clip_low_s, self.p.rmax_clip_high_s)
                if dt >= rmax:
                    use_tlow = True

            if not use_tlow:
                if bp[i] > thr and (i == 0 or bp[i-1] <= thr):
                    idx = self._fwd_bwd_localise(bp, i)
                    if (idx == 0 or bp[idx] >= bp[idx-1]) and (idx == len(bp)-1 or bp[idx] >= bp[idx+1]):
                        if len(peaks) == 0 or idx - peaks[-1] >= refr:
                            peaks.append(idx); last_window_det = True
                            if len(peaks) >= 2: rr_list.append((peaks[-1] - peaks[-2]) / fs)
                            i = idx + 1; continue
            else:
                if len(rr_list) >= 3:
                    seg_rr = np.array(rr_list[-self.p.rrlong_n:]) if len(rr_list) >= self.p.rrlong_n else np.array(rr_list)
                    if seg_rr.size >= 5:
                        med = np.median(seg_rr)
                        eps = np.abs(seg_rr - med)
                        if eps.size >= 2:
                            idx_rm = np.argsort(eps)[-2:]; mask = np.ones_like(eps, bool); mask[idx_rm] = False
                            delta_val = np.mean(eps[mask]) if mask.any() else np.mean(eps)
                        else:
                            delta_val = np.mean(eps)
                        delta_samples = delta_val * fs
                        high_var = (delta_samples > self.p.delta_thresh_samples)
                    else:
                        high_var = False
                    lookback_rr = self.p.rrshort_n if high_var else self.p.rrlong_n
                    lookback_s = np.sum(rr_list[-lookback_rr:]) if len(rr_list) >= 1 else 2.0
                else:
                    lookback_s = 2.0
                lookback_samp = int(round(lookback_s * fs))
                st = max(0, i - lookback_samp)
                seg = bp[st:i+1]; tl = tlow[st:i+1]
                cand = np.where(seg >= tl)[0]
                if cand.size > 0:
                    ci = st + cand[np.argmax(seg[cand])]
                    ci = self._fwd_bwd_localise(bp, ci)
                    if len(peaks) == 0 or ci - peaks[-1] >= refr:
                        peaks.append(ci); last_window_det = True
                        if len(peaks) >= 2: rr_list.append((peaks[-1] - peaks[-2]) / fs)
                        i = ci + 1; continue

            if i >= next_window_edge - 1:
                if not last_window_det:
                    pass  # shadow-behaviour placeholder
                last_window_det = False
                next_window_edge += Lwin
            i += 1

        return np.array(sorted(set(peaks)), dtype=int)

# -----------------------------
# TDMS read helpers
# -----------------------------
def read_tdms(tdms_path, group, channel, time_channel=None):
    if not _HAVE_TDMS:
        raise ImportError("nptdms not installed. pip install nptdms")
    td = TdmsFile.read(tdms_path)
    ch = td[group][channel]
    x = ch.data.astype(float)

    if time_channel is not None:
        t = td[group][time_channel].data.astype(float)
        if t.size != x.size:
            raise ValueError("time_channel length != channel length")
        fs = 1.0 / np.median(np.diff(t)) if t.size > 1 else None
        return x, t, fs

    dt = None
    if 'wf_increment' in ch.properties: dt = float(ch.properties['wf_increment'])
    elif 'dt' in ch.properties:         dt = float(ch.properties['dt'])
    else:
        fs_prop = ch.properties.get('SamplingRate', None)
        if fs_prop is not None:
            dt = 1.0 / float(fs_prop)

    if dt is None:
        return x, None, None

    t0 = float(ch.properties.get('wf_start_offset', 0.0))
    t = t0 + np.arange(x.size) * dt
    fs = 1.0 / dt if dt > 0 else None
    return x, t, fs

def _infer_tdms_group_channel(td):
    groups = td.groups()
    if not groups:
        raise ValueError("TDMS contains no groups.")
    g0 = groups[0]
    chans = g0.channels()
    if not chans:
        for g in groups[1:]:
            if g.channels():
                return g.name, g.channels()[0].name
        raise ValueError("TDMS contains no channels.")
    return g0.name, chans[0].name

def read_tdms_auto(tdms_path, fs_override=None, index_path=None):
    if not _HAVE_TDMS:
        raise ImportError("nptdms not installed. pip install nptdms")
    td = TdmsFile.read(tdms_path)
    g_name, ch_name = _infer_tdms_group_channel(td)
    ch = td[g_name][ch_name]
    x = ch.data.astype(float)

    dt = None
    if 'wf_increment' in ch.properties: dt = float(ch.properties['wf_increment'])
    elif 'dt' in ch.properties:         dt = float(ch.properties['dt'])
    elif 'SamplingRate' in ch.properties: dt = 1.0/float(ch.properties['SamplingRate'])

    fs = (fs_override if fs_override is not None else (None if dt is None else 1.0/dt))
    if fs is None:
        raise ValueError("Sampling rate could not be determined; provide fs_override.")

    t_seconds = None
    if dt is not None:
        t_seconds = (0.0 + np.arange(x.size) * dt).astype(float)

    timestamps = None
    wf_start = ch.properties.get("wf_start_time", None)
    if wf_start is not None and dt is not None:
        step_ns = int(round((1.0/fs) * 1e9))
        try:
            start_np = np.datetime64(wf_start, 'ns')
            timestamps = start_np + np.arange(x.size) * np.timedelta64(step_ns, 'ns')
        except Exception:
            timestamps = None

    # valgfrit: forsøg at læse fra sidecar/index
    if timestamps is None and index_path is not None:
        try:
            td_idx = TdmsFile.read(index_path)
            wf_start = None
            for g in td_idx.groups():
                for c in g.channels():
                    if "wf_start_time" in c.properties:
                        wf_start = c.properties["wf_start_time"]
                        break
            if wf_start is not None:
                step_ns = int(round((1.0/fs) * 1e9))
                start_np = np.datetime64(wf_start, 'ns')
                timestamps = start_np + np.arange(x.size) * np.timedelta64(step_ns, 'ns')
        except Exception:
            pass

    return x, t_seconds, fs, timestamps, g_name, ch_name

# -----------------------------
# Core runner & API
# -----------------------------
def _finish(rr_sec, t_rr, out, plots, thr_modcsi_x_slope, thr_csi_x_slope, auto_baseline_beats, auto_pct, window_beats):
    features = compute_features(rr_sec, t_rr, window_beats=window_beats)

    thr_mod = thr_modcsi_x_slope
    thr_csi = thr_csi_x_slope
    if auto_baseline_beats and auto_baseline_beats > window_beats:
        mask = (features["beat_idx_end"].values < auto_baseline_beats)
        if np.any(mask):
            if thr_mod is None: thr_mod = auto_threshold(features.loc[mask, "modcsi_filt_times_slope"].values, auto_pct)
            if thr_csi is None: thr_csi = auto_threshold(features.loc[mask, "csi_times_slope"].values, auto_pct)
    if thr_mod is None: thr_mod = auto_threshold(features["modcsi_filt_times_slope"].values, auto_pct)
    if thr_csi is None: thr_csi = auto_threshold(features["csi_times_slope"].values, auto_pct)

    times = features["t_end_s"].values
    ev_mod = detect_events(features["modcsi_filt_times_slope"].values, times, thr_mod, 50)
    ev_csi  = detect_events(features["csi_times_slope"].values, times, thr_csi, 50)
    all_events = sorted(ev_mod + ev_csi, key=lambda x: x[0])
    merged = []
    for st, en in all_events:
        if not merged: merged.append([st, en]); continue
        if st <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en])

    detections = pd.DataFrame([{
        "win_start_idx": int(st),
        "win_end_idx":   int(en),
        "t_start_s":     float(times[st]),
        "t_end_s":       float(times[min(en-1, len(times)-1)]),
        "dur_s":         float(times[min(en-1, len(times)-1)] - times[st])
    } for st, en in merged])

    if plots and out:
        plt.figure()
        plt.plot(features["t_end_s"].values, features["modcsi_filt_times_slope"].values)
        plt.axhline(thr_mod, linestyle="--")
        plt.xlabel("Time (s)"); plt.ylabel("ModCSI_filt × slope"); plt.title("Feature")
        plt.savefig(f"{out}_modcsi_times_slope.png", dpi=160); plt.close()

        plt.figure()
        plt.plot(features["t_end_s"].values, features["csi_times_slope"].values)
        plt.axhline(thr_csi, linestyle="--")
        plt.xlabel("Time (s)"); plt.ylabel("CSI × slope"); plt.title("Feature")
        plt.savefig(f"{out}_csi_times_slope.png", dpi=160); plt.close()

        if len(rr_sec) > 101:
            rr_win = rr_sec[-100:]
            plt.figure(); plt.scatter(rr_win[:-1], rr_win[1:], s=8)
            plt.xlabel("RR[i] (s)"); plt.ylabel("RR[i+1] (s)"); plt.title("Lorenz (sample)")
            plt.savefig(f"{out}_lorenz.png", dpi=160); plt.close()

    meta = {
        "window_beats": window_beats,
        "auto_baseline_beats": auto_baseline_beats,
        "auto_pct": auto_pct,
        "thr_modcsi_filt_times_slope": thr_mod,
        "thr_csi_times_slope": thr_csi
    }
    return features, detections, meta

def run_pipeline_from_ecg(
    ecg, fs, t: Optional[np.ndarray] = None, target_fs: Optional[float] = None,
    rpeak_mode: str = "labview", window_beats: int = 100,
    thr_modcsi_x_slope: Optional[float] = None, thr_csi_x_slope: Optional[float] = None,
    auto_baseline_beats: int = 0, auto_pct: float = 99.5,
    plots: bool = False, out: Optional[str] = None
) -> Dict[str, Any]:
    """Kør hele pipelinen på rå ECG."""
    if target_fs and abs(target_fs - fs) > 1e-9:
        up = int(round(target_fs)); down = int(round(fs))
        ecg = resample_poly(ecg, up, down)
        fs = float(target_fs)
        t = np.arange(len(ecg)) / fs
    elif t is None:
        t = np.arange(len(ecg)) / fs

    if rpeak_mode == "simple":
        peaks = rpeaks_simple(ecg, fs)
    else:
        peaks = LabVIEWRpeak(LVParams(fs=fs)).detect(ecg)

    rr_sec, t_rr = rr_from_peaks(peaks, fs)
    if len(rr_sec) < max(10, window_beats + 1):
        raise ValueError("Not enough beats for analysis.")

    features, detections, meta = _finish(rr_sec, t_rr, out, plots,
                                         thr_modcsi_x_slope, thr_csi_x_slope,
                                         auto_baseline_beats, auto_pct, window_beats)
    return {"peaks": peaks, "rr_sec": rr_sec, "t_rr": t_rr,
            "features": features, "detections": detections, "meta": meta}

def run_pipeline_from_tdms(
    tdms_path: str, group: str, channel: str, time_channel: Optional[str] = None,
    fs: Optional[float] = None, target_fs: Optional[float] = None,
    rpeak_mode: str = "labview", window_beats: int = 100,
    thr_modcsi_x_slope: Optional[float] = None, thr_csi_x_slope: Optional[float] = None,
    auto_baseline_beats: int = 0, auto_pct: float = 99.5,
    plots: bool = False, out: Optional[str] = None
) -> Dict[str, Any]:
    """Kør pipeline ved at læse TDMS fra navngivet group/channel."""
    if not _HAVE_TDMS:
        raise ImportError("nptdms not installed. pip install nptdms")
    x, t_inferred, fs_inferred = read_tdms(tdms_path, group, channel, time_channel)
    fs = fs if fs is not None else fs_inferred
    if fs is None:
        raise ValueError("Sampling rate unknown. Provide fs or ensure TDMS has timing props or time_channel.")
    return run_pipeline_from_ecg(ecg=x, fs=fs, t=t_inferred, target_fs=target_fs,
                                 rpeak_mode=rpeak_mode, window_beats=window_beats,
                                 thr_modcsi_x_slope=thr_modcsi_x_slope, thr_csi_x_slope=thr_csi_x_slope,
                                 auto_baseline_beats=auto_baseline_beats, auto_pct=auto_pct,
                                 plots=plots, out=out)

def run_pipeline_from_tdms_auto(
    tdms_path: str, fs_override: float = None, index_path: str = None,
    rpeak_mode: str = "labview", window_beats: int = 100,
    thr_modcsi_x_slope: float = None, thr_csi_x_slope: float = None,
    auto_baseline_beats: int = 0, auto_pct: float = 99.5,
    plots: bool = False, out: str = None
) -> Dict[str, Any]:
    """Auto: vælg første kanal i filen; brug TDMS timing eller fs_override."""
    x, t_sec, fs, timestamps, g_name, ch_name = read_tdms_auto(tdms_path, fs_override, index_path)
    res = run_pipeline_from_ecg(ecg=x, fs=fs, t=t_sec,
                                rpeak_mode=rpeak_mode, window_beats=window_beats,
                                thr_modcsi_x_slope=thr_modcsi_x_slope, thr_csi_x_slope=thr_csi_x_slope,
                                auto_baseline_beats=auto_baseline_beats, auto_pct=auto_pct,
                                plots=plots, out=out)
    # TDMS-meta
    res["meta"].update({
        "tdms_path": tdms_path,
        "tdms_group": g_name,
        "tdms_channel": ch_name,
        "timestamps_available": timestamps is not None
    })
    return res
