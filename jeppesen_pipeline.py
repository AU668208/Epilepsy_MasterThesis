#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jeppesen ECG Seizure Pipeline — clean module
Public API:
- run_pipeline_from_ecg(...)
- run_pipeline_from_tdms(...)
- run_pipeline_from_tdms_auto(...)
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly, remez

# ---------- TDMS (valgfri) ----------
try:
    from nptdms import TdmsFile  # type: ignore
    _HAVE_TDMS = True
except Exception:
    _HAVE_TDMS = False

# ---------- Hjælpere ----------
def butter_bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 32.0, order: int = 4) -> np.ndarray:
    nyq = fs*0.5
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(float).copy()
    return np.convolve(x.astype(float), np.ones(win)/float(win), mode="same")

def sd1_sd2(rr: np.ndarray) -> Tuple[float, float]:
    if rr.size < 2: return np.nan, np.nan
    d = (rr[:-1]-rr[1:]) / np.sqrt(2.0)
    s = (rr[:-1]+rr[1:]) / np.sqrt(2.0)
    sd1 = np.std(d, ddof=1) if d.size > 1 else np.std(d)
    sd2 = np.std(s, ddof=1) if s.size > 1 else np.std(s)
    return float(sd1), float(sd2)

def lorenz_features(rr: np.ndarray) -> Tuple[float,float,float,float]:
    sd1, sd2 = sd1_sd2(rr)
    if not np.isfinite(sd1) or not np.isfinite(sd2) or sd1 == 0:
        return np.nan, np.nan, np.nan, np.nan
    T = 4.0*sd1; L = 4.0*sd2
    CSI = L/T; ModCSI = (L**2)/T
    return T, L, CSI, ModCSI

def rr_from_peaks(peaks: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if peaks is None or peaks.size < 2: return np.array([]), np.array([])
    rr = np.diff(peaks)/float(fs)
    t_rr = peaks[1:]/float(fs)
    return rr, t_rr

def sliding_windows_idx(n: int, w: int, step: int = 1):
    i = 0
    while i + w <= n:
        yield i, i + w
        i += step

def median_filter_beats(rr_sec: np.ndarray, win_beats: int = 7) -> np.ndarray:
    if rr_sec.size == 0: return rr_sec.copy()
    pad = win_beats // 2
    rr_pad = np.pad(rr_sec, (pad, pad), mode="edge")
    out = np.zeros_like(rr_sec)
    for i in range(rr_sec.size):
        out[i] = np.median(rr_pad[i:i+win_beats])
    return out

def tachogram_slope(hr_bpm: np.ndarray, t_s: np.ndarray) -> float:
    if hr_bpm.size < 2: return np.nan
    a, _ = np.polyfit(t_s.astype(float), hr_bpm.astype(float), 1)
    return float(a)

def compute_features(rr_sec: np.ndarray, t_rr: np.ndarray, window_beats: int = 100) -> pd.DataFrame:
    med_rr = median_filter_beats(rr_sec, 7)
    hr_med = 60.0/np.maximum(med_rr, 1e-6)
    hr_raw = 60.0/np.maximum(rr_sec, 1e-6)

    slopes = []
    T_list: List[float] = []; L_list: List[float] = []
    CSI_list: List[float] = []; ModCSI_list: List[float] = []
    CSI_raw_list: List[float] = []; idx_end: List[int] = []

    for s, e in sliding_windows_idx(rr_sec.size, window_beats, 1):
        slopes.append(tachogram_slope(hr_med[s:e], t_rr[s:e]))
        T,L,CSI,ModCSI = lorenz_features(med_rr[s:e]); T_list.append(T); L_list.append(L)
        CSI_list.append(CSI); ModCSI_list.append(ModCSI)
        Tr, Lr, CSIr, _ = lorenz_features(rr_sec[s:e]); CSI_raw_list.append(CSIr)
        idx_end.append(e-1)

    t_end = t_rr[np.asarray(idx_end, int)]
    slopes = np.asarray(slopes, float)
    T_arr = np.asarray(T_list, float); L_arr = np.asarray(L_list, float)
    CSI = np.asarray(CSI_list, float); ModCSI = np.asarray(ModCSI_list, float)
    CSI_raw = np.asarray(CSI_raw_list, float)

    df = pd.DataFrame({
        "beat_idx_end": idx_end,
        "t_end_s": t_end,
        "slope_bpm_per_s": slopes,
        "T": T_arr, "L": L_arr,
        "CSI": CSI, "ModCSI": ModCSI, "CSI_raw": CSI_raw,
    })
    df["modcsi_filt_times_slope"] = df["ModCSI"] * df["slope_bpm_per_s"]
    df["csi_times_slope"] = df["CSI_raw"] * df["slope_bpm_per_s"]
    return df

def auto_threshold(series: np.ndarray, pct: float = 99.5) -> float:
    s = series[np.isfinite(series)]
    return float(np.percentile(s, pct)) if s.size else np.nan

def detect_events(feature_series: np.ndarray, times: np.ndarray, thr: float, min_separation_beats: int = 50):
    mask = np.isfinite(feature_series) & (feature_series >= thr)
    events: List[List[int]] = []
    i = 0; n = mask.size
    while i < n:
        if not mask[i]: i += 1; continue
        st = i
        while i < n and mask[i]: i += 1
        en = i
        events.append([st,en])
    merged: List[List[int]] = []
    for ev in events:
        if not merged: merged.append(ev); continue
        if ev[0] - merged[-1][1] <= min_separation_beats:
            merged[-1][1] = max(merged[-1][1], ev[1])
        else:
            merged.append(ev)
    return merged

# ---------- Polaritets-hjælpere ----------
def _choose_polarity_from_first16s(x: np.ndarray, fs: float) -> float:
    n = int(16*fs); seg = x[:min(x.size, n)]
    if seg.size == 0: return 1.0
    return 1.0 if np.percentile(seg,95) >= abs(np.percentile(seg,5)) else -1.0

def _apply_local_polarity(x: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    L = int(round(win_s*fs)); y = x.copy()
    for st in range(0, x.size, L):
        en = min(x.size, st+L); seg = x[st:en]
        if seg.size and abs(seg.min()) > seg.max(): y[st:en] = -seg
    return y

# ---------- Preprocessing (LabVIEW-lignende) ----------
def _preprocess_labview_bandpass(
    ecg: np.ndarray, fs: float,
    delete_start_s: float = 60.0, delete_end_s: float = 300.0,
    hp_stop: float = 1.0, hp_pass: float = 1.2,
    lp_pass: float = 32.0, lp_stop: float = 50.0,
    smooth_win_samples: int = 4
) -> Tuple[np.ndarray, int]:
    x = np.asarray(ecg, float)

    # 1) trim
    n0, n1 = int(round(delete_start_s*fs)), int(round(delete_end_s*fs))
    if n0 + n1 >= x.size:
        raise ValueError("delete_start_s + delete_end_s fjerner hele signalet.")
    x = x[n0: x.size - (n1 if n1>0 else 0)]

    # 2) HP equiripple
    nyq = fs/2.0
    if not (0 < hp_stop < hp_pass < nyq): raise ValueError("HP kræver 0 < stop < pass < fs/2")
    taps_hp = remez(
        numtaps=101,
        bands=[0.0, hp_stop, hp_pass, nyq],
        desired=[0.0, 1.0],
        weight=[10.0, 1.0],
        fs=fs, type="bandpass"
    )
    x = filtfilt(taps_hp, [1.0], x, padlen=min(300, x.size-1))

    # 3) LP equiripple
    if not (0 < lp_pass < lp_stop < nyq): raise ValueError("LP kræver 0 < pass < stop < fs/2")
    taps_lp = remez(
        numtaps=101,
        bands=[0.0, lp_pass, lp_stop, nyq],
        desired=[1.0, 0.0],
        weight=[1.0, 10.0],
        fs=fs, type="bandpass"
    )
    x = filtfilt(taps_lp, [1.0], x, padlen=min(300, x.size-1))

    # 4) 4-sample moving average
    if smooth_win_samples > 1:
        ker = np.ones(int(smooth_win_samples))/float(smooth_win_samples)
        x = np.convolve(x, ker, mode="same")

    return x, n0

# ---------- Simpel detektor (til sanity) ----------
def rpeaks_simple(ecg: np.ndarray, fs: float) -> np.ndarray:
    x = butter_bandpass(ecg, fs, 0.5, 32.0, 4)
    # enkel kvadreret MWI
    mwi = moving_average((np.gradient(x))**2, max(1, int(round(0.150*fs))))
    thr = np.percentile(mwi, 95)*0.5
    peaks: List[int] = []; last = -10**9; refr = int(round(0.25*fs)); alpha=0.01
    for i in range(1, mwi.size-1):
        thr = (1-alpha)*thr + alpha*mwi[i]
        if mwi[i]>thr and mwi[i]>=mwi[i-1] and mwi[i]>=mwi[i+1]:
            if i - last >= refr: peaks.append(i); last = i
    return np.asarray(peaks, int)

# ---------- LabVIEW R-peak detektor ----------
@dataclass
class LVParams:
    fs: float = 256.0
    win_s: float = 2.0
    thigh_alpha: float = 0.75
    fwd_bwd_radius_samp: int = 15    # @256 Hz (skaleres internt)
    delta_thresh_samples: int = 35   # @256 Hz (skaleres internt)
    rrshort_n: int = 8
    rrlong_n: int = 34
    refractory_s: float = 0.25
    rmax_clip_low_s: float = 0.4
    rmax_clip_high_s: float = 1.2
    auto_polarity: str = "global16s"  # 'none' | 'global16s' | 'per_window'
    enable_tlow: bool = True

class LabVIEWRpeak:
    def __init__(self, params: LVParams):
        self.p = params
        scale = max(1.0, self.p.fs/256.0)
        self._radius = max(1, int(round(self.p.fwd_bwd_radius_samp*scale)))
        self._delta_thr = max(1, int(round(self.p.delta_thresh_samples*scale)))

    def _window_max_series(self, sig: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
        fs = self.p.fs; L = int(round(self.p.win_s*fs))
        rect = np.maximum(sig, 0.0); m: List[float] = []; edges: List[Tuple[int,int]] = []
        for st in range(0, sig.size, L):
            en = min(sig.size, st+L); m.append(np.max(rect[st:en]) if en>st else 0.0); edges.append((st,en))
        return np.asarray(m), edges

    def _thigh_series(self, window_max: np.ndarray) -> np.ndarray:
        alpha = self.p.thigh_alpha; N = window_max.size; thigh = np.zeros(N)
        for k in range(N):
            if k == 0: thigh[k] = alpha*window_max[0]
            else:
                lo = max(0, k-8); ref = window_max[lo:k]
                thigh[k] = alpha*(np.median(ref) if ref.size else window_max[k])
        return thigh

    def _localise(self, x: np.ndarray, idx: int) -> int:
        r = self._radius; st = max(0, idx-r); en = min(x.size, idx+r+1)
        seg = x[st:en]; off = np.argmax(seg) if seg.size else 0
        return st + off

    def detect(self, ecg: np.ndarray) -> np.ndarray:
        x = ecg.astype(float)
        # auto-polaritet
        if self.p.auto_polarity == "global16s":
            x *= _choose_polarity_from_first16s(x, self.p.fs)
        elif self.p.auto_polarity == "per_window":
            x = _apply_local_polarity(x, self.p.fs, self.p.win_s)

        # 1–32 Hz som i pipen (efter LabVIEW-preproc ligger vi typisk i dette bånd)
        bp = butter_bandpass(x, self.p.fs, 0.5, 32.0, 4)

        # Thigh/Tlow
        wmax, edges = self._window_max_series(bp)
        thigh_w = self._thigh_series(wmax)
        thigh = np.zeros(bp.size)
        for k,(st,en) in enumerate(edges): thigh[st:en] = thigh_w[k]
        tlow = 0.4*thigh if self.p.enable_tlow else None

        peaks: List[int] = []; rr_list: List[float] = []
        fs = self.p.fs; refr = int(round(self.p.refractory_s*fs))
        Lwin = int(round(self.p.win_s*fs)); next_edge = Lwin

        i = 0
        while i < bp.size:
            if peaks and i - peaks[-1] < refr: i += 1; continue

            use_tlow = False
            if self.p.enable_tlow and peaks:
                dt = (i - peaks[-1]) / fs
                rlong = np.median(rr_list[-self.p.rrlong_n:]) if len(rr_list)>=2 else 0.8
                rmax = float(np.clip(rlong, self.p.rmax_clip_low_s, self.p.rmax_clip_high_s))
                if dt >= rmax: use_tlow = True

            if not use_tlow:
                # Thigh-kryds
                if bp[i] > thigh[i] and (i==0 or bp[i-1] <= thigh[i-1]):
                    idx = self._localise(bp, i)
                    # lokalt max
                    if (idx==0 or bp[idx]>=bp[idx-1]) and (idx==bp.size-1 or bp[idx]>=bp[idx+1]):
                        if not peaks or idx - peaks[-1] >= refr:
                            peaks.append(idx)
                            if len(peaks)>=2: rr_list.append((peaks[-1]-peaks[-2])/fs)
                            i = idx + 1; continue
            else:
                # DELTA / variabilitet
                high_var = False
                if len(rr_list) >= 5:
                    seg_rr = np.asarray(rr_list[-self.p.rrlong_n:]) if len(rr_list)>self.p.rrlong_n else np.asarray(rr_list)
                    med = float(np.median(seg_rr))
                    eps = np.abs(seg_rr - med)
                    if eps.size >= 2:
                        rm = np.argsort(eps)[-2:]     # fjern 2 største afvigelser
                        keep = np.ones_like(eps, bool); keep[rm] = False
                        delta_val = float(np.mean(eps[keep])) if keep.any() else float(np.mean(eps))
                    else:
                        delta_val = float(np.mean(eps))
                    high_var = (delta_val*fs > 1.4*self._delta_thr)  # lidt strammere end 1:1

                lookback_rr = self.p.rrshort_n if high_var else self.p.rrlong_n
                rlong = np.median(rr_list[-self.p.rrlong_n:]) if len(rr_list)>=2 else 0.8
                rmax = float(np.clip(rlong, self.p.rmax_clip_low_s, self.p.rmax_clip_high_s))
                lookback_s = min( (rlong if len(rr_list) else 1.0), 1.1*rmax )
                st = max(0, i - int(round(lookback_s*fs)))
                seg = bp[st:i+1]; tl = tlow[st:i+1]  # type: ignore
                cand = np.where(seg >= tl)[0]
                if cand.size:
                    ci = st + cand[np.argmax(seg[cand])]
                    ci = self._localise(bp, ci)
                    # kræv lokalt max (prominens light)
                    if 0 < ci < bp.size-1 and not (bp[ci]>=bp[ci-1] and bp[ci]>=bp[ci+1]):
                        ci = None  # forkast
                    if ci is not None and (not peaks or ci - peaks[-1] >= refr):
                        peaks.append(ci)
                        if len(peaks)>=2: rr_list.append((peaks[-1]-peaks[-2])/fs)
                        i = ci + 1; continue

            if i >= next_edge-1: next_edge += Lwin
            i += 1

        return np.asarray(sorted(set(peaks)), int)

# ---------- TDMS read ----------
def read_tdms(tdms_path: str, group: str, channel: str, time_channel: Optional[str] = None):
    if not _HAVE_TDMS: raise ImportError("pip install nptdms")
    td = TdmsFile.read(tdms_path)
    ch = td[group][channel]; x = ch.data.astype(float)
    if time_channel is not None:
        t = td[group][time_channel].data.astype(float)
        if t.size != x.size: raise ValueError("time_channel length != channel length")
        fs = 1.0/np.median(np.diff(t)) if t.size>1 else None
        return x, t, fs
    dt = ch.properties.get("wf_increment", None)
    if dt is None: dt = ch.properties.get("dt", None)
    if dt is None and "SamplingRate" in ch.properties:
        dt = 1.0/float(ch.properties["SamplingRate"])
    fs = None if dt is None else 1.0/float(dt)
    t = None if dt is None else (0.0 + np.arange(x.size)*float(dt))
    return x, t, fs

def _infer_tdms_group_channel(td: "TdmsFile") -> Tuple[str,str]:  # type: ignore
    groups = td.groups()
    if not groups: raise ValueError("TDMS contains no groups.")
    g0 = groups[0]
    chans = g0.channels()
    if chans: return g0.name, chans[0].name
    for g in groups[1:]:
        chs = g.channels()
        if chs: return g.name, chs[0].name
    raise ValueError("TDMS contains no channels.")

def read_tdms_auto(tdms_path: str, fs_override: float = None, index_path: str = None):
    if not _HAVE_TDMS: raise ImportError("pip install nptdms")
    td = TdmsFile.read(tdms_path)
    g, c = _infer_tdms_group_channel(td)
    ch = td[g][c]; x = ch.data.astype(float)
    dt = ch.properties.get("wf_increment", None)
    if dt is None: dt = ch.properties.get("dt", None)
    if dt is None and "SamplingRate" in ch.properties: dt = 1.0/float(ch.properties["SamplingRate"])
    fs = fs_override if fs_override is not None else (None if dt is None else 1.0/float(dt))
    if fs is None: raise ValueError("Sampling rate unknown, angiv fs_override.")

    t_seconds = None if dt is None else (0.0 + np.arange(x.size)*float(dt)).astype(float)
    # timestamps fra wf_start_time (valgfrit)
    timestamps = None
    wf_start = ch.properties.get("wf_start_time", None)
    if wf_start is not None and dt is not None:
        try:
            step_ns = int(round((1.0/fs)*1e9))
            start_np = np.datetime64(wf_start, "ns")
            timestamps = start_np + np.arange(x.size)*np.timedelta64(step_ns, "ns")
        except Exception:
            timestamps = None
    return x, t_seconds, float(fs), timestamps, g, c

# ---------- Core finish ----------
def _finish(rr_sec, t_rr, out, plots,
            thr_modcsi_x_slope, thr_csi_x_slope,
            auto_baseline_beats, auto_pct, window_beats):
    features = compute_features(rr_sec, t_rr, window_beats)
    thr_mod = thr_modcsi_x_slope
    thr_csi = thr_csi_x_slope
    if auto_baseline_beats and auto_baseline_beats > window_beats:
        mask = features["beat_idx_end"].values < auto_baseline_beats
        if np.any(mask):
            if thr_mod is None: thr_mod = auto_threshold(features.loc[mask, "modcsi_filt_times_slope"].values, auto_pct)
            if thr_csi is None: thr_csi = auto_threshold(features.loc[mask, "csi_times_slope"].values, auto_pct)
    if thr_mod is None: thr_mod = auto_threshold(features["modcsi_filt_times_slope"].values, auto_pct)
    if thr_csi is None: thr_csi = auto_threshold(features["csi_times_slope"].values, auto_pct)

    times = features["t_end_s"].values
    ev_mod = detect_events(features["modcsi_filt_times_slope"].values, times, thr_mod, 50)
    ev_csi = detect_events(features["csi_times_slope"].values, times, thr_csi, 50)
    all_events = sorted(ev_mod + ev_csi, key=lambda x: x[0])
    merged: List[List[int]] = []
    for st,en in all_events:
        if not merged or st > merged[-1][1]:
            merged.append([st,en])
        else:
            merged[-1][1] = max(merged[-1][1], en)
    detections = pd.DataFrame([{
        "win_start_idx": int(st),
        "win_end_idx": int(en),
        "t_start_s": float(times[st]),
        "t_end_s": float(times[min(en-1, len(times)-1)]),
        "dur_s": float(times[min(en-1, len(times)-1)] - times[st]),
    } for st,en in merged])

    if plots and out:
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(times, features["modcsi_filt_times_slope"].values); plt.axhline(thr_mod, ls="--")
        plt.xlabel("Time (s)"); plt.ylabel("ModCSI_filt × slope"); plt.title("Feature"); plt.tight_layout()
        plt.savefig(f"{out}_modcsi_times_slope.png", dpi=160); plt.close()

        plt.figure(); plt.plot(times, features["csi_times_slope"].values); plt.axhline(thr_csi, ls="--")
        plt.xlabel("Time (s)"); plt.ylabel("CSI × slope"); plt.title("Feature"); plt.tight_layout()
        plt.savefig(f"{out}_csi_times_slope.png", dpi=160); plt.close()

    meta = {"window_beats": window_beats, "auto_baseline_beats": auto_baseline_beats,
            "auto_pct": auto_pct, "thr_modcsi_filt_times_slope": thr_mod, "thr_csi_times_slope": thr_csi}
    return features, detections, meta

# ---------- Runners ----------
def run_pipeline_from_ecg(
    ecg: np.ndarray, fs: float, t: Optional[np.ndarray] = None, target_fs: Optional[float] = None,
    rpeak_mode: str = "labview", window_beats: int = 100,
    thr_modcsi_x_slope: Optional[float] = None, thr_csi_x_slope: Optional[float] = None,
    auto_baseline_beats: int = 0, auto_pct: float = 99.5,
    plots: bool = False, out: Optional[str] = None
) -> Dict[str, Any]:
    x = np.asarray(ecg, float)
    if target_fs and abs(target_fs - fs) > 1e-9:
        x = resample_poly(x, int(round(target_fs)), int(round(fs)))
        fs = float(target_fs)
        t = np.arange(x.size)/fs
    elif t is None:
        t = np.arange(x.size)/fs

    # LabVIEW-lignende preproc (her uden start/slut trim)
    x_proc, _ = _preprocess_labview_bandpass(
        x, fs, delete_start_s=0.0, delete_end_s=0.0,
        hp_stop=1.0, hp_pass=1.2, lp_pass=32.0, lp_stop=50.0, smooth_win_samples=4
    )

    # R-peaks
    if rpeak_mode == "simple":
        peaks = rpeaks_simple(x_proc, fs)
    else:
        peaks = LabVIEWRpeak(LVParams(fs=fs, auto_polarity="global16s", enable_tlow=True)).detect(x_proc)

    rr_sec, t_rr = rr_from_peaks(peaks, fs)
    if rr_sec.size < max(10, window_beats+1): raise ValueError("Not enough beats for analysis.")
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
    if not _HAVE_TDMS: raise ImportError("pip install nptdms")
    x, t_inferred, fs_inferred = read_tdms(tdms_path, group, channel, time_channel)
    fs_use = fs if fs is not None else fs_inferred
    if fs_use is None: raise ValueError("Sampling rate unknown.")
    return run_pipeline_from_ecg(x, fs_use, t=t_inferred, target_fs=target_fs,
                                 rpeak_mode=rpeak_mode, window_beats=window_beats,
                                 thr_modcsi_x_slope=thr_modcsi_x_slope, thr_csi_x_slope=thr_csi_x_slope,
                                 auto_baseline_beats=auto_baseline_beats, auto_pct=auto_pct,
                                 plots=plots, out=out)

def run_pipeline_from_tdms_auto(
    tdms_path: str,
    *,
    fs_override: float = None,
    index_path: str = None,             # nu ikke brugt direkte, men kan udvides
    rpeak_mode: str = "labview",
    preproc: str = "labview",           # "labview" | "none"
    ui_defaults: bool = True,
    delete_start_s: float = None,
    delete_end_s: float = None,
    hp_stop: float = None, hp_pass: float = None,
    lp_pass: float = None, lp_stop: float = None,
    smooth_win_samples: int = None,
    start_s: float = None, dur_s: float = None,
    window_beats: int = 100,
    auto_baseline_beats: int = 20000, auto_pct: float = 99.5,
    plots: bool = True, out: str = "run"
) -> Dict[str, Any]:
    x, t_sec, fs, *_ = read_tdms_auto(tdms_path, fs_override=fs_override)

    # evt. slice (hurtig fejlsøgning)
    if start_s is not None and dur_s is not None:
        i0, i1 = int(start_s*fs), int((start_s+dur_s)*fs)
        x = x[i0:i1]
        if t_sec is not None: t_sec = t_sec[i0:i1]
        # ved korte slices: typisk ingen trim
        if ui_defaults and (delete_start_s is None): delete_start_s = 0.0
        if ui_defaults and (delete_end_s   is None): delete_end_s   = 0.0

    # UI defaults fra VI
    if ui_defaults:
        if delete_start_s is None: delete_start_s = 60.0
        if delete_end_s   is None: delete_end_s   = 300.0
        if hp_stop        is None: hp_stop        = 1.0
        if hp_pass        is None: hp_pass        = 1.2
        if lp_pass        is None: lp_pass        = 32.0
        if lp_stop        is None: lp_stop        = 50.0
        if smooth_win_samples is None: smooth_win_samples = 4

    # preproc
    if preproc == "labview":
        x_proc, n0 = _preprocess_labview_bandpass(
            x, fs, delete_start_s=delete_start_s, delete_end_s=delete_end_s,
            hp_stop=hp_stop, hp_pass=hp_pass, lp_pass=lp_pass, lp_stop=lp_stop,
            smooth_win_samples=smooth_win_samples
        )
        if t_sec is not None: t_sec = t_sec[n0:n0+x_proc.size]
    else:
        x_proc = x

    # R-peaks
    if rpeak_mode == "simple":
        peaks = rpeaks_simple(x_proc, fs)
    else:
        lv = LabVIEWRpeak(LVParams(fs=fs, auto_polarity="global16s", enable_tlow=True))
        peaks = lv.detect(x_proc)

    # HRV + detections
    rr_sec, t_rr = rr_from_peaks(peaks, fs)
    features, detections, meta = _finish(rr_sec, t_rr, out, plots,
                                         thr_modcsi_x_slope=None, thr_csi_x_slope=None,
                                         auto_baseline_beats=auto_baseline_beats, auto_pct=auto_pct,
                                         window_beats=window_beats)

    meta_out = {
        "fs_used": fs, "tdms_path": tdms_path,
        "preproc": preproc, "ui_defaults": ui_defaults,
        "delete_start_s": delete_start_s, "delete_end_s": delete_end_s,
        "hp_stop": hp_stop, "hp_pass": hp_pass, "lp_pass": lp_pass, "lp_stop": lp_stop,
        "smooth_win_samples": smooth_win_samples,
        "slice_start_s": start_s, "slice_dur_s": dur_s
    }
    meta_out.update(meta or {})

    return {
        "peaks": peaks, "rr_sec": rr_sec, "t_rr": t_rr,
        "features": features, "detections": detections,
        "meta": meta_out, "ecg_proc": x_proc, "t_proc": t_sec
    }
