#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Seizure Pipeline (TDMS-native, LabVIEW-parallel)
====================================================
What's new vs v2:
- Reads **TDMS directly** (via nptdms) with: --tdms --group --channel [--time_channel]
- Keeps both R-peak modes:
   --rpeak_mode simple  (Pan–Tompkins-like)
   --rpeak_mode labview (Jeppesen VI: Thigh/Tlow, Rmax, DELTA, ±15-sample localisation, Tref=0.25s)
- Everything else unchanged: RR→CSI/mCSI, slope, thresholds, detections, plots.

Install once:
  pip install nptdms pandas numpy scipy matplotlib

Examples
--------
1) Direct TDMS input, known ECG channel (256 Hz):
   python labview_to_python_ecg_v3_tdms.py --tdms rec.tdms --group "ECG" --channel "Lead_I" --fs 256 \
     --rpeak_mode labview --out run_tdms --plots

2) Same, but let TDMS waveform properties provide timing (wf_increment / SamplingRate):
   python labview_to_python_ecg_v3_tdms.py --tdms rec.tdms --group "ECG" --channel "Lead_I" \
     --rpeak_mode labview --out run_tdms --plots

3) TDMS includes a separate time channel (same group), specify it:
   python labview_to_python_ecg_v3_tdms.py --tdms rec.tdms --group "ECG" --channel "Lead_I" --time_channel "t"
"""
import argparse, json, math, sys
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
import matplotlib.pyplot as plt
from nptdms import TdmsFile

# -----------------------------
# Filters & helpers
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5*fs
    b,a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b,a

def bandpass_filter(x, fs, low=0.5, high=32.0, order=4):
    b,a = butter_bandpass(low, high, fs, order)
    return filtfilt(b,a,x)

def derivative_filter(x, fs):
    kern = np.array([-1,-2,0,2,1]) * (fs/8.0)
    return np.convolve(x, kern, mode='same')

def moving_average(x, win):
    if win<=1: return x.copy()
    return np.convolve(x, np.ones(win)/win, mode='same')

def sd1_sd2(rr):
    if len(rr)<2: return np.nan, np.nan
    d = (rr[:-1]-rr[1:])/math.sqrt(2.0)
    s = (rr[:-1]+rr[1:])/math.sqrt(2.0)
    sd1 = np.std(d, ddof=1) if len(d)>1 else np.std(d)
    sd2 = np.std(s, ddof=1) if len(s)>1 else np.std(s)
    return sd1, sd2

def lorenz_features(rr):
    sd1,sd2 = sd1_sd2(rr)
    if not np.isfinite(sd1) or not np.isfinite(sd2) or sd1==0: return np.nan, np.nan, np.nan, np.nan
    T = 4.0*sd1; L = 4.0*sd2
    CSI = L/T
    ModCSI = (L**2)/T
    return T,L,CSI,ModCSI

# -----------------------------
# Simple R-peak (Pan–Tompkins style)
# -----------------------------
def adaptive_peak_detect(integrated, fs, refractory_ms=250, alpha=0.01):
    thr = np.percentile(integrated,95)*0.5
    peaks=[]; last=-10**9; refr = int(round(refractory_ms*fs/1000.0))
    for i in range(1,len(integrated)-1):
        thr = (1-alpha)*thr + alpha*integrated[i]
        if integrated[i]>thr and integrated[i]>integrated[i-1] and integrated[i]>=integrated[i+1]:
            if i-last>=refr: peaks.append(i); last=i
    return np.array(peaks, dtype=int)

def rpeaks_simple(ecg, fs):
    bp = bandpass_filter(ecg, fs, 0.5, 32.0, 4)
    der = derivative_filter(bp, fs)
    sqr = der**2
    mwi = moving_average(sqr, max(1,int(round(0.150*fs))))
    return adaptive_peak_detect(mwi, fs, refractory_ms=250, alpha=0.01)

# -----------------------------
# LabVIEW-style R-peak
# -----------------------------
@dataclass
class LVParams:
    fs: float = 256.0
    win_s: float = 2.0               # 2-sec windows
    thigh_alpha: float = 0.75        # scale on median(max of last 8 windows)
    fwd_bwd_radius_samp: int = 15    # ±15 samples search
    refractory_s: float = 0.25       # Tref
    delta_thresh_samples: int = 35   # ~0.137 s @256 Hz
    rrshort_n: int = 8
    rrlong_n: int = 34
    rmax_clip_low_s: float = 0.4
    rmax_clip_high_s: float = 1.2

class LabVIEWRpeak:
    def __init__(self, params: LVParams):
        self.p = params

    def _window_max_series(self, sig):
        """Max per 2s window of rectified filtered ECG (negatives→0)."""
        fs = self.p.fs; L = int(round(self.p.win_s*fs))
        rect = np.maximum(sig, 0.0)
        m=[]; edges=[]
        for st in range(0, len(rect), L):
            en=min(len(rect), st+L)
            m.append(np.max(rect[st:en]) if en>st else 0.0)
            edges.append((st,en))
        return np.array(m), edges

    def _thigh_series(self, window_max):
        alpha = self.p.thigh_alpha
        N=len(window_max); thigh=np.zeros(N)
        for k in range(N):
            if k==0:
                thigh[k]=alpha*window_max[0]
            else:
                lo=max(0,k-8)
                ref = window_max[lo:k] if k>0 else window_max[:1]
                med = np.median(ref) if ref.size>0 else window_max[k]
                thigh[k]=alpha*med
        return thigh

    def _fwd_bwd_localise(self, x, idx):
        r = self.p.fwd_bwd_radius_samp
        st=max(0, idx-r); en=min(len(x), idx+r+1)
        if en<=st: return idx
        seg=x[st:en]; off=np.argmax(seg)
        return st+off

    def detect(self, ecg):
        fs = self.p.fs
        bp = bandpass_filter(ecg, fs, 0.5, 32.0, 4)

        wmax, edges = self._window_max_series(bp)
        thigh_series = self._thigh_series(wmax)

        # Upsample Thigh to sample domain
        thigh = np.zeros(len(bp))
        for k,(st,en) in enumerate(edges):
            thigh[st:en]=thigh_series[k]
        # Tlow as scaled Thigh
        tlow = 0.4*thigh

        peaks=[]; rr_list=[]
        refr = int(round(self.p.refractory_s*fs))
        Lwin=int(round(self.p.win_s*fs))
        last_window_det=False; next_window_edge=Lwin

        i=0
        while i < len(bp):
            # Refractory
            if peaks and i - peaks[-1] < refr:
                i += 1; continue

            # Overdue? → Tlow searchback
            thr = thigh[i]; use_tlow=False
            if len(peaks)>0:
                dt = (i - peaks[-1]) / fs
                rlong = np.median(rr_list[-self.p.rrlong_n:]) if len(rr_list)>=2 else 0.8
                rmax = np.clip(rlong, self.p.rmax_clip_low_s, self.p.rmax_clip_high_s)
                if dt >= rmax: use_tlow=True

            if not use_tlow:
                if bp[i] > thr and (i==0 or bp[i-1] <= thr):
                    idx = self._fwd_bwd_localise(bp, i)
                    if (idx==0 or bp[idx]>=bp[idx-1]) and (idx==len(bp)-1 or bp[idx]>=bp[idx+1]):
                        if len(peaks)==0 or idx - peaks[-1] >= refr:
                            peaks.append(idx); last_window_det=True
                            if len(peaks)>=2: rr_list.append((peaks[-1]-peaks[-2])/fs)
                            i = idx + 1; continue
            else:
                # Tlow searchback
                if len(rr_list) >= 3:
                    seg_rr = np.array(rr_list[-self.p.rrlong_n:]) if len(rr_list)>=self.p.rrlong_n else np.array(rr_list)
                    if seg_rr.size>=5:
                        med = np.median(seg_rr)
                        eps = np.abs(seg_rr - med)
                        if eps.size>=2:
                            idx_rm = np.argsort(eps)[-2:]; mask = np.ones_like(eps, bool); mask[idx_rm]=False
                            delta_val = np.mean(eps[mask]) if mask.any() else np.mean(eps)
                        else:
                            delta_val = np.mean(eps)
                        delta_samples = delta_val * fs
                        high_var = (delta_samples > self.p.delta_thresh_samples)
                    else:
                        high_var=False
                    lookback_rr = self.p.rrshort_n if high_var else self.p.rrlong_n
                    lookback_s = np.sum(rr_list[-lookback_rr:]) if len(rr_list)>=1 else 2.0
                else:
                    lookback_s=2.0
                lookback_samp = int(round(lookback_s*fs))
                st=max(0, i - lookback_samp)
                seg = bp[st:i+1]; tl = tlow[st:i+1]
                cand = np.where(seg >= tl)[0]
                if cand.size>0:
                    ci = st + cand[np.argmax(seg[cand])]
                    ci = self._fwd_bwd_localise(bp, ci)
                    if len(peaks)==0 or ci - peaks[-1] >= refr:
                        peaks.append(ci); last_window_det=True
                        if len(peaks)>=2: rr_list.append((peaks[-1]-peaks[-2])/fs)
                        i = ci + 1; continue

            # Shadow behaviour at window boundary
            if i>=next_window_edge-1:
                if not last_window_det:
                    pass
                last_window_det=False
                next_window_edge += Lwin
            i += 1

        return np.array(sorted(set(peaks)), dtype=int)

# -----------------------------
# RR utils & features
# -----------------------------
def rr_from_peaks(peaks, fs):
    if len(peaks)<2: return np.array([]), np.array([])
    rr = np.diff(peaks)/float(fs)
    t_rr = peaks[1:]/float(fs)
    return rr, t_rr

def median_filter_beats(rr_sec, win_beats=7):
    if len(rr_sec)==0: return rr_sec.copy()
    pad = win_beats//2
    rr_pad = np.pad(rr_sec, (pad,pad), mode='edge')
    out = np.zeros_like(rr_sec)
    for i in range(len(rr_sec)):
        out[i] = np.median(rr_pad[i:i+win_beats])
    return out

def sliding_windows_idx(n, w, step=1):
    i=0
    while i+w<=n:
        yield i,i+w
        i+=step

def tachogram_slope(hr_bpm, t_s):
    if len(hr_bpm)<2: return np.nan
    a,_ = np.polyfit(np.asarray(t_s), np.asarray(hr_bpm), 1)
    return a

def compute_features(rr_sec, t_rr, window_beats=100):
    med_rr = median_filter_beats(rr_sec, 7)
    hr = 60.0/np.maximum(rr_sec,1e-6)
    hr_med = 60.0/np.maximum(med_rr,1e-6)
    t_cum = t_rr.copy()

    slopes_med=[]; T_list=[]; L_list=[]; CSI_list=[]; ModCSI_list=[]; CSI_raw_list=[]; idx_end=[]
    for s,e in sliding_windows_idx(len(rr_sec), window_beats, 1):
        slopes_med.append(tachogram_slope(hr_med[s:e], t_cum[s:e]))
        T,L,csi,modcsi = lorenz_features(med_rr[s:e])
        T_list.append(T); L_list.append(L); CSI_list.append(csi); ModCSI_list.append(modcsi)
        Tr,Lr,csir,_ = lorenz_features(rr_sec[s:e])
        CSI_raw_list.append(csir)
        idx_end.append(e-1)

    t_end = t_rr[np.array(idx_end, dtype=int)]
    slopes_med = np.asarray(slopes_med, float)
    T_arr = np.asarray(T_list,float); L_arr=np.asarray(L_list,float)
    CSI = np.asarray(CSI_list,float); ModCSI=np.asarray(ModCSI_list,float)
    CSI_raw=np.asarray(CSI_raw_list,float)
    modcsi_filt_times_slope = ModCSI*slopes_med
    csi_times_slope = CSI_raw*slopes_med
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

# -----------------------------
# Thresholding & detections
# -----------------------------
def auto_threshold(series: np.ndarray, pct: float = 99.5):
    series = series[np.isfinite(series)]
    return float(np.percentile(series, pct)) if series.size>0 else np.nan

def detect_events(feature_series, times, thr, min_separation_beats=50):
    mask = np.isfinite(feature_series) & (feature_series>=thr)
    events=[]; i=0; n=len(mask)
    while i<n:
        if not mask[i]: i+=1; continue
        st=i
        while i<n and mask[i]: i+=1
        en=i
        events.append([st,en])
    merged=[]
    for ev in events:
        if not merged: merged.append(ev); continue
        if ev[0]-merged[-1][1] <= min_separation_beats:
            merged[-1][1]=max(merged[-1][1], ev[1])
        else:
            merged.append(ev)
    return merged

# -----------------------------
# TDMS reader
# -----------------------------
def read_tdms(path, group, channel, time_channel=None):
    td = TdmsFile.read(path)
    ch = td[group][channel]
    x = ch.data.astype(float)

    # If explicit time channel provided
    if time_channel is not None:
        t = td[group][time_channel].data.astype(float)
        if t.size != x.size:
            raise ValueError("time_channel length != channel length")
        fs = 1.0/np.median(np.diff(t)) if t.size>1 else None
        return x, t, fs

    # Else infer timing from NI waveform properties
    dt = None
    if 'wf_increment' in ch.properties:
        dt = float(ch.properties['wf_increment'])
    elif 'dt' in ch.properties:
        dt = float(ch.properties['dt'])
    else:
        fs_prop = ch.properties.get('SamplingRate', None)
        if fs_prop is not None:
            dt = 1.0/float(fs_prop)

    if dt is None:
        # No timing found: return only signal; caller must supply --fs
        return x, None, None

    t0 = float(ch.properties.get('wf_start_offset', 0.0))
    t = t0 + np.arange(x.size)*dt
    fs = 1.0/dt if dt>0 else None
    return x, t, fs

# -----------------------------
# I/O & main
# -----------------------------
def load_input(args):
    """Return (ecg_or_rr, t, fs, is_rr). If is_rr=True -> vector is RR in seconds."""
    if args.rr_csv:
        df = pd.read_csv(args.rr_csv)
        rr = df[args.rr_col].astype(float).values
        if args.rr_time_col and args.rr_time_col in df.columns:
            t_rr = df[args.rr_time_col].astype(float).values
        else:
            t_rr = np.cumsum(rr)
        return rr, t_rr, None, True

    if args.tdms:
        ecg, t, fs_inferred = read_tdms(args.tdms, args.group, args.channel, args.time_channel)
        fs = args.fs if args.fs else fs_inferred
        if fs is None:
            raise ValueError("Sampling rate unknown. Provide --fs or ensure TDMS has wf_increment/dt/SamplingRate or a time channel.")
        if t is None:
            t = np.arange(len(ecg))/fs
        if args.target_fs and abs(args.target_fs - fs) > 1e-9:
            up=int(round(args.target_fs)); down=int(round(fs))
            ecg = resample_poly(ecg, up, down)
            fs = float(args.target_fs)
            t = np.arange(len(ecg))/fs
        return ecg, t, fs, False

    # CSV ECG fallback
    df = pd.read_csv(args.in_csv)
    if args.time_col and args.time_col in df.columns:
        t = df[args.time_col].astype(float).values
        fs = args.fs if args.fs else 1.0/np.median(np.diff(t))
    else:
        if not args.fs: raise ValueError("Provide --fs if no time column is present.")
        fs = float(args.fs); t = np.arange(len(df))/fs
    ecg = df[args.ecg_col].astype(float).values
    if args.target_fs and abs(args.target_fs - fs) > 1e-9:
        up=int(round(args.target_fs)); down=int(round(fs))
        ecg = resample_poly(ecg, up, down); fs = float(args.target_fs); t = np.arange(len(ecg))/fs
    return ecg, t, fs, False

def main():
    ap = argparse.ArgumentParser(description="ECG seizure pipeline (TDMS-native, LabVIEW-parallel)")
    # TDMS input
    ap.add_argument("--tdms", type=str, default=None, help="Path to .tdms file")
    ap.add_argument("--group", type=str, default=None, help="TDMS group name")
    ap.add_argument("--channel", type=str, default=None, help="TDMS channel name (ECG)")
    ap.add_argument("--time_channel", type=str, default=None, help="Optional TDMS time channel")

    # CSV ECG input (fallback)
    ap.add_argument("--in", dest="in_csv", type=str, default=None)
    ap.add_argument("--time_col", type=str, default=None)
    ap.add_argument("--ecg_col", type=str, default="ecg")

    # RR input
    ap.add_argument("--rr_csv", type=str, default=None)
    ap.add_argument("--rr_col", type=str, default="rr_s")
    ap.add_argument("--rr_time_col", type=str, default=None)

    # Sampling
    ap.add_argument("--fs", type=float, default=None, help="Sampling rate Hz (needed if not in TDMS)")
    ap.add_argument("--target_fs", type=float, default=None, help="Optional resample target Hz (e.g., 256)")

    # R-peak mode
    ap.add_argument("--rpeak_mode", type=str, choices=["simple","labview"], default="labview")

    # HRV windows & outputs
    ap.add_argument("--window_beats", type=int, default=100)
    ap.add_argument("--out", type=str, default="run")
    ap.add_argument("--plots", action="store_true")

    # Thresholds
    ap.add_argument("--thr_modcsi_x_slope", type=float, default=None)
    ap.add_argument("--thr_csi_x_slope", type=float, default=None)
    ap.add_argument("--auto_baseline_beats", type=int, default=0)
    ap.add_argument("--auto_pct", type=float, default=99.5)

    args = ap.parse_args()

    # Load
    vec, t, fs, is_rr = load_input(args)

    if is_rr:
        rr_sec, t_rr = vec, t
        fs_used = None
    else:
        ecg = vec
        fs_used = fs
        if args.rpeak_mode=="simple":
            peaks = rpeaks_simple(ecg, fs)
        else:
            lv = LabVIEWRpeak(LVParams(fs=fs))
            peaks = lv.detect(ecg)
        rr_sec, t_rr = rr_from_peaks(peaks, fs)

    if len(rr_sec) < max(10, args.window_beats+1):
        print("Not enough beats for analysis."); sys.exit(1)

    feat = compute_features(rr_sec, t_rr, window_beats=args.window_beats)

    thr_mod = args.thr_modcsi_x_slope
    thr_csi = args.thr_csi_x_slope

    if args.auto_baseline_beats and args.auto_baseline_beats > args.window_beats:
        mask = (feat["beat_idx_end"].values < args.auto_baseline_beats)
        if np.any(mask):
            if thr_mod is None: thr_mod = auto_threshold(feat.loc[mask,"modcsi_filt_times_slope"].values, args.auto_pct)
            if thr_csi is None: thr_csi = auto_threshold(feat.loc[mask,"csi_times_slope"].values, args.auto_pct)
    if thr_mod is None: thr_mod = auto_threshold(feat["modcsi_filt_times_slope"].values, args.auto_pct)
    if thr_csi is None: thr_csi = auto_threshold(feat["csi_times_slope"].values, args.auto_pct)

    times = feat["t_end_s"].values
    ev_mod = detect_events(feat["modcsi_filt_times_slope"].values, times, thr_mod, 50)
    ev_csi = detect_events(feat["csi_times_slope"].values, times, thr_csi, 50)
    all_events = sorted(ev_mod + ev_csi, key=lambda x:x[0])
    merged=[]
    for st,en in all_events:
        if not merged: merged.append([st,en]); continue
        if st <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], en)
        else: merged.append([st,en])

    feat.to_csv(f"{args.out}_features.csv", index=False)
    pd.DataFrame([{
        "win_start_idx": int(st),
        "win_end_idx": int(en),
        "t_start_s": float(times[st]),
        "t_end_s": float(times[min(en-1,len(times)-1)]),
        "dur_s": float(times[min(en-1,len(times)-1)] - times[st])
    } for st,en in merged]).to_csv(f"{args.out}_detections.csv", index=False)

    with open(f"{args.out}_meta.json","w") as f:
        json.dump({
            "input": {"tdms": args.tdms, "group": args.group, "channel": args.channel, "time_channel": args.time_channel,
                      "csv_ecg": args.in_csv, "rr_csv": args.rr_csv},
            "rpeak_mode": args.rpeak_mode,
            "fs_used": fs_used,
            "window_beats": args.window_beats,
            "auto_baseline_beats": args.auto_baseline_beats,
            "auto_pct": args.auto_pct,
            "thr_modcsi_filt_times_slope": thr_mod,
            "thr_csi_times_slope": thr_csi
        }, f, indent=2)

    print("[OK] Features, detections and meta written.")
    if args.plots:
        plt.figure()
        plt.plot(feat["t_end_s"].values, feat["modcsi_filt_times_slope"].values)
        plt.axhline(thr_mod, linestyle="--")
        plt.xlabel("Time (s)"); plt.ylabel("ModCSI_filt × slope"); plt.title("Feature")
        plt.savefig(f"{args.out}_modcsi_times_slope.png", dpi=160); plt.close()

        plt.figure()
        plt.plot(feat["t_end_s"].values, feat["csi_times_slope"].values)
        plt.axhline(thr_csi, linestyle="--")
        plt.xlabel("Time (s)"); plt.ylabel("CSI × slope"); plt.title("Feature")
        plt.savefig(f"{args.out}_csi_times_slope.png", dpi=160); plt.close()

        if len(rr_sec)>101:
            rr_win = rr_sec[-100:]
            plt.figure(); plt.scatter(rr_win[:-1], rr_win[1:], s=8)
            plt.xlabel("RR[i] (s)"); plt.ylabel("RR[i+1] (s)"); plt.title("Lorenz (sample)")
            plt.savefig(f"{args.out}_lorenz.png", dpi=160); plt.close()

if __name__=="__main__":
    main()
