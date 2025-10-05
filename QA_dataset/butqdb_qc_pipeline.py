#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BUT QDB — Ultra-light Noise Baseline (focus on 1/2/3 noise classes)
"""

import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import warnings
import wfdb

warnings.filterwarnings("ignore", category=RuntimeWarning)

def read_ecg_from_base(base_path: Path):
    base = str(base_path) + "_ECG"
    rec = wfdb.rdrecord(base, channels=[0])
    fs = float(rec.fs)
    x = rec.p_signal[:, 0].astype(np.float64)
    return x, fs

def read_butqdb_annotations(csv_path: Path, n_samples: int):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 12:
        raise ValueError(f"Expected 12 columns in {csv_path.name}, got {df.shape[1]}")
    starts = df.iloc[:, 9].to_numpy(dtype=np.int64)
    stops  = df.iloc[:,10].to_numpy(dtype=np.int64)
    clss   = df.iloc[:,11].to_numpy(dtype=np.int64)
    mask = np.zeros(n_samples, dtype=np.int8)
    for s,e,c in zip(starts, stops, clss):
        c = int(c)
        if c == 0:
            continue
        s = max(0, int(s)); e = min(n_samples-1, int(e))
        if e > s:
            mask[s:e] = c
    return mask

def cheap_downsample(x, fs, target=200.0):
    if fs <= target + 1:
        return x, fs
    q = int(max(1, np.floor(fs / target)))
    if q <= 1:
        return x, fs
    return x[::q], fs / q

def welch_psd(x, fs):
    n = len(x)
    nperseg = int(min(max(128, fs), n))
    if nperseg < 16:
        nperseg = min(16, n)
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return f, Pxx

def band_power_ratio(f, Pxx, lo, hi, base_lo, base_hi):
    band = (f >= lo) & (f <= hi)
    base = (f >= base_lo) & (f <= base_hi)
    bp = np.trapz(Pxx[band], f[band]) if np.any(band) else 0.0
    bt = np.trapz(Pxx[base], f[base]) if np.any(base) else 0.0
    return float(bp / bt) if bt > 0 else 0.0

def noise_features(x, fs):
    xd, fs_d = cheap_downsample(x - np.mean(x), fs, target=200.0)
    f, Pxx = welch_psd(xd, fs_d)
    feats = {}
    feats["rms"] = float(np.sqrt(np.mean(xd**2)))
    feats["bw_ratio"] = band_power_ratio(f, Pxx, 0.1, 0.5, 0.1, 40.0)
    feats["hf_ratio"] = band_power_ratio(f, Pxx, 20.0, 40.0, 0.1, 40.0)
    feats["pl_ratio"] = band_power_ratio(f, Pxx, 45.0, 55.0, 0.1, 60.0)
    sgn = np.signbit(xd)
    zc = np.count_nonzero(sgn[1:] != sgn[:-1])
    feats["zcr"] = float(zc / max(1, len(xd)-1))
    return feats

def window_indices(n, fs, win_sec, step_sec):
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    starts = np.arange(0, max(1, n - win + 1), step, dtype=int)
    ends = np.minimum(starts + win, n)
    return list(zip(starts, ends))

def label_window(mask, s, e):
    seg = mask[s:e]
    if seg.size == 0:
        return 0
    vals, counts = np.unique(seg[seg > 0], return_counts=True)
    if len(vals) == 0:
        return 0
    return int(vals[np.argmax(counts)])

def find_header(subdir: Path, rid: str):
    for name in [f"{rid}_ECG.hea", f"{rid}_Ecg.hea", f"{rid}_ecg.hea"]:
        p = subdir / name
        if p.exists():
            return p
    hits = list(subdir.glob("*_ECG.hea")) or list(subdir.glob("*_Ecg.hea")) or list(subdir.glob("*_ecg.hea"))
    return hits[0] if hits else None

def find_annotations(subdir: Path, rid: str):
    p = subdir / f"{rid}_annotations.csv"
    if p.exists():
        return p
    hits = [q for q in subdir.glob("*.csv") if "annot" in q.name.lower()]
    return hits[0] if hits else None

def discover_by_subfolder(root: Path, only_ids=None):
    root = Path(root)
    if only_ids is not None:
        want = set(map(str, only_ids))
        subs = [d for d in root.iterdir() if d.is_dir() and d.name in want]
    else:
        subs = [d for d in root.iterdir() if d.is_dir()]
    items = []
    for sub in sorted(subs, key=lambda p: p.name):
        rid = sub.name
        hea = find_header(sub, rid)
        ann = find_annotations(sub, rid)
        if not hea or not ann:
            continue
        base = hea.with_name(rid)
        items.append({"rid": rid, "base": base, "ann": ann})
    return items

def build_features_for_item(item, win_sec=8.0, step_sec=2.0):
    rid, base, ann = item["rid"], item["base"], item["ann"]
    ecg, fs = read_ecg_from_base(base)
    mask = read_butqdb_annotations(ann, len(ecg))
    rows = []
    for s,e in window_indices(len(ecg), fs, win_sec, step_sec):
        if np.all(mask[s:e] == 0):
            continue
        feats = noise_features(ecg[s:e], fs)
        feats["start"] = s; feats["end"] = e; feats["fs"] = fs; feats["record"] = rid
        feats["y3"] = label_window(mask, s, e)
        rows.append(feats)
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame):
    agg = df.groupby("y3")[["rms","bw_ratio","hf_ratio","pl_ratio","zcr"]].median().rename_axis("class").reset_index()
    cnt = df["y3"].value_counts().sort_index().rename("n").reset_index().rename(columns={"index":"class"})
    return pd.merge(agg, cnt, on="class", how="outer").sort_values("class")

def plot_quick(df, outdir):
    fig, ax = plt.subplots()
    data = [df[df["y3"]==c]["hf_ratio"].dropna() for c in [1,2,3]]
    ax.boxplot(data, labels=["1","2","3"], showfliers=False)
    ax.set_title("HF ratio by class"); ax.set_xlabel("Consensus class"); ax.set_ylabel("HF ratio (20–40 Hz / total 0.1–40)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "box_hf_ratio.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots()
    data = [df[df["y3"]==c]["bw_ratio"].dropna() for c in [1,2,3]]
    ax.boxplot(data, labels=["1","2","3"], showfliers=False)
    ax.set_title("BW ratio by class"); ax.set_xlabel("Consensus class"); ax.set_ylabel("BW ratio (0.1–0.5 Hz / total)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "box_bw_ratio.png"), dpi=150); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="BUT QDB — Ultra-light Noise Baseline")
    ap.add_argument("--root", required=True, help="Top-level folder containing one subfolder per record (e.g., 100001)")
    ap.add_argument("--records", nargs="+", help="Optional list of subfolder names to process")
    ap.add_argument("--win", type=float, default=8.0, help="Window length (seconds)")
    ap.add_argument("--step", type=float, default=2.0, help="Step/stride (seconds)")
    ap.add_argument("--outdir", default="./outputs_noise", help="Output folder")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    items = discover_by_subfolder(args.root, only_ids=args.records)
    if not items:
        raise SystemExit("No records discovered under --root. Expecting ROOT/<rid>/<rid>_ECG.* + <rid>_annotations.csv")

    summaries = []
    for it in items:
        rid = it["rid"]
        print(f"[+] {rid}")
        df = build_features_for_item(it, win_sec=args.win, step_sec=args.step)
        if df.empty:
            print(f"[!] {rid} produced no labeled windows; skipping")
            continue
        out_csv = Path(args.outdir)/f"{rid}_noise_features.csv"
        df.to_csv(out_csv, index=False)
        sm = summarize(df); sm.insert(0, "record", rid)
        summaries.append(sm)

    if not summaries:
        raise SystemExit("No data produced.")

    big = pd.concat(summaries, ignore_index=True)
    big.to_csv(Path(args.outdir)/"summary_per_record.csv", index=False)

    if len(items) == 1:
        df = pd.read_csv(Path(args.outdir)/f"{items[0]['rid']}_noise_features.csv")
        plot_quick(df, args.outdir)

    print(f"[✓] Done. Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()
