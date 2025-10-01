
# --- ekg_tdms_pipeline_v4.py ---
# Robust TDMS + annotation pipeline (Danish date/time) with explicit date+time merging.

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
from dateutil import tz
import warnings
import os
import re

try:
    from scipy.signal import welch, iirnotch, filtfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from nptdms import TdmsFile
    NPTDMS_AVAILABLE = True
except Exception:
    NPTDMS_AVAILABLE = False


@dataclass
class RecordingMeta:
    fs: float
    start_time: datetime
    n_samples: int
    channel_name: str
    units: Optional[str] = None
    path: Optional[str] = None


def _to_datetime_safe(x) -> Optional[datetime]:
    if pd.isna(x):
        return None
    try:
        dt = pd.to_datetime(x, errors="raise", dayfirst=True, utc=False)
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.tz_localize("Europe/Copenhagen", ambiguous="NaT", nonexistent="NaT")
        return dt
    except Exception:
        return None


def load_tdms(path: str, channel_hint: Optional[str] = None) -> Tuple[np.ndarray, RecordingMeta]:
    if not NPTDMS_AVAILABLE:
        raise ImportError("nptdms is not installed. Please `pip install nptdms`.")
    tf = TdmsFile.read(path)
    groups = tf.groups()
    if not groups:
        raise ValueError("No groups found in TDMS.")
    grp = next((g for g in groups if g.channels()), groups[0])
    channels = grp.channels() or [ch for g in tf.groups() for ch in g.channels()]
    if not channels:
        raise ValueError("No channels found in TDMS.")
    ch = None
    if channel_hint:
        for c in channels:
            if channel_hint.lower() in c.name.lower():
                ch = c; break
    ch = ch or channels[0]
    data = np.asarray(ch[:], dtype=float)
    n_samples = data.shape[0]
    props, grp_props, file_props = ch.properties, grp.properties, tf.properties
    units = props.get("unit_string") or props.get("NI_UnitDescription") or props.get("unit") or None
    fs = None
    incr = props.get("wf_increment") or props.get("NI_wfIncrement")
    if incr:
        try: fs = 1.0/float(incr)
        except Exception: fs = None
    if fs is None:
        for key in ["fs","sampling_frequency","Sample Rate","NI_SampleRate"]:
            if key in props:
                try: fs = float(props[key]); break
                except Exception: pass
    start_time = None
    for source in (props, grp_props, file_props):
        if start_time is not None: break
        for key in ["wf_start_time","NI_wfStartTime","start_time","Start Time","Timestamp"]:
            if key in source:
                st = source[key]
                start_time = st if isinstance(st, datetime) else _to_datetime_safe(st)
                break
    if fs is None:
        fs = 512.0
        warnings.warn("Could not read fs from TDMS; defaulting to 512 Hz.")
    if start_time is None:
        start_time = pd.Timestamp.now(tz="Europe/Copenhagen").to_pydatetime()
        warnings.warn("Could not read start time; defaulting to now().")
    if getattr(start_time,'tzinfo',None) is None:
        start_time = start_time.replace(tzinfo=tz.gettz("Europe/Copenhagen"))
    meta = RecordingMeta(fs=float(fs), start_time=start_time, n_samples=int(n_samples),
                         channel_name=str(ch.name), units=units, path=path)
    return data, meta


def _read_excel_any(path: str, header: Optional[Union[int, List[int]]] = 6, skiprows: Optional[List[int]] = None) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    engine = None
    if suffix == ".xls":
        engine = "xlrd"
    elif suffix == ".xlsx":
        engine = "openpyxl"
    try:
        return pd.read_excel(path, engine=engine, header=header, skiprows=skiprows)
    except Exception:
        return pd.read_excel(path, header=header, skiprows=skiprows)


def preview_annotations(path: str, header_guess: int = 6) -> pd.DataFrame:
    try:
        df = _read_excel_any(path, header=header_guess)
    except Exception:
        df = _read_excel_any(path, header=None)
    return df.head(20)


def _standardize_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = s.replace("\n"," ").replace("\r"," ").replace("  "," ")
        out.append(s)
    return out


def _clean_time_candidate(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s.endswith("."):
        s = s[:-1]
    s2 = s.replace(".", ":")
    if "-" in s2 and ":" not in s2:
        return None
    parts = s2.split(":")
    if len(parts) < 2 or len(parts) > 3:
        return None
    try:
        nums = [int(p) for p in parts]
    except Exception:
        return None
    if len(nums) == 2:
        h, m = nums; ssec = 0
    else:
        h, m, ssec = nums
    if not (0 <= h <= 29 and 0 <= m < 60 and 0 <= ssec < 60):
        return None
    return f"{h:02d}:{m:02d}:{ssec:02d}"


def _parse_danish_date(val) -> Optional[pd.Timestamp]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"^(\d{1,2}\.\d{1,2})-(\d{4})$", r"\1.\2", s)
    try:
        dt = pd.to_datetime(s, errors="raise", dayfirst=True, utc=False)
        return dt
    except Exception:
        return None


def _combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = date_series.apply(_parse_danish_date).fillna(method="ffill")
    times = time_series.apply(_clean_time_candidate)
    out = []
    for d, t in zip(dates, times):
        if pd.isna(d) or t is None:
            out.append(pd.NaT)
        else:
            try:
                dt = pd.to_datetime(f"{d.date()} {t}", dayfirst=False, utc=False)
            except Exception:
                dt = pd.NaT
            out.append(dt)
    return pd.Series(out, index=time_series.index)


def read_annotations(path: str,
                     header: Optional[Union[int, List[int]]] = 6,
                     use_eeg: bool = True,
                     use_klinisk: bool = True,
                     default_duration_sec: float = 30.0) -> pd.DataFrame:
    df = _read_excel_any(path, header=header)
    df.columns = _standardize_columns(df.columns.tolist())

    def find_col(needles: List[str]) -> Optional[str]:
        for c in df.columns:
            if all(n in c for n in needles):
                return c
        return None

    col_date = find_col(["dato"])
    if not col_date:
        raise ValueError(f"Fandt ikke en 'Dato' kolonne. Kolonner: {list(df.columns)}")

    start_klin = find_col(["anfaldsstart", "klin"]) or find_col(["anfald", "start", "klin"])
    stop_klin  = find_col(["anfaldsstop", "klin"])  or find_col(["anfald", "stop", "klin"])
    start_eeg  = find_col(["anfaldsstart", "eeg"])  or find_col(["anfald", "start", "eeg"])
    stop_eeg   = find_col(["anfaldsstop", "eeg"])   or find_col(["anfald", "stop", "eeg"])
    col_type   = find_col(["anfaldstype"]) or find_col(["type"])

    rows = []

    def add_rows(kind: str, start_col: Optional[str], stop_col: Optional[str]):
        if start_col is None and stop_col is None:
            return
        onset = _combine_date_time(df[col_date], df[start_col]) if start_col in df else pd.Series(pd.NaT, index=df.index)
        offset= _combine_date_time(df[col_date], df[stop_col])  if stop_col  in df else pd.Series(pd.NaT, index=df.index)
        part = pd.DataFrame({"label": kind, "onset_time": onset, "offset_time": offset})
        rows.append(part)

    if use_eeg:
        add_rows("EEG", start_eeg, stop_eeg)
    if use_klinisk:
        add_rows("Klinisk", start_klin, stop_klin)

    if not rows:
        raise ValueError(f"Kunne ikke finde kolonner for Anfaldsstart/-stop (EEG/Klinisk). Kolonner: {list(df.columns)}")

    out = pd.concat(rows, ignore_index=True)

    if col_type:
        typ = df[col_type].astype(str).str.strip().str.strip(",;:. ")
        out["seizure_type"] = np.resize(typ.values, out.shape[0])

    missing_off = out["offset_time"].isna() & out["onset_time"].notna()
    out.loc[missing_off, "offset_time"] = out.loc[missing_off, "onset_time"] + pd.to_timedelta(default_duration_sec, unit="s")

    out = out[out["onset_time"].notna()].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def align_annotations_to_samples(ann: pd.DataFrame, meta: RecordingMeta) -> pd.DataFrame:
    tz_cph = tz.gettz("Europe/Copenhagen")
    out = ann.copy()
    for col in ["onset_time","offset_time"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        if getattr(out[col].dt.tz, 'zone', None) is None:
            out[col] = out[col].dt.tz_localize(tz_cph, ambiguous="NaT", nonexistent="NaT")
    out = out.dropna(subset=["onset_time","offset_time"]).copy()
    duration_sec = meta.n_samples / meta.fs
    rec_end = meta.start_time + timedelta(seconds=duration_sec)
    out["onset_time"] = out["onset_time"].clip(lower=meta.start_time, upper=rec_end)
    out["offset_time"] = out["offset_time"].clip(lower=meta.start_time, upper=rec_end)
    dt_on = (out["onset_time"] - meta.start_time).dt.total_seconds()
    dt_off= (out["offset_time"] - meta.start_time).dt.total_seconds()
    out["onset_idx"] = (dt_on * meta.fs).round().astype("int64")
    out["offset_idx"] = (dt_off * meta.fs).round().astype("int64")
    out["offset_idx"] = np.maximum(out["offset_idx"], out["onset_idx"] + 1)
    out["onset_idx"] = np.clip(out["onset_idx"], 0, meta.n_samples - 1)
    out["offset_idx"] = np.clip(out["offset_idx"], 1, meta.n_samples)
    keep = ["label","onset_time","offset_time","onset_idx","offset_idx"]
    if "seizure_type" in out.columns:
        keep.append("seizure_type")
    extras = [c for c in out.columns if c not in keep]
    return out[keep + extras]


def _downsample_for_plot(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1: return x
    n = (x.shape[0] // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def plot_overview(meta: RecordingMeta, ann_idx: pd.DataFrame):
    total_sec = meta.n_samples / meta.fs
    fig, ax = plt.subplots(figsize=(10,2))
    ax.add_patch(Rectangle((0,0), total_sec, 1, fill=False))
    for _, row in ann_idx.iterrows():
        s = row["onset_idx"]/meta.fs; e = row["offset_idx"]/meta.fs
        ax.add_patch(Rectangle((s,0), e-s, 1, alpha=0.3))
    ax.set_xlim(0, total_sec); ax.set_yticks([])
    ax.set_xlabel("Tid (sekunder fra start)")
    ax.set_title("Oversigt: markerede anfald")
    plt.show()


def plot_segment_with_annotations(sig: np.ndarray, meta: RecordingMeta, ann_idx: pd.DataFrame,
                                  t_start_sec: float, t_window_sec: float = 60.0, max_points: int = 20000):
    i0 = int(max(0, t_start_sec*meta.fs)); i1 = int(min(meta.n_samples, i0 + t_window_sec*meta.fs))
    segment = sig[i0:i1]
    factor = max(1, int(np.ceil(len(segment)/max_points)))
    y = _downsample_for_plot(segment, factor)
    t = np.linspace(i0/meta.fs, i1/meta.fs, num=len(y), endpoint=False)
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(t, y)
    for _, row in ann_idx.iterrows():
        s = row["onset_idx"]/meta.fs; e = row["offset_idx"]/meta.fs
        if e >= t[0] and s <= t[-1]:
            ax.axvspan(max(s, t[0]), min(e, t[-1]), alpha=0.2)
    ax.set_xlabel("Tid (s)"); ax.set_ylabel(meta.units or "Amplitude")
    ax.set_title(f"Signaludsnit {t_start_sec:.1f}s–{t_start_sec + t_window_sec:.1f}s ({meta.channel_name})")
    plt.show()


def compute_noise_report(sig: np.ndarray, meta: RecordingMeta, line_freq: float = 50.0) -> Dict[str,float]:
    out = {}
    sig = np.asarray(sig, dtype=float)
    out["rms"] = float(np.sqrt(np.mean(sig**2)))
    if not SCIPY_AVAILABLE:
        out["line_noise_ratio"] = np.nan; out["hf_noise_ratio"] = np.nan
        return out
    nperseg = min(int(4*meta.fs), len(sig))
    if nperseg < 16: nperseg = len(sig)
    f, Pxx = welch(sig, fs=meta.fs, nperseg=nperseg)
    def band(f1,f2):
        m = (f>=f1)&(f<=f2)
        return float(np.trapz(Pxx[m], f[m])) if np.any(m) else 0.0
    bp_base = band(0.5, 40.0); bp_line = band(line_freq-2.0, line_freq+2.0); bp_hf = band(40.0, 100.0)
    out["line_noise_ratio"] = float(bp_line/(bp_base+1e-12))
    out["hf_noise_ratio"]    = float(bp_hf/(bp_base+1e-12))
    try:
        b,a = iirnotch(w0=line_freq/(meta.fs/2), Q=30.0)
        y = filtfilt(b,a,sig)
        out["rms_after_notch"] = float(np.sqrt(np.mean(y**2)))
    except Exception:
        out["rms_after_notch"] = np.nan
    return out


def seizure_time_of_day_stats(ann_idx: pd.DataFrame) -> pd.DataFrame:
    tz_cph = tz.gettz("Europe/Copenhagen")
    df = ann_idx.copy()
    df["onset_time"] = pd.to_datetime(df["onset_time"]).dt.tz_convert(tz_cph)
    df["hour"] = df["onset_time"].dt.hour + df["onset_time"].dt.minute/60.0
    df["period"] = np.where((df["hour"]>=22.0)|(df["hour"]<6.0), "Nat", "Dag")
    counts = df["period"].value_counts(dropna=False)
    print("Antal anfald (Dag/Nat):")
    for k in ["Dag","Nat"]:
        print(f"  {k}: {int(counts.get(k,0))}")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(df["hour"], bins=24, range=(0,24))
    ax.set_xlabel("Klokkeslæt (timer)"); ax.set_ylabel("Antal anfald")
    ax.set_title("Tidspunkt for anfald på døgnet")
    plt.show()
    return df[["onset_time","offset_time","period","hour","label"]]
