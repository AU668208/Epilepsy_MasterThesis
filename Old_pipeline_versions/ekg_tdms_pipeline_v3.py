
# --- ekg_tdms_pipeline_v3.py ---
# Robust TDMS + annotation pipeline with Danish column handling and NaT-safe alignment.

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
    # Prefer first non-empty group
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


def _read_excel_any(path: str, header: Optional[Union[int, List[int]]] = 0, skiprows: Optional[List[int]] = None) -> pd.DataFrame:
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
    """
    Helper to peek at the header row and first rows. Does not modify anything.
    """
    try:
        df = _read_excel_any(path, header=header_guess)
    except Exception:
        df = _read_excel_any(path, header=None)
    # return first rows so user can inspect
    return df.head(10)


def _standardize_columns(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = s.replace("\n", " ").replace("\r", " ").replace("  ", " ")
        # normalize Danish letters are fine; keep as-is
        out.append(s)
    return out


def read_annotations(path: str,
                     header: Optional[Union[int, List[int]]] = 6,  # row 7 for many of your sheets
                     skiprows: Optional[List[int]] = None,
                     use_eeg: bool = True,
                     use_klinisk: bool = True,
                     default_duration_sec: float = 30.0) -> pd.DataFrame:
    """
    Reads an annotations file and extracts seizure events.
    - Assumes header at row 6 (0-indexed) by default (since your table starts on Excel row 7).
    - Looks for Danish variants of 'Anfaldstart'/'Anfaldstop' for both EEG og klinisk.
    - Also pulls 'anfaldstype' if present.
    Returns a long-format table:
      label ('EEG'/'Klinisk'), onset_time, offset_time, seizure_type (if any), plus raw columns.
    """
    df = _read_excel_any(path, header=header, skiprows=skiprows)
    df.columns = _standardize_columns(df.columns.tolist())

    # Candidate regex patterns
    patt_start_eeg = re.compile(r"^(anfald\s*start.*eeg|eeg.*anfald\s*start|anfaldstart.*eeg)$")
    patt_stop_eeg  = re.compile(r"^(anfald\s*stop.*eeg|eeg.*anfald\s*stop|anfaldstop.*eeg)$")
    patt_start_kl  = re.compile(r"^(anfald\s*start.*klin|klin.*anfald\s*start|anfaldstart.*klin)$")
    patt_stop_kl   = re.compile(r"^(anfald\s*stop.*klin|klin.*anfald\s*stop|anfaldstop.*klin)$")
    patt_type      = re.compile(r"^(anfalds?type|type)$")

    def find_col(patt: re.Pattern) -> Optional[str]:
        for c in df.columns:
            if patt.match(c):
                return c
        return None

    start_eeg = find_col(patt_start_eeg)
    stop_eeg  = find_col(patt_stop_eeg)
    start_kl  = find_col(patt_start_kl)
    stop_kl   = find_col(patt_stop_kl)
    col_type  = find_col(patt_type)

    # If regex fails, try looser contains-based search
    def find_loose(subs: List[str]) -> Optional[str]:
        for c in df.columns:
            if all(s in c for s in subs):
                return c
        return None

    if start_eeg is None: start_eeg = find_loose(["anfald", "start", "eeg"])
    if stop_eeg  is None: stop_eeg  = find_loose(["anfald", "stop",  "eeg"])
    if start_kl  is None: start_kl  = find_loose(["anfald", "start", "klin"])
    if stop_kl   is None: stop_kl   = find_loose(["anfald", "stop",  "klin"])
    if col_type  is None: col_type  = find_loose(["type"]) or find_loose(["anfald", "type"])

    rows = []

    def add_rows(kind: str, c_start: Optional[str], c_stop: Optional[str]):
        if c_start is None and c_stop is None:
            return
        # Parse times
        onset = pd.to_datetime(df[c_start], errors="coerce", dayfirst=True, utc=False) if c_start in df else pd.Series([pd.NaT]*len(df))
        offset= pd.to_datetime(df[c_stop],  errors="coerce", dayfirst=True, utc=False) if c_stop  in df else pd.Series([pd.NaT]*len(df))
        # If offset missing but onset exists, synthesize using default duration
        need_offset = offset.isna() & onset.notna()
        offset.loc[need_offset] = onset.loc[need_offset] + pd.to_timedelta(default_duration_sec, unit="s")
        # Build partial frame
        part = pd.DataFrame({
            "label": kind,
            "onset_time": onset,
            "offset_time": offset
        })
        if col_type in df:
            part["seizure_type"] = df[col_type]
        rows.append(part)

    if use_eeg:
        add_rows("EEG", start_eeg, stop_eeg)
    if use_klinisk:
        add_rows("Klinisk", start_kl, stop_kl)

    if not rows:
        # Fallback: try any columns named 'start'/'slut'
        start_any = next((c for c in df.columns if "start" in c), None)
        stop_any  = next((c for c in df.columns if "stop"  in c or "slut" in c), None)
        if start_any or stop_any:
            add_rows("EEG", start_any, stop_any)

    if not rows:
        raise ValueError(f"Kunne ikke finde kolonner for Anfaldstart/Anfaldstop (EEG/Klinisk). Kolonner fundet: {list(df.columns)}")

    out = pd.concat(rows, ignore_index=True)

    # Drop rows with no onset
    before = len(out)
    out = out[out["onset_time"].notna()].copy()
    dropped_onset = before - len(out)

    # If some offsets are still NaT (e.g., both missing), fill with default duration
    missing_offset = out["offset_time"].isna().sum()
    if missing_offset > 0:
        out.loc[out["offset_time"].isna(), "offset_time"] = out.loc[out["offset_time"].isna(),"onset_time"] + pd.to_timedelta(default_duration_sec, unit="s")

    if dropped_onset or missing_offset:
        warnings.warn(f"Annoteringer renset: droppede {dropped_onset} rækker uden onset; udfyldte {missing_offset} manglende offset.")

    # Keep original ordering index if useful
    out.reset_index(drop=True, inplace=True)
    return out


def align_annotations_to_samples(ann: pd.DataFrame, meta: RecordingMeta) -> pd.DataFrame:
    tz_cph = tz.gettz("Europe/Copenhagen")
    out = ann.copy()

    # Ensure datetime tz-aware
    for col in ["onset_time","offset_time"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        if getattr(out[col].dt.tz, 'zone', None) is None:
            out[col] = out[col].dt.tz_localize(tz_cph, ambiguous="NaT", nonexistent="NaT")

    # Drop rows still missing onset/offset
    before = len(out)
    out = out.dropna(subset=["onset_time", "offset_time"]).copy()
    dropped = before - len(out)
    if dropped:
        warnings.warn(f"Fjernede {dropped} rækker pga. NaT efter parsing.")

    # Clamp to recording duration
    duration_sec = meta.n_samples / meta.fs
    rec_end = meta.start_time + timedelta(seconds=duration_sec)
    out["onset_time"] = out["onset_time"].clip(lower=meta.start_time, upper=rec_end)
    out["offset_time"] = out["offset_time"].clip(lower=meta.start_time, upper=rec_end)

    # Compute indices (safe: times are non-NaT)
    dt_on = (out["onset_time"] - meta.start_time).dt.total_seconds()
    dt_off= (out["offset_time"] - meta.start_time).dt.total_seconds()
    out["onset_idx"] = (dt_on * meta.fs).round().astype("int64")
    out["offset_idx"] = (dt_off * meta.fs).round().astype("int64")

    # Clean anomalies
    out["offset_idx"] = np.maximum(out["offset_idx"], out["onset_idx"] + 1)
    out["onset_idx"] = np.clip(out["onset_idx"], 0, meta.n_samples - 1)
    out["offset_idx"] = np.clip(out["offset_idx"], 1, meta.n_samples)

    # Select final columns
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
