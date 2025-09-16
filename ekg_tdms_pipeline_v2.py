
# --- ekg_tdms_pipeline_v2.py ---
# Like v1, but with a more robust read_annotations() that handles "Unnamed: x" columns and flexible headers.

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
    grp = groups[0]
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


def _score_datetime_col(s: pd.Series) -> float:
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
    return float(dt.notna().mean())


def _score_numeric_col(s: pd.Series) -> float:
    num = pd.to_numeric(s, errors="coerce")
    return float(num.notna().mean())


def read_annotations(path: str,
                     start_col: Optional[Union[str,int]] = None,
                     end_col: Optional[Union[str,int]] = None,
                     label_col: Optional[Union[str,int]] = None,
                     header: Optional[Union[int,List[int]]] = 0,
                     skiprows: Optional[List[int]] = None,
                     default_duration_sec: float = 30.0) -> pd.DataFrame:
    """
    Robust reader for annotation files (.xls/.xlsx/.csv).
    - Allows specifying column names/indices (start_col, end_col, label_col).
    - If not provided, auto-detects columns even when headers are 'Unnamed: x' or merged.
    - Accepts absolute datetimes or relative seconds since recording start.
    Returns normalized DataFrame with onset/offset as either *_time or *_rel_sec and 'label'.
    """
    df = _read_excel_any(path, header=header, skiprows=skiprows)
    # If all/most columns are 'Unnamed', try treating the first non-empty row as header
    if sum([str(c).lower().startswith("unnamed") for c in df.columns]) >= max(2, int(0.5*len(df.columns))):
        # find the first row that contains any of our keyword headers
        keyword_candidates = ["start","start_tid","onset","onset_time","t_start","begin","begynd","anfald_start","starttime","start time",
                              "end","slut","offset","offset_time","t_end","finish","anfald_slut","endtime","end time",
                              "label","type","event","annotation","klasse","kategori","note","kommentar"]
        found_row = None
        for i in range(min(len(df), 15)):  # look at first 15 rows
            row_vals = [str(v).strip().lower() for v in df.iloc[i].values]
            if any(k in row_vals for k in keyword_candidates):
                found_row = i; break
        if found_row is not None:
            df2 = _read_excel_any(path, header=found_row, skiprows=None)
            df = df2

    # Standardize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If user provided explicit columns (by name or index), map them
    def col_by_any(x):
        if x is None: return None
        if isinstance(x, int):
            return df.columns[x] if 0 <= x < len(df.columns) else None
        x = str(x).lower().strip()
        # exact match first
        if x in df.columns: return x
        # try relaxed
        for c in df.columns:
            if x in c:
                return c
        return None

    start_col = col_by_any(start_col)
    end_col   = col_by_any(end_col)
    label_col = col_by_any(label_col)

    # Auto-pick if missing
    start_keys = ["start","start_tid","onset","onset_time","t_start","begin","begynd","anfald_start","starttime","start time"]
    end_keys   = ["end","slut","offset","offset_time","t_end","finish","anfald_slut","endtime","end time"]
    label_keys = ["label","type","event","annotation","klasse","kategori","note","kommentar"]

    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    start_col = start_col or pick(start_keys)
    end_col   = end_col or pick(end_keys)
    label_col = label_col or pick(label_keys)

    # If still no start/end, auto-detect by content types
    if start_col is None or (end_col is None and "duration" not in df.columns):
        scores_dt = {c:_score_datetime_col(df[c]) for c in df.columns}
        scores_num= {c:_score_numeric_col(df[c]) for c in df.columns}
        # Candidate datetime columns: >60% parseable
        cand_dt = [c for c,s in scores_dt.items() if s >= 0.6]
        # Candidate numeric columns (for relative sec or duration): >60% numeric
        cand_num = [c for c,s in scores_num.items() if s >= 0.6]
        if start_col is None and cand_dt:
            start_col = cand_dt[0]
        # Prefer another datetime for end; otherwise a duration-like numeric
        if end_col is None:
            if len(cand_dt) >= 2:
                end_col = cand_dt[1]
            elif "duration" in df.columns:
                end_col = "duration"
            elif cand_num:
                # use as duration seconds
                end_col = cand_num[0]  # later treated as duration if start is datetime

    # Build normalized frame
    norm = pd.DataFrame()
    if label_col is None:
        norm["label"] = "seizure"
    else:
        norm["label"] = df[label_col]

    def parse_series(s):
        # Try numeric seconds
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.8:
            return num.astype(float), "relative_seconds"
        # else datetime
        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)
        return dt, "absolute_datetime"

    if start_col is None:
        raise ValueError(f"Could not auto-detect a start/onset column. Columns: {list(df.columns)}")

    start_vals, start_kind = parse_series(df[start_col])
    if end_col is not None:
        end_vals, end_kind = parse_series(df[end_col])
    else:
        end_vals, end_kind = None, None

    if start_kind == "relative_seconds":
        norm["onset_rel_sec"] = start_vals
    else:
        norm["onset_time"] = pd.to_datetime(start_vals, utc=False, dayfirst=True)

    if end_vals is not None:
        if end_kind == "relative_seconds" and start_kind == "absolute_datetime":
            # treat as duration seconds
            norm["offset_time"] = pd.to_datetime(norm["onset_time"]) + pd.to_timedelta(end_vals, unit="s")
        elif end_kind == "relative_seconds":
            norm["offset_rel_sec"] = end_vals
        else:
            norm["offset_time"] = pd.to_datetime(end_vals, utc=False, dayfirst=True)
    else:
        # synthesize using default duration
        if "onset_rel_sec" in norm:
            norm["offset_rel_sec"] = norm["onset_rel_sec"] + float(default_duration_sec)
        else:
            norm["offset_time"] = norm["onset_time"] + pd.to_timedelta(float(default_duration_sec), unit="s")

    # Preserve extra columns
    for c in df.columns:
        if c not in {start_col, end_col, label_col}:
            norm[c] = df[c]

    return norm


def align_annotations_to_samples(ann: pd.DataFrame, meta: RecordingMeta) -> pd.DataFrame:
    tz_cph = tz.gettz("Europe/Copenhagen")
    out = ann.copy()
    if "onset_time" not in out.columns and "onset_rel_sec" in out.columns:
        out["onset_time"] = pd.to_datetime(meta.start_time) + pd.to_timedelta(out["onset_rel_sec"], unit="s")
    if "offset_time" not in out.columns and "offset_rel_sec" in out.columns:
        out["offset_time"] = pd.to_datetime(meta.start_time) + pd.to_timedelta(out["offset_rel_sec"], unit="s")
    for col in ["onset_time","offset_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            if getattr(out[col].dt.tz, 'zone', None) is None:
                out[col] = out[col].dt.tz_localize(tz_cph, ambiguous="NaT", nonexistent="NaT")
    duration_sec = meta.n_samples / meta.fs
    rec_end = meta.start_time + timedelta(seconds=duration_sec)
    out["onset_time"] = out["onset_time"].clip(lower=meta.start_time, upper=rec_end)
    out["offset_time"] = out["offset_time"].clip(lower=meta.start_time, upper=rec_end)
    out["onset_idx"] = ((out["onset_time"] - meta.start_time).dt.total_seconds() * meta.fs).round().astype(int)
    out["offset_idx"] = ((out["offset_time"] - meta.start_time).dt.total_seconds() * meta.fs).round().astype(int)
    out["offset_idx"] = np.maximum(out["offset_idx"], out["onset_idx"] + 1)
    out["onset_idx"] = np.clip(out["onset_idx"], 0, meta.n_samples - 1)
    out["offset_idx"] = np.clip(out["offset_idx"], 1, meta.n_samples)
    keep = ["label","onset_time","offset_time","onset_idx","offset_idx"]
    extra = [c for c in out.columns if c not in keep]
    return out[keep + extra]


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
