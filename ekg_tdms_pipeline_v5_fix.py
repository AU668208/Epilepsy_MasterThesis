
# --- ekg_tdms_pipeline_v5_fix.py ---
# v5 with robust timezone localization compatible with older pandas (no errors= in tz_localize).

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
            return dt.tz_localize("Europe/Copenhagen")
        return dt
    except Exception:
        return None


def load_tdms(path: str, channel_hint: Optional[str] = None) -> Tuple[np.ndarray, 'RecordingMeta']:
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


# ---- Danish date/time helpers (same as v4/v5) ----
def _standardize_columns(cols: List[str]) -> List[str]:
    return [str(c).strip().lower().replace("\n"," ").replace("\r"," ").replace("  "," ") for c in cols]


def _clean_time_candidate(val) -> Optional[str]:
    if pd.isna(val): return None
    s = str(val).strip()
    if s.endswith("."): s = s[:-1]
    s2 = s.replace(".", ":")
    if "-" in s2 and ":" not in s2: return None
    parts = s2.split(":")
    if len(parts) < 2 or len(parts) > 3: return None
    try: nums = [int(p) for p in parts]
    except Exception: return None
    if len(nums) == 2: h, m = nums; ssec = 0
    else: h, m, ssec = nums
    if not (0 <= h <= 29 and 0 <= m < 60 and 0 <= ssec < 60): return None
    return f"{h:02d}:{m:02d}:{ssec:02d}"


import re
def _parse_danish_date(val) -> Optional[pd.Timestamp]:
    if pd.isna(val): return None
    s = str(val).strip()
    s = re.sub(r"^(\d{1,2}\.\d{1,2})-(\d{4})$", r"\1.\2", s)
    try: return pd.to_datetime(s, errors="raise", dayfirst=True, utc=False)
    except Exception: return None


def _combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    dates = date_series.apply(_parse_danish_date).fillna(method="ffill")
    times = time_series.apply(_clean_time_candidate)
    out = []
    for d, t in zip(dates, times):
        if pd.isna(d) or t is None: out.append(pd.NaT)
        else:
            try: dt = pd.to_datetime(f"{d.date()} {t}", dayfirst=False, utc=False)
            except Exception: dt = pd.NaT
            out.append(dt)
    return pd.Series(out, index=time_series.index)


# ---- Structured annotations ----
def _read_excel_any(path: str, header: Optional[Union[int, List[int]]] = 6, skiprows: Optional[List[int]] = None) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    engine = None
    if suffix == ".xls": engine = "xlrd"
    elif suffix == ".xlsx": engine = "openpyxl"
    try: return pd.read_excel(path, engine=engine, header=header, skiprows=skiprows)
    except Exception: return pd.read_excel(path, header=header, skiprows=skiprows)


def read_annotations_structured(path: str, header: int = 6) -> pd.DataFrame:
    df = _read_excel_any(path, header=header)
    df.columns = _standardize_columns(df.columns.tolist())

    def find_col(needles: List[str]) -> Optional[str]:
        for c in df.columns:
            if all(n in c for n in needles): return c
        return None

    col_id   = find_col(["anfald", "nr"]) or find_col(["nr"])
    col_date = find_col(["dato"])
    if not col_date:
        raise ValueError(f"Fandt ikke 'Dato' kolonne. Kolonner: {list(df.columns)}")
    s_klin = find_col(["anfaldsstart","klin"]) or find_col(["anfald","start","klin"])
    e_klin = find_col(["anfaldsstop","klin"])  or find_col(["anfald","stop","klin"])
    s_eeg  = find_col(["anfaldsstart","eeg"])  or find_col(["anfald","start","eeg"])
    e_eeg  = find_col(["anfaldsstop","eeg"])   or find_col(["anfald","stop","eeg"])
    col_type = find_col(["anfaldstype"]) or find_col(["type"])
    col_notes= find_col(["bemærk"]) or find_col(["bem"])

    out = pd.DataFrame({
        "seizure_id": df[col_id] if col_id else np.arange(1, len(df)+1),
        "date": df[col_date]
    })
    out["eeg_onset_time"]  = _combine_date_time(df[col_date], df[s_eeg]) if s_eeg in df else pd.NaT
    out["eeg_offset_time"] = _combine_date_time(df[col_date], df[e_eeg]) if e_eeg in df else pd.NaT
    out["klin_onset_time"]  = _combine_date_time(df[col_date], df[s_klin]) if s_klin in df else pd.NaT
    out["klin_offset_time"] = _combine_date_time(df[col_date], df[e_klin]) if e_klin in df else pd.NaT
    out["seizure_type"] = (df[col_type].astype(str).str.strip().str.strip(",;:. ") if col_type else pd.NA)
    out["notes"] = (df[col_notes] if col_notes else pd.NA)

    has_any_onset = out["eeg_onset_time"].notna() | out["klin_onset_time"].notna()
    out = out[has_any_onset].copy().reset_index(drop=True)
    return out


def structured_to_long(ann_struct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in ann_struct.iterrows():
        if pd.notna(r.get("eeg_onset_time")):
            rows.append({"seizure_id": r["seizure_id"], "label": "EEG",
                         "onset_time": r["eeg_onset_time"], "offset_time": r.get("eeg_offset_time", pd.NaT),
                         "seizure_type": r.get("seizure_type", pd.NA)})
        if pd.notna(r.get("klin_onset_time")):
            rows.append({"seizure_id": r["seizure_id"], "label": "Klinisk",
                         "onset_time": r["klin_onset_time"], "offset_time": r.get("klin_offset_time", pd.NaT),
                         "seizure_type": r.get("seizure_type", pd.NA)})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["seizure_id","label","onset_time","offset_time","seizure_type"])
    missing_off = df["offset_time"].isna() & df["onset_time"].notna()
    df.loc[missing_off, "offset_time"] = df.loc[missing_off, "onset_time"] + pd.to_timedelta(30, unit="s")
    return df


def _localize_series_safe(series: pd.Series, tzinfo) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    out = []
    for v in s:
        if pd.isna(v):
            out.append(pd.NaT); continue
        try:
            if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
                # naive -> localize
                out.append(v.tz_localize(tzinfo))
            else:
                # already tz-aware -> convert
                out.append(v.tz_convert(tzinfo))
        except Exception:
            # fallback: treat as invalid
            out.append(pd.NaT)
    return pd.Series(out, index=series.index)


def align_structured_to_samples(ann_struct: pd.DataFrame, meta: RecordingMeta) -> pd.DataFrame:
    tz_cph = tz.gettz("Europe/Copenhagen")
    out = ann_struct.copy()

    for col in ["eeg_onset_time","eeg_offset_time","klin_onset_time","klin_offset_time"]:
        if col in out.columns:
            out[col] = _localize_series_safe(out[col], tz_cph)

    duration_sec = meta.n_samples / meta.fs
    rec_end = meta.start_time + timedelta(seconds=duration_sec)

    def to_idx(series_time: pd.Series) -> pd.Series:
        s = pd.to_datetime(series_time, errors="coerce")
        s = s.clip(lower=meta.start_time, upper=rec_end)
        dt = (s - meta.start_time).dt.total_seconds()
        return (dt * meta.fs).round()

    if "eeg_onset_time" in out: out["eeg_onset_idx"] = to_idx(out["eeg_onset_time"]).astype("Int64")
    if "eeg_offset_time" in out: out["eeg_offset_idx"] = to_idx(out["eeg_offset_time"]).astype("Int64")
    if "klin_onset_time" in out: out["klin_onset_idx"] = to_idx(out["klin_onset_time"]).astype("Int64")
    if "klin_offset_time" in out: out["klin_offset_idx"] = to_idx(out["klin_offset_time"]).astype("Int64")

    for pref in ["eeg","klin"]:
        oi, fi = f"{pref}_onset_idx", f"{pref}_offset_idx"
        if oi in out and fi in out:
            valid = out[oi].notna() & out[fi].notna()
            out.loc[valid & (out[fi] <= out[oi]), fi] = out.loc[valid & (out[fi] <= out[oi]), oi] + 1

    return out


def count_seizures(ann_struct: pd.DataFrame) -> int:
    return int(((ann_struct["eeg_onset_time"].notna()) | (ann_struct["klin_onset_time"].notna())).sum())


def plot_overview_structured(meta: RecordingMeta, ann_struct_aligned: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, max(2, 0.35*len(ann_struct_aligned)+1)))
    y = np.arange(len(ann_struct_aligned))
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{int(r.seizure_id)}" if pd.notna(r.seizure_id) else f"{i+1}" for i, r in ann_struct_aligned.iterrows()])
    for i, r in ann_struct_aligned.iterrows():
        if pd.notna(r.get("eeg_onset_idx")) and pd.notna(r.get("eeg_offset_idx")):
            s = r["eeg_onset_idx"]/meta.fs; e = r["eeg_offset_idx"]/meta.fs
            ax.add_patch(Rectangle((s, i-0.3), max(1e-6, e-s), 0.25, alpha=0.3))
        if pd.notna(r.get("klin_onset_idx")) and pd.notna(r.get("klin_offset_idx")):
            s = r["klin_onset_idx"]/meta.fs; e = r["klin_offset_idx"]/meta.fs
            ax.add_patch(Rectangle((s, i+0.05), max(1e-6, e-s), 0.25, alpha=0.5))
    ax.set_xlabel("Tid (sekunder fra start)"); ax.set_ylabel("Anfald (ID)")
    ax.set_title("Oversigt: EEG vs. Klinisk per anfald (samme række)")
    ax.set_ylim(-1, len(ann_struct_aligned)); ax.set_xlim(0, meta.n_samples/meta.fs)
    plt.tight_layout(); plt.show()
