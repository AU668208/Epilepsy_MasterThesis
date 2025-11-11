
# --- ekg_tdms_pipeline.py ---
# Utilities to load TDMS EKG data, merge seizure annotations, visualize, and run basic analyses.
#
# Requirements (install in your environment if missing):
#   pip install nptdms pandas numpy scipy matplotlib xlrd openpyxl pytz python-dateutil
#
# Usage in a notebook:
#   from ekg_tdms_pipeline import (
#       load_tdms, read_annotations, align_annotations_to_samples,
#       plot_overview, plot_segment_with_annotations,
#       compute_noise_report, seizure_time_of_day_stats,
#       compute_signal_trends
#   )
#   tdms_path = "path/to/patient_data.tdms"
#   ann_path  = "path/to/Patient 2.xls"   # or .xlsx/.csv
#   sig, meta = load_tdms(tdms_path, channel_hint=None)  # supply channel_hint if multiple channels
#   ann = read_annotations(ann_path)
#   ann_idx = align_annotations_to_samples(ann, meta)
#   plot_overview(meta, ann_idx)
#   plot_segment_with_annotations(sig, meta, ann_idx, t_start_sec=0, t_window_sec=60)
#   noise = compute_noise_report(sig, meta)
#   tod = seizure_time_of_day_stats(ann_idx)
#   trends = compute_signal_trends(sig, meta)
#
# Notes:
# - This module attempts to auto-detect common annotation column names in Danish/English.
# - For TDMS, it will auto-pick the first channel if channel_hint is not provided.
# - All plots use matplotlib with default styling (no specific colors set).

from __future__ import annotations
from typing import Optional, Tuple, Iterable, Dict, Any
import re
from datetime import datetime, date, time, timezone
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dateutil import tz
from datetime import datetime, timedelta
import warnings
import os

# SciPy is optional but strongly recommended for spectral analysis.
try:
    from scipy.signal import welch, iirnotch, filtfilt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# nptdms is required to read TDMS
try:
    from nptdms import TdmsFile
    NPTDMS_AVAILABLE = True
except Exception:
    NPTDMS_AVAILABLE = False


@dataclass
class RecordingMeta:
    fs: float                 # Sampling frequency [Hz]
    start_time: datetime      # Recording absolute start timestamp (timezone-aware if possible)
    n_samples: int            # Number of samples
    channel_name: str         # Channel label / name
    units: Optional[str] = None  # Engineering units if available (e.g., mV)
    path: Optional[str] = None   # Source file path

# ---------------------------
# Helpers: time parsing & correction
# ---------------------------

_TZ_EU_CPH = "Europe/Copenhagen"

def _to_datetime_safe(value: Any) -> Optional[datetime]:
    """
    Try to coerce common TDMS/NI property values into a Python datetime.
    Returns timezone-aware or naive depending on the input.
    If the value cannot be parsed, returns None.
    """
    if value is None:
        return None

    # Already datetime?
    if isinstance(value, datetime):
        return value

    # Pandas/Timestamp?
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    # Numeric epoch? (seconds or milliseconds)
    if isinstance(value, (int, float)):
        # Heuristic: treat very large numbers as ms epoch
        try:
            if value > 1e12:
                # ms
                return datetime.utcfromtimestamp(value / 1000.0)
            elif value > 1e9:
                # seconds in epoch ~ 2001+; still fine
                return datetime.utcfromtimestamp(value)
        except Exception:
            pass

    # String parsing
    try:
        # pandas is robust to many NI/TDMS formats
        ts = pd.to_datetime(str(value), utc=False, errors="raise")
        # to_pydatetime preserves timezone info if present
        return ts.to_pydatetime()
    except Exception:
        return None


def _to_local_naive(dt: datetime, prefer_tz: str = "Europe/Copenhagen", assume_source_tz: Optional[str] = "UTC") -> datetime:
    """
    Convert an input datetime to *local clock time* in `prefer_tz`,
    then drop timezone info so the result is NAIVE.
    Rules:
      - If `dt` already has tzinfo -> convert to prefer_tz, then return dt.replace(tzinfo=None).
      - If `dt` is naive:
          * If `assume_source_tz` is not None, interpret `dt` as that tz,
            then convert to prefer_tz, and drop tzinfo.
          * If `assume_source_tz` is None, treat `dt` as already local clock time and just return it.
    """
    from dateutil import tz as _tz

    if dt.tzinfo is not None:
        # tz-aware -> to local -> make naive
        local = dt.astimezone(_tz.gettz(prefer_tz))
        return local.replace(tzinfo=None)

    # naive input
    if assume_source_tz:
        src = _tz.gettz(assume_source_tz)
        as_src = dt.replace(tzinfo=src)
        local = as_src.astimezone(_tz.gettz(prefer_tz))
        return local.replace(tzinfo=None)
    else:
        # Already intended as local clock time; leave as-is but ensure it's a true datetime
        return dt


def _first_present(d: Mapping[str, Any], keys: list[str]) -> Any:
    """Return the first found value for any of `keys` in mapping `d`, else None."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def _extract_tdms_start_time(props_chain: list[Mapping[str, Any]]) -> Optional[datetime]:
    """
    Look through a chain of property dicts (channel -> group -> file) and
    try to find a plausible start time field.
    """
    candidate_keys = [
        # very common NI / nptdms keys
        "wf_start_time", "NI_wfStartTime",
        # other plausible labels
        "start_time", "Start Time", "Timestamp",
        "NI_ExpStartTimeStamp", "NI_ExpTimeStamp",  # sometimes present
        "Date", "Time"  # rarely useful alone, but pandas can sometimes parse combined strings upstream
    ]
    for props in props_chain:
        raw = _first_present(props, candidate_keys)
        if raw is None:
            continue
        dt = _to_datetime_safe(raw)
        if dt is not None:
            return dt
    return None


# ---------------------------
# Public: refined loader using helpers
# ---------------------------
def load_tdms(path: str, channel_hint: Optional[str] = None,
              prefer_tz: str = "Europe/Copenhagen",
              assume_source_tz: Optional[str] = "UTC",
              prefer_naive_local: bool = True) -> Tuple[np.ndarray, RecordingMeta]:
    """
    Load a TDMS file and return (signal, metadata).

    - Sampling frequency (fs) is inferred from common NI/nptdms properties.
    - Start time is read from channel/group/file properties.
    - If `prefer_naive_local` is True (default), `start_time` is returned as a NAIVE datetime
      in the `prefer_tz` local clock (i.e., '11:05:02', not '09:05:02+02:00').
      For naive inputs we assume the source timestamp is in `assume_source_tz` (default 'UTC').

    Args:
        path: Path to the TDMS file.
        channel_hint: Optional substring to help pick the intended channel.
        prefer_tz: IANA name of the intended local time zone (default Europe/Copenhagen).
        assume_source_tz: If input start time is naive, interpret it as being in this tz (default 'UTC').
        prefer_naive_local: If True, convert to `prefer_tz` and drop tzinfo (return naive).

    Returns:
        (signal: np.ndarray[float], meta: RecordingMeta)
    """
    if not NPTDMS_AVAILABLE:
        raise ImportError("nptdms is not installed. Please `pip install nptdms`.")

    tf = TdmsFile.read(path)

    # --- Choose group and channel ---
    groups = tf.groups()
    if not groups:
        raise ValueError("No groups found in TDMS.")
    grp = groups[0]

    channels = grp.channels()
    if not channels:
        # fall back to first channel across all groups
        channels = [ch for g in tf.groups() for ch in g.channels()]
        if not channels:
            raise ValueError("No channels found in TDMS.")

    ch = None
    if channel_hint:
        for c in channels:
            if channel_hint.lower() in c.name.lower():
                ch = c
                break
    if ch is None:
        ch = channels[0]

    # --- Read signal ---
    data = np.asarray(ch[:], dtype=float)
    n_samples = int(data.shape[0])

    # --- Properties (channel -> group -> file) ---
    ch_props   = getattr(ch, "properties", {}) or {}
    grp_props  = getattr(grp, "properties", {}) or {}
    file_props = getattr(tf, "properties", {}) or {}
    props_chain = [ch_props, grp_props, file_props]

    # --- Units ---
    units = (
        _first_present(ch_props, ["unit_string", "NI_UnitDescription", "unit"])
        or None
    )

    # --- Sampling frequency ---
    fs = None
    wf_increment = _first_present(ch_props, ["wf_increment", "NI_wfIncrement"])
    if wf_increment is not None:
        try:
            fs = 1.0 / float(wf_increment)
        except Exception:
            fs = None

    if fs is None:
        for key in ["fs", "sampling_frequency", "Sample Rate", "NI_SampleRate"]:
            v = ch_props.get(key)
            if v is None:
                v = grp_props.get(key)
            if v is None:
                v = file_props.get(key)
            if v is not None:
                try:
                    fs = float(v)
                    break
                except Exception:
                    pass

    if fs is None:
        fs = 512.0  # sensible default for your dataset
        warnings.warn("Could not read sampling frequency from TDMS; defaulting to fs=512 Hz.")

    # --- Start time (correct to NAIVE local if requested) ---
    start_time_raw = _extract_tdms_start_time(props_chain)

    if start_time_raw is None:
        warnings.warn("Could not read start time from TDMS; defaulting to current local time.")
        # Use current time in prefer_tz as naive
        now_local = pd.Timestamp.now(tz=prefer_tz).to_pydatetime()
        start_time = now_local.replace(tzinfo=None)
    else:
        if prefer_naive_local:
            # Convert to prefer_tz, then drop tzinfo -> NAIVE local clock time
            start_time = _to_local_naive(
                start_time_raw,
                prefer_tz=prefer_tz,
                assume_source_tz=assume_source_tz
            )
        else:
            # Keep timezone if any; otherwise keep naive
            start_time = start_time_raw

    meta = RecordingMeta(
        fs=float(fs),
        start_time=start_time,          # <— NAIVE local datetime if prefer_naive_local=True
        n_samples=n_samples,
        channel_name=str(ch.name),
        units=units,
        path=path,
    )
    return data, meta




def _read_excel_any(path: str) -> pd.DataFrame:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    # Prefer engine based on extension
    engine = None
    if suffix == ".xls":
        engine = "xlrd"
    elif suffix == ".xlsx":
        engine = "openpyxl"
    try:
        return pd.read_excel(path, engine=engine)
    except Exception:
        # Last resort, try without engine
        return pd.read_excel(path)


def read_annotations(path: str) -> pd.DataFrame:
    """
    Reads an annotations file (Excel or CSV) and tries to normalize to columns:
      ['onset_time', 'offset_time', 'label'] as datetimes for onset/offset.
    Accepts absolute timestamps or relative-in-recording seconds.
    Also supports columns named in Danish/English variants.
    """
    df = _read_excel_any(path)

    # Standardize column names (lower + strip)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Candidate mappings
    start_keys = ["start", "start_tid", "onset", "onset_time", "t_start", "begin", "begynd", "anfald_start", "starttime", "start time"]
    end_keys   = ["end", "slut", "offset", "offset_time", "t_end", "finish", "anfald_slut", "endtime", "end time"]
    label_keys = ["label", "type", "event", "annotation", "klasse", "kategori", "note", "kommentar"]

    # Helper to find a present column
    def pick(keys):
        for k in keys:
            if k in df.columns:
                return k
        return None

    start_col = pick(start_keys)
    end_col   = pick(end_keys)
    label_col = pick(label_keys)

    # If no label column, create a default
    if label_col is None:
        df["label"] = "seizure"
        label_col = "label"

    # If only one timestamp column is provided (e.g., onset only), create short fixed duration (e.g., 30s)
    default_duration_sec = 30.0

    # Parse times: they might be absolute timestamps or seconds-from-start
    def parse_time_series(series: pd.Series):
        # Try numeric seconds
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().mean() > 0.8:
            return numeric, "relative_seconds"
        # Else try datetimes
        dt = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
        return dt, "absolute_datetime"

    if start_col is None:
        raise ValueError("Could not locate a 'start/onset' column in annotations file. Columns found: %s" % list(df.columns))

    start_vals, start_kind = parse_time_series(df[start_col])
    if end_col is not None:
        end_vals, end_kind = parse_time_series(df[end_col])
    else:
        end_vals, end_kind = None, None

    # Build normalized DataFrame
    norm = pd.DataFrame({
        "label": df[label_col] if label_col in df.columns else "seizure",
    })
    if start_kind == "relative_seconds":
        norm["onset_rel_sec"] = start_vals.astype(float)
    else:
        norm["onset_time"] = pd.to_datetime(start_vals, utc=False, dayfirst=True)

    if end_vals is not None:
        if end_kind == "relative_seconds":
            norm["offset_rel_sec"] = end_vals.astype(float)
        else:
            norm["offset_time"] = pd.to_datetime(end_vals, utc=False, dayfirst=True)
    else:
        # synthesize offset from default duration
        if start_kind == "relative_seconds":
            norm["offset_rel_sec"] = norm["onset_rel_sec"] + default_duration_sec
        else:
            norm["offset_time"] = norm["onset_time"] + pd.to_timedelta(default_duration_sec, unit="s")

    # Keep any extra useful columns (e.g., confidence, notes)
    for extra in df.columns:
        if extra not in [start_col, end_col, label_col]:
            norm[extra] = df[extra]

    return norm


def align_annotations_to_samples(ann: pd.DataFrame, meta: RecordingMeta) -> pd.DataFrame:
    """
    Given normalized annotations and recording meta, compute sample indices and absolute times
    for each event. Returns DataFrame with columns:
      onset_time, offset_time, onset_idx, offset_idx, label
    """
    from dateutil import tz as _tz
    tz_cph = _tz.gettz("Europe/Copenhagen")

    out = ann.copy()

    # Compute onset/offset absolute time
    if "onset_time" not in out.columns and "onset_rel_sec" in out.columns:
        out["onset_time"] = pd.to_datetime(meta.start_time) + pd.to_timedelta(out["onset_rel_sec"], unit="s")
    if "offset_time" not in out.columns and "offset_rel_sec" in out.columns:
        out["offset_time"] = pd.to_datetime(meta.start_time) + pd.to_timedelta(out["offset_rel_sec"], unit="s")

    # Localize times if naive
    for col in ["onset_time", "offset_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
            # If naive, localize; if already tz-aware, leave as is
            if getattr(out[col].dt.tz, 'zone', None) is None:
                out[col] = out[col].dt.tz_localize(tz_cph, ambiguous="NaT", nonexistent="NaT")

    # Clamp to recording duration
    duration_sec = meta.n_samples / meta.fs
    rec_end = meta.start_time + timedelta(seconds=duration_sec)

    out["onset_time"] = out["onset_time"].clip(lower=meta.start_time, upper=rec_end)
    out["offset_time"] = out["offset_time"].clip(lower=meta.start_time, upper=rec_end)

    # Compute sample indices
    out["onset_idx"] = ((out["onset_time"] - meta.start_time).dt.total_seconds() * meta.fs).round().astype(int)
    out["offset_idx"] = ((out["offset_time"] - meta.start_time).dt.total_seconds() * meta.fs).round().astype(int)

    # Clean anomalies
    out["offset_idx"] = np.maximum(out["offset_idx"], out["onset_idx"] + 1)
    out["onset_idx"] = np.clip(out["onset_idx"], 0, meta.n_samples - 1)
    out["offset_idx"] = np.clip(out["offset_idx"], 1, meta.n_samples)

    # Keep key columns
    keep = ["label", "onset_time", "offset_time", "onset_idx", "offset_idx"]
    extra_cols = [c for c in out.columns if c not in keep]
    return out[keep + extra_cols]


def _downsample_for_plot(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x
    n = (x.shape[0] // factor) * factor
    x = x[:n].reshape(-1, factor).mean(axis=1)
    return x


def plot_overview(meta: RecordingMeta, ann_idx: pd.DataFrame):
    """
    Plot an overview timeline showing where seizures occur across the full recording.
    """
    total_sec = meta.n_samples / meta.fs
    fig, ax = plt.subplots(figsize=(10, 2))
    # Draw rectangle baseline
    ax.add_patch(Rectangle((0, 0), total_sec, 1, fill=False))
    # Add seizure spans
    for _, row in ann_idx.iterrows():
        s = row["onset_idx"] / meta.fs
        e = row["offset_idx"] / meta.fs
        ax.add_patch(Rectangle((s, 0), e - s, 1, alpha=0.3))
    ax.set_xlim(0, total_sec)
    ax.set_yticks([])
    ax.set_xlabel("Tid (sekunder fra start)")
    ax.set_title("Oversigt: markerede anfald")
    plt.show()


def plot_segment_with_annotations(sig: np.ndarray, meta: RecordingMeta, ann_idx: pd.DataFrame,
                                  t_start_sec: float, t_window_sec: float = 60.0,
                                  max_points: int = 20000):
    """
    Plot a signal segment starting at t_start_sec with given window length, overlaying seizures.
    """
    i0 = int(max(0, t_start_sec * meta.fs))
    i1 = int(min(meta.n_samples, i0 + t_window_sec * meta.fs))
    segment = sig[i0:i1]

    # Downsample for plotting if needed
    factor = max(1, int(np.ceil(len(segment) / max_points)))
    y = _downsample_for_plot(segment, factor)
    x = np.linspace(i0 / meta.fs, i1 / meta.fs, num=len(y), endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y)
    # Overlay seizures as shaded regions
    for _, row in ann_idx.iterrows():
        s = row["onset_idx"] / meta.fs
        e = row["offset_idx"] / meta.fs
        # If overlap with [x[0], x[-1]]
        if e >= x[0] and s <= x[-1]:
            ax.axvspan(max(s, x[0]), min(e, x[-1]), alpha=0.2)
    ax.set_xlabel("Tid (s)")
    ax.set_ylabel(meta.units or "Amplitude")
    ax.set_title(f"Signaludsnit {t_start_sec:.1f}s–{t_start_sec + t_window_sec:.1f}s ({meta.channel_name})")
    plt.show()


def compute_noise_report(sig: np.ndarray, meta: RecordingMeta, line_freq: float = 50.0) -> Dict[str, float]:
    """
    Basic noise metrics:
      - RMS over full record
      - Line noise bandpower around 50 Hz (+/- 2 Hz) relative to 0.5–40 Hz band
      - High-frequency noise (40–100 Hz) band fraction
    Returns a dict of metrics. If SciPy is unavailable, returns NaNs where needed.
    """
    out = {}
    sig = np.asarray(sig).astype(float)
    out["rms"] = float(np.sqrt(np.mean(np.square(sig))))

    if not SCIPY_AVAILABLE:
        out.update({
            "line_noise_ratio": np.nan,
            "hf_noise_ratio": np.nan,
        })
        return out

    # Welch PSD
    nperseg = min(int(4 * meta.fs), len(sig))
    if nperseg < 16:
        nperseg = len(sig)
    f, Pxx = welch(sig, fs=meta.fs, nperseg=nperseg)
    # Helper to integrate band
    def bandpower(f1, f2):
        m = (f >= f1) & (f <= f2)
        if not np.any(m):
            return 0.0
        return float(np.trapz(Pxx[m], f[m]))
    # Bands
    bp_base = bandpower(0.5, 40.0)
    bp_line = bandpower(line_freq - 2.0, line_freq + 2.0)
    bp_hf   = bandpower(40.0, 100.0)

    out["line_noise_ratio"] = float(bp_line / (bp_base + 1e-12))
    out["hf_noise_ratio"]   = float(bp_hf / (bp_base + 1e-12))

    # Optional notch filtered RMS improvement
    try:
        b_notch, a_notch = iirnotch(w0=line_freq/(meta.fs/2), Q=30.0)
        y = filtfilt(b_notch, a_notch, sig)
        out["rms_after_notch"] = float(np.sqrt(np.mean(np.square(y))))
    except Exception:
        out["rms_after_notch"] = np.nan

    return out


def seizure_time_of_day_stats(ann_idx: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize when seizures occur by time-of-day and day/night categorization.
    Returns a DataFrame with per-event info and prints a small summary.
    Night is defined as 22:00–06:00 local time by default.
    """
    from dateutil import tz as _tz
    tz_cph = _tz.gettz("Europe/Copenhagen")
    df = ann_idx.copy()
    df["onset_time"] = pd.to_datetime(df["onset_time"]).dt.tz_convert(tz_cph)

    df["hour"] = df["onset_time"].dt.hour + df["onset_time"].dt.minute / 60.0
    # Night definition
    night_hours = ((df["hour"] >= 22.0) | (df["hour"] < 6.0))
    df["period"] = np.where(night_hours, "Nat", "Dag")

    # Basic counts
    counts = df["period"].value_counts(dropna=False)
    print("Antal anfald (Dag/Nat):")
    for k in ["Dag", "Nat"]:
        print(f"  {k}: {int(counts.get(k, 0))}")

    # Simple histogram plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df["hour"], bins=24, range=(0,24))
    ax.set_xlabel("Klokkeslæt (timer)")
    ax.set_ylabel("Antal anfald")
    ax.set_title("Tidspunkt for anfald på døgnet")
    plt.show()

    return df[["onset_time", "offset_time", "period", "hour", "label"]]


def compute_signal_trends(sig: np.ndarray, meta: RecordingMeta,
                          window_sec: float = 5.0, show_plots: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute simple trends: rolling RMS, rolling mean (baseline), and a rough z-score artifact flag.
    Optionally shows plots.
    """
    sig = np.asarray(sig).astype(float)
    N = len(sig)
    win = int(max(1, round(window_sec * meta.fs)))
    # Rolling RMS via convolution
    rms = np.sqrt(np.convolve(sig*sig, np.ones(win), mode="same") / win)
    baseline = np.convolve(sig, np.ones(win)/win, mode="same")

    # Z-score w.r.t rolling mean and std
    # For std, approximate via rolling RMS relation: std^2 = rms^2 - mean^2
    std_approx = np.sqrt(np.maximum(rms**2 - baseline**2, 1e-12))
    z = (sig - baseline) / np.maximum(std_approx, 1e-9)

    if show_plots:
        t = np.arange(N) / meta.fs
        # Plot RMS
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, rms)
        ax.set_xlabel("Tid (s)")
        ax.set_ylabel("RMS")
        ax.set_title(f"Rullende RMS (~{window_sec:.1f}s vindue)")
        plt.show()

        # Plot baseline
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, baseline)
        ax.set_xlabel("Tid (s)")
        ax.set_ylabel("Baseline")
        ax.set_title("Rullende middelværdi (baseline)")
        plt.show()

    return {"rms": rms, "baseline": baseline, "z": z}
# --- End of module ---
