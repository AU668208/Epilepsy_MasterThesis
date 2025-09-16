
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


def _to_datetime_safe(x) -> Optional[datetime]:
    if pd.isna(x):
        return None
    # Try pandas to_datetime first
    try:
        dt = pd.to_datetime(x, errors="raise", dayfirst=True, utc=False)
        # If it's timezone-aware already, keep it. Otherwise localize to Europe/Copenhagen
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            return dt.tz_localize("Europe/Copenhagen", ambiguous="NaT", nonexistent="NaT")
        return dt
    except Exception:
        return None


def load_tdms(path: str, channel_hint: Optional[str] = None) -> Tuple[np.ndarray, RecordingMeta]:
    """
    Load a TDMS file and return (signal, metadata).
    Attempts to extract fs and start_time from TDMS properties commonly used by NI.
    """
    if not NPTDMS_AVAILABLE:
        raise ImportError("nptdms is not installed. Please `pip install nptdms`.")
    tf = TdmsFile.read(path)

    # Choose a group and channel
    groups = tf.groups()
    if len(groups) == 0:
        raise ValueError("No groups found in TDMS.")
    grp = groups[0]
    # Find channel by hint or the first available
    channels = grp.channels()
    if len(channels) == 0:
        # Try any channel across groups
        channels = [ch for g in tf.groups() for ch in g.channels()]
        if len(channels) == 0:
            raise ValueError("No channels found in TDMS.")

    ch = None
    if channel_hint is not None:
        for c in channels:
            if channel_hint.lower() in c.name.lower():
                ch = c
                break
    if ch is None:
        ch = channels[0]

    data = ch[:]
    data = np.asarray(data).astype(float)
    n_samples = data.shape[0]

    # Default fs and start_time fallbacks
    fs = None
    start_time = None
    units = None

    # Try common property keys
    props = ch.properties
    grp_props = grp.properties
    file_props = tf.properties

    # Units
    units = props.get("unit_string") or props.get("NI_UnitDescription") or props.get("unit") or None

    # Sampling period
    wf_increment = props.get("wf_increment") or props.get("NI_wfIncrement") or None
    if wf_increment:
        try:
            fs = 1.0 / float(wf_increment)
        except Exception:
            fs = None

    # Sampling freq direct
    if fs is None:
        for key in ["fs", "sampling_frequency", "Sample Rate", "NI_SampleRate"]:
            if key in props:
                try:
                    fs = float(props[key])
                    break
                except Exception:
                    pass

    # Start time
    for source_props in (props, grp_props, file_props):
        if start_time is not None:
            break
        for key in ["wf_start_time", "NI_wfStartTime", "start_time", "Start Time", "Timestamp"]:
            if key in source_props:
                st = source_props[key]
                # nptdms may provide datetime already
                if isinstance(st, datetime):
                    start_time = st
                else:
                    start_time = _to_datetime_safe(st)
                break

    # Fallbacks
    if fs is None:
        fs = 512.0  # default to your stated sample rate
        warnings.warn("Could not read sampling frequency from TDMS; defaulting to fs=512 Hz.")
    if start_time is None:
        # Assume 'now' as a placeholder; user should override if needed.
        start_time = pd.Timestamp.now(tz="Europe/Copenhagen").to_pydatetime()
        warnings.warn("Could not read start time from TDMS; defaulting to now().")

    if getattr(start_time, 'tzinfo', None) is None:
        from dateutil import tz as _tz
        start_time = start_time.replace(tzinfo=_tz.gettz("Europe/Copenhagen"))

    meta = RecordingMeta(
        fs=float(fs),
        start_time=start_time,
        n_samples=int(n_samples),
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
