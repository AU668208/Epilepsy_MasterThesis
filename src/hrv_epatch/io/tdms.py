from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import os
import warnings

import numpy as np
import pandas as pd
from dateutil import tz as _tz
from nptdms import TdmsFile


# ============================================================
# Dataclass for TDMS-metadata
# ============================================================

@dataclass
class TdmsMeta:
    fs: float
    start_time: Optional[datetime]
    n_samples: int
    channel_name: str
    units: Optional[str] = None
    path: Optional[str] = None


# (If using RecordingMeta then can be done:)
RecordingMeta = TdmsMeta


# ============================================================
# Helper-function for time
# ============================================================

def _to_datetime_safe(value: Any) -> Optional[datetime]:
    """
    Attempt to parse a TDMS/NI property to a Python datetime.
    Retains timezone info if present.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    # Numeric epoch? (sekunder / millisekunder)
    if isinstance(value, (int, float)):
        try:
            if value > 1e12:
                return datetime.utcfromtimestamp(value / 1000.0)
            elif value > 1e9:
                return datetime.utcfromtimestamp(value)
        except Exception:
            pass

    # Strings
    try:
        ts = pd.to_datetime(str(value), utc=False, errors="raise")
        return ts.to_pydatetime()
    except Exception:
        return None


def _to_local_naive(
    dt: datetime,
    prefer_tz: str = "Europe/Copenhagen",
    assume_source_tz: Optional[str] = "UTC",
) -> datetime:
    """
    Convert a datetime to *local time* in `prefer_tz`, and drop tzinfo.
    Rules:
      - If dt is already tz-aware: convert to prefer_tz -> return naive.
      - If dt is naive:
          * If assume_source_tz != None: interpret as that tz, convert to prefer_tz -> naive.
          * If assume_source_tz == None: assume it is already local time -> return dt.
    """
    if dt.tzinfo is not None:
        local = dt.astimezone(_tz.gettz(prefer_tz))
        return local.replace(tzinfo=None)

    if assume_source_tz:
        src = _tz.gettz(assume_source_tz)
        as_src = dt.replace(tzinfo=src)
        local = as_src.astimezone(_tz.gettz(prefer_tz))
        return local.replace(tzinfo=None)
    else:
        return dt


def _first_present(d: Dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None


def _extract_tdms_start_time(props_chain: list[Dict[str, Any]]) -> Optional[datetime]:
    """
    Search for a plausible start time in channel->group->file properties.
    """
    candidate_keys = [
        "wf_start_time",
        "NI_wfStartTime",
        "start_time",
        "Start Time",
        "Timestamp",
        "NI_ExpStartTimeStamp",
        "NI_ExpTimeStamp",
    ]
    for props in props_chain:
        raw = _first_present(props, candidate_keys)
        if raw is None:
            continue
        dt = _to_datetime_safe(raw)
        if dt is not None:
            return dt
    return None


# ============================================================
# LOADER: From TDMS-path
# ============================================================

def load_tdms_from_path(
    path: str,
    channel_hint: Optional[str] = None,
    prefer_tz: str = "Europe/Copenhagen",
    assume_source_tz: Optional[str] = "UTC",
    prefer_naive_local: bool = True,
) -> Tuple[np.ndarray, TdmsMeta]:
    """
    TDMS loader.
    - Finds channel (optionally based on channel_hint)
    - Calculates fs (sampling frequency)
    - Reads start time from TDMS properties
    - Adjusts time to NAIVE local time in `prefer_tz` (default Europe/Copenhagen)
    Args
    ----
    path : str
        Full path to the TDMS file.
    channel_hint : Optional[str]
        Substring to match the channel name (e.g., 'EKG').
        If None -> first channel is used.
    prefer_tz : str
        Local timezone (IANA name). Default "Europe/Copenhagen".
    assume_source_tz : Optional[str]
        If start time in TDMS is naive: assume it is in this timezone (e.g., 'UTC').
        If None: assume it is already local.
    prefer_naive_local : bool
        If True: return start_time as naive local (without tzinfo).
        If False: can in the future be used to return tz-aware, but currently still returns naive.

    Returns
    -------
    signal : np.ndarray[float]
    meta   : TdmsMeta
    """
    tf = TdmsFile.read(path)
    groups = tf.groups()
    if not groups:
        raise ValueError(f"No groups found in TDMS file: {path}")

    # Vælg en gruppe og kanal
    channels = [ch for g in groups for ch in g.channels()]
    if not channels:
        raise ValueError(f"No channels found in TDMS file: {path}")

    ch = None
    if channel_hint:
        for c in channels:
            if channel_hint.lower() in c.name.lower():
                ch = c
                break
    if ch is None:
        ch = channels[0]

    data = np.asarray(ch[:], dtype=float)
    n_samples = int(data.shape[0])

    ch_props = getattr(ch, "properties", {}) or {}

    # nptdms: TdmsChannel har .group, ikke .parent
    try:
        grp = getattr(ch, "group", None)
        grp_props = getattr(grp, "properties", {}) or {}
    except Exception:
        grp_props = {}

    file_props = getattr(tf, "properties", {}) or {}
    props_chain = [ch_props, grp_props, file_props]


    units = (
        _first_present(ch_props, ["unit_string", "NI_UnitDescription", "unit"])
        or None
    )

    # ----- Samplingfrekvens -----
    fs = None
    wf_inc = _first_present(ch_props, ["wf_increment", "NI_wfIncrement"])
    if wf_inc is not None:
        try:
            fs = 1.0 / float(wf_inc)
        except Exception:
            fs = None

    if fs is None:
        for key in ("fs", "sampling_frequency", "Sample Rate", "NI_SampleRate"):
            v = ch_props.get(key) or grp_props.get(key) or file_props.get(key)
            if v is not None:
                try:
                    fs = float(v)
                    break
                except Exception:
                    pass

    if fs is None:
        fs = 512.0
        warnings.warn(
            f"Could not read sampling frequency from TDMS ({path}); defaulting to fs=512 Hz."
        )

    # ----- Starttid -----
    start_time = _extract_tdms_start_time(props_chain)
    if start_time is not None and prefer_naive_local:
        start_time = _to_local_naive(
            start_time,
            prefer_tz=prefer_tz,
            assume_source_tz=assume_source_tz,
        )

    meta = TdmsMeta(
        fs=float(fs),
        start_time=start_time,
        n_samples=n_samples,
        channel_name=str(ch.name),
        units=units,
        path=path,
    )
    return data, meta

# ============================================================
# (Optional) helper: timestamps as Pandas DataFrame
# ============================================================

def build_ecg_dataframe(signal: np.ndarray, meta: TdmsMeta) -> pd.DataFrame:
    """
    Build a DataFrame with time axis from meta.start_time + fs.
    If start_time is None, a simple index-based time axis in seconds is created.
    """
    N = len(signal)
    if meta.start_time is not None:
        # time axis in ns
        step_ns = int(round(1.0 / meta.fs * 1e9))
        start_np = np.datetime64(meta.start_time, "ns")
        timestamps = start_np + np.arange(N) * np.timedelta64(step_ns, "ns")
        return pd.DataFrame({"Timestamp": timestamps, "Value": signal})
    else:
        t = np.arange(N) / meta.fs
        return pd.DataFrame({"TimeSec": t, "Value": signal})