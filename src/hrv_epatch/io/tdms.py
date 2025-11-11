# src/hrv_epatch/io/tdms.py
from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from nptdms import TdmsFile

@dataclass
class TdmsMeta:
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

def extract_tdms_channel(path: str, channel_hint: Optional[str] = None) -> Tuple[np.ndarray, TdmsMeta]:
    """
    Load one numeric TDMS channel. Auto-detect fs and start_time from common properties.
    """
    tf = TdmsFile.read(path)
    groups = tf.groups()
    if not groups:
        raise ValueError("No groups found in TDMS.")
    # pick a channel (hinted name like 'ECG' preferred)
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

    data = np.asarray(ch[:], dtype=float)
    n_samples = data.size

    props = ch.properties
    grp_props = ch.parent.properties
    file_props = tf.properties

    units = props.get("unit_string") or props.get("NI_UnitDescription") or props.get("unit") or None

    wf_increment = props.get("wf_increment") or props.get("NI_wfIncrement") or None
    fs = None
    if wf_increment:
        try:
            fs = 1.0 / float(wf_increment)
        except Exception:
            fs = None
    if fs is None:
        for key in ("fs", "sampling_frequency", "Sample Rate", "NI_SampleRate"):
            if key in props:
                try:
                    fs = float(props[key]); break
                except Exception:
                    pass
    if fs is None:
        fs = 512.0
        warnings.warn("Could not read sampling frequency from TDMS; defaulting to fs=512 Hz.")

    start_time = None
    for source in (props, grp_props, file_props):
        if start_time is not None:
            break
        for key in ("wf_start_time", "NI_wfStartTime", "start_time", "Start Time", "Timestamp"):
            if key in source:
                st = source[key]
                if isinstance(st, datetime):
                    start_time = st
                else:
                    start_time = _to_datetime_safe(st)
                break

    if start_time is None:
        start_time = pd.Timestamp.now(tz="Europe/Copenhagen").to_pydatetime()
        warnings.warn("Could not read start time from TDMS; defaulting to now().")

    if getattr(start_time, "tzinfo", None) is None:
        from dateutil import tz as _tz
        start_time = start_time.replace(tzinfo=_tz.gettz("Europe/Copenhagen"))

    meta = TdmsMeta(
        fs=float(fs),
        start_time=start_time,
        n_samples=int(n_samples),
        channel_name=str(ch.name),
        units=units,
        path=path,
    )
    return data, meta
