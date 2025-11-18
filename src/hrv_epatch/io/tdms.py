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
# Dataklasser
# ============================================================

@dataclass
class TdmsMeta:
    fs: float
    start_time: Optional[datetime]
    n_samples: int
    channel_name: str
    units: Optional[str] = None
    path: Optional[str] = None


# (Hvis du andre steder bruger RecordingMeta-navnet, kan du gøre:)
RecordingMeta = TdmsMeta


# ============================================================
# Helper-funktioner til tid
# ============================================================

def _to_datetime_safe(value: Any) -> Optional[datetime]:
    """
    Forsøg at parse TDMS/NI property til en Python datetime.
    Beholder evt. timezone-info hvis den er der.
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
    Konvertér en datetime til *lokal klokkeslæt* i `prefer_tz`, og drop tzinfo.
    Regler:
      - Hvis dt allerede er tz-aware: konvertér til prefer_tz -> returnér naive.
      - Hvis dt er naive:
          * Hvis assume_source_tz != None: tolkes som den tz, konvertér til prefer_tz -> naive.
          * Hvis assume_source_tz == None: antages at være lokal klokkeslæt allerede -> returnér dt.
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
    Søg efter en plausibel starttid i channel->group->file properties.
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
# KANONISK LOADER: fra TDMS-path
# ============================================================

def load_tdms_from_path(
    path: str,
    channel_hint: Optional[str] = None,
    prefer_tz: str = "Europe/Copenhagen",
    assume_source_tz: Optional[str] = "UTC",
    prefer_naive_local: bool = True,
) -> Tuple[np.ndarray, TdmsMeta]:
    """
    Kanonisk TDMS-loader.
    - Finder kanal (evt. baseret på channel_hint)
    - Udregner fs (sampling frequency)
    - Læser starttid fra TDMS-properties
    - Korrigerer tid til NAIV lokal tid i `prefer_tz` (default Europe/Copenhagen)

    Args
    ----
    path : str
        Fuld sti til TDMS-filen.
    channel_hint : Optional[str]
        Delstreng der skal matche kanalnavnet (fx 'EKG').
        Hvis None -> første kanal bruges.
    prefer_tz : str
        Lokal tidszone (IANA navn). Default "Europe/Copenhagen".
    assume_source_tz : Optional[str]
        Hvis starttid i TDMS er naive: antag at den er i denne tz (fx 'UTC').
        Hvis None: antag at den er lokal allerede.
    prefer_naive_local : bool
        Hvis True: returnér start_time som naive lokal (uden tzinfo).
        Hvis False: kan i fremtiden bruges til at returnere tz-aware, men pt. returneres stadig naive.

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
# Loader: find TDMS-fil ud fra patient-ID
# ============================================================

def _find_base_dirs(base_dir: str) -> Tuple[str, Optional[str]]:
    """
    Find under-mapper for:
      - data_dir: 'Patients ePatch data' (eller lign. navn)
      - seizure_log_dir: 'Seizure log...' (hvis den findes)

    Returnerer (data_dir, seizure_log_dir_or_None)
    """
    entries = os.listdir(base_dir)
    data_dir = None
    seizure_dir = None

    for name in entries:
        lower = name.lower()
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if "patient" in lower and "epatch" in lower and "data" in lower:
            data_dir = full
        if "seizure" in lower and "log" in lower:
            seizure_dir = full

    if data_dir is None:
        raise RuntimeError(f"Could not locate 'Patients ePatch data' folder under {base_dir}")

    return data_dir, seizure_dir


def _find_patient_tdms_path(
    patient_id: str,
    base_dir: str,
) -> str:
    """
    Find TDMS-fil for en patient ud fra mappestrukturen:
      base_dir/
        ... 'Patients ePatch data'/Patient {id}/[enrollment|recording].../*.tdms
    """
    data_dir, _ = _find_base_dirs(base_dir)
    patient_path = os.path.join(data_dir, patient_id)
    if not os.path.isdir(patient_path):
        raise FileNotFoundError(f"Patient folder not found: {patient_path}")

    subfolders = [
        f
        for f in os.listdir(patient_path)
        if any(k in f.lower() for k in ["enrollment", "recording"])
    ]
    if not subfolders:
        raise FileNotFoundError(f"No enrollment/recording folder found under {patient_path}")

    # Vælg første (eller sortér, hvis du vil være deterministisk)
    subfolders.sort()
    session_path = os.path.join(patient_path, subfolders[0])

    files = os.listdir(session_path)
    tdms_file = next((os.path.join(session_path, f) for f in files if f.lower().endswith(".tdms")), None)
    if tdms_file is None:
        raise FileNotFoundError(f"No .tdms file found in session folder {session_path}")

    return tdms_file


def load_tdms_for_patient(
    patient_id: str,
    base_dir: str,
    channel_hint: Optional[str] = "EKG",
    prefer_tz: str = "Europe/Copenhagen",
    assume_source_tz: Optional[str] = "UTC",
    prefer_naive_local: bool = True,
) -> Tuple[np.ndarray, TdmsMeta]:
    """
    Find TDMS-fil for patienten og kald den kanoniske loader.

    Eksempel:
        sig, meta = load_tdms_for_patient(
            "Patient 5",
            base_dir=r"E:\\ML algoritme tl anfaldsdetektion vha HRV\\ePatch data from Aarhus to Lausanne",
        )
    """
    tdms_path = _find_patient_tdms_path(patient_id, base_dir)
    return load_tdms_from_path(
        tdms_path,
        channel_hint=channel_hint,
        prefer_tz=prefer_tz,
        assume_source_tz=assume_source_tz,
        prefer_naive_local=prefer_naive_local,
    )


# ============================================================
# (Valgfri) helper: timestamps som Pandas DataFrame
# ============================================================

def build_ecg_dataframe(signal: np.ndarray, meta: TdmsMeta) -> pd.DataFrame:
    """
    Byg et DataFrame med tidsakse ud fra meta.start_time + fs.
    Hvis start_time er None, laves en simpel index-baseret tidsakse i sekunder.
    """
    N = len(signal)
    if meta.start_time is not None:
        # tidsakse i ns
        step_ns = int(round(1.0 / meta.fs * 1e9))
        start_np = np.datetime64(meta.start_time, "ns")
        timestamps = start_np + np.arange(N) * np.timedelta64(step_ns, "ns")
        return pd.DataFrame({"Timestamp": timestamps, "Value": signal})
    else:
        t = np.arange(N) / meta.fs
        return pd.DataFrame({"TimeSec": t, "Value": signal})
