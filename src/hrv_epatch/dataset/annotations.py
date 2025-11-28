from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import os
import re
import datetime

import numpy as np
import pandas as pd

from .naming import RecordingKey


# ---------- Hjælper: robust parsing af tidsceller ----------

def _parse_time_cell(date_cell, time_cell) -> pd.Timestamp:
    """
    Kombinér en datocelle og en tidscelle til en pandas.Timestamp.
    Returnerer pd.NaT hvis enten er missing eller ikke kan parses.
    """
    if pd.isna(time_cell) or pd.isna(date_cell):
        return pd.NaT

    # --- PARS DATO KONSEKVENT SOM dd.mm.yy ---
    if isinstance(date_cell, (pd.Timestamp, datetime.datetime, datetime.date)):
        date_ts = pd.to_datetime(date_cell, errors="coerce")
    else:
        s = str(date_cell).strip()
        date_ts = pd.to_datetime(s, format="%d.%m.%y", errors="coerce")

    if pd.isna(date_ts):
        return pd.NaT

    date_norm = date_ts.normalize()

    # resten af funktionen som før:
    if isinstance(time_cell, (int, float, np.integer, np.floating)):
        return date_norm + pd.to_timedelta(float(time_cell), unit="D")

    if isinstance(time_cell, pd.Timestamp):
        t = time_cell
        return date_norm + pd.to_timedelta(t.hour, unit="h") \
                         + pd.to_timedelta(t.minute, unit="m") \
                         + pd.to_timedelta(t.second, unit="s") \
                         + pd.to_timedelta(t.microsecond, unit="us")

    if isinstance(time_cell, datetime.datetime):
        t = time_cell
        return date_norm + pd.to_timedelta(t.hour, unit="h") \
                         + pd.to_timedelta(t.minute, unit="m") \
                         + pd.to_timedelta(t.second, unit="s") \
                         + pd.to_timedelta(t.microsecond, unit="us")

    if isinstance(time_cell, datetime.time):
        return datetime.datetime.combine(date_norm.date(), time_cell)

    s = str(time_cell).strip()
    if s == "" or s.lower() in {"nan", "none", "na"}:
        return pd.NaT
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return date_norm + pd.to_timedelta(parsed.hour, unit="h") \
                     + pd.to_timedelta(parsed.minute, unit="m") \
                     + pd.to_timedelta(parsed.second, unit="s") \
                     + pd.to_timedelta(parsed.microsecond, unit="us")



def _seconds_since_midnight(ts: pd.Timestamp) -> float:
    if pd.isna(ts):
        return np.nan
    return ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6


# ---------- Indlæsning af EN annoteringsfil (Patient X.xls) ----------

def load_annotations(path: Path) -> pd.DataFrame:
    """
    Load én annotationsfil (Patient X[a/b].xls) med epilepsi-anfald.
    Antager samme struktur som dine eksisterende filer:
      - tabelheader på række 7 => header=6
      - kolonner som 'Dato', 'Anfaldsstart Klinisk (tt:mm:ss)', osv.

    Returnerer et DataFrame med mindst disse kolonner:
      - seizure_number
      - date (normaliseret dato, midnat)
      - start_clinic, start_eeg, end_clinic, end_eeg  (alle som Timestamp/pd.NaT)
      - seizure_type, other (hvis tilgængelig)
      - start_*_seconds / _hour osv. til evt. statistik
      - source_file (filnavn)
    """
    if path is None:
        return pd.DataFrame()

    # Læs med header på række 7 (index 6), som i din gamle kode
    df = pd.read_excel(path, header=6)

    # Normalisér kolonnenavne (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(key: str) -> Optional[str]:
        key_l = key.lower()
        for c in df.columns:
            if key_l in c.lower():
                return c
        return None

    num_col    = find_col("Anfald nr") or find_col("seizure")
    date_col   = find_col("Dato")
    s_clin_col = find_col("Anfaldsstart Klinisk") or find_col("Anfaldsstart klinisk")
    s_eeg_col  = find_col("Anfaldsstart EEG") or find_col("Anfaldsstart eeg")
    e_clin_col = find_col("Anfaldstop Klinisk") or find_col("Anfaldstop klinisk")
    e_eeg_col  = find_col("Anfaldstop EEG") or find_col("Anfaldstop eeg")
    type_col   = find_col("Anfaldstype") or find_col("anfaldstype")
    other_col  = find_col("Evt. bemærkninger") or find_col("note") or find_col("other")

    res = pd.DataFrame()

    # Seizure-nummer
    res["seizure_number"] = df[num_col] if num_col else (df.index + 1)

    # Dato normaliseret til midnat
    if date_col:
        res["date"] = pd.to_datetime(
            df[date_col].astype(str).str.strip(),
            format="%d.%m.%y",
            errors="coerce"
        ).dt.normalize()

    else:
        res["date"] = pd.NaT

    # Kombinér dato + tidskolonner til fulde timestamps
    res["start_clinic"] = df.apply(
        lambda r: _parse_time_cell(r[date_col], r[s_clin_col]) if s_clin_col else pd.NaT,
        axis=1,
    )
    res["start_eeg"] = df.apply(
        lambda r: _parse_time_cell(r[date_col], r[s_eeg_col]) if s_eeg_col else pd.NaT,
        axis=1,
    )
    res["end_clinic"] = df.apply(
        lambda r: _parse_time_cell(r[date_col], r[e_clin_col]) if e_clin_col else pd.NaT,
        axis=1,
    )
    res["end_eeg"] = df.apply(
        lambda r: _parse_time_cell(r[date_col], r[e_eeg_col]) if e_eeg_col else pd.NaT,
        axis=1,
    )

    # Type/other hvis tilgængelig
    res["seizure_type"] = df[type_col] if type_col else None
    res["other"] = df[other_col] if other_col else None

    # Sørg for at tidskolonner er rigtige datotider
    time_cols = ["start_clinic", "start_eeg", "end_clinic", "end_eeg"]
    for c in time_cols:
        res[c] = pd.to_datetime(res[c], errors="coerce")

    # Hjælpekolonner til evt. statistikker
    for prefix in time_cols:
        res[f"{prefix}_seconds"] = res[prefix].apply(_seconds_since_midnight)
        res[f"{prefix}_hour"] = res[prefix].dt.hour

    # Traceability
    res["source_file"] = path.name

    return res



# ---------- Find korrekt annoteringsfil for en given recording ----------

def find_annotation_file(
    key: RecordingKey,
    annotations_root: Path,
) -> Optional[Path]:
    """
    Find den rigtige .xls/.xlsx til (patient, enrollment).

    Eksempler:
      Patient 38a -> 'patient 38a.xls'
      Patient 5   -> 'patient 5.xls'
    """
    candidates = []
    if key.enrollment_id is not None:
        candidates.append(annotations_root / f"patient {key.patient_id}{key.enrollment_id}.xls")
        candidates.append(annotations_root / f"Patient {key.patient_id}{key.enrollment_id}.xls")
    candidates.append(annotations_root / f"patient {key.patient_id}.xls")
    candidates.append(annotations_root / f"Patient {key.patient_id}.xls")

    for c in candidates:
        if c.exists():
            return c
    return None