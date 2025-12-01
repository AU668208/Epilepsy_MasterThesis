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

def _parse_time_only(cell):
    """
    Parse a cell that contains ONLY a time-of-day (hh:mm:ss),
    ignoring any date component that Excel may have added.
    Returns a datetime.time or pd.NaT.
    """
    if pd.isna(cell):
        return pd.NaT

    # Hvis det allerede er en time:
    if isinstance(cell, datetime.time):
        return cell

    # Hvis det er en fuld datetime -> ignorér dato, brug kun time-delen
    if isinstance(cell, datetime.datetime):
        return cell.time()

    # Excel kan gemme tid som fraktion af en dag (float)
    if isinstance(cell, (int, float, np.integer, np.floating)):
        # 0.0 = 00:00, 0.5 = 12:00, 1.0 = 24:00 (næste dag) osv.
        try:
            base = pd.Timestamp("1970-01-01")
            t = base + pd.to_timedelta(float(cell), unit="D")
            return t.time()
        except Exception:
            return pd.NaT

    # Fallback: parse streng som tid
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none", "na"}:
        return pd.NaT

    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed.time()


def _combine_date_and_time(date_ts, time_val):
    """
    Kombinér en pd.Timestamp (dato, normaliseret til midnat) og en datetime.time
    til en fuld Timestamp. Returnerer pd.NaT hvis enten mangler.
    """
    if pd.isna(date_ts) or pd.isna(time_val):
        return pd.NaT

    return pd.Timestamp(
        year=date_ts.year,
        month=date_ts.month,
        day=date_ts.day,
        hour=time_val.hour,
        minute=time_val.minute,
        second=time_val.second,
        microsecond=time_val.microsecond,
    )


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

    res = pd.DataFrame(index=df.index)

    # Seizure-nummer
    if num_col:
        res["seizure_number"] = df[num_col]
    else:
        res["seizure_number"] = df.index + 1

    # Dato normaliseret til midnat (dd.mm.yy), med fallback
    if date_col:
        raw_dates = df[date_col].astype(str).str.strip()
        dates = pd.to_datetime(
            raw_dates,
            format="%d.%m.%y",
            errors="coerce"
        )
        # fallback: mere tolerant parser med dayfirst=True for evt. afvigende formater
        mask = dates.isna()
        if mask.any():
            dates[mask] = pd.to_datetime(
                raw_dates[mask],
                dayfirst=True,
                errors="coerce",
            )
        res["date"] = dates.dt.normalize()
    else:
        res["date"] = pd.NaT

    # Hjælpefunktion til at bygge fulde timestamps ud fra dato + tidskolonne
    def _build_ts_column(time_col_name: Optional[str]) -> pd.Series:
        if not time_col_name:
            return pd.Series(pd.NaT, index=df.index)
        times_only = df[time_col_name].apply(_parse_time_only)
        return pd.Series(
            [
                _combine_date_and_time(d, t)
                for d, t in zip(res["date"], times_only)
            ],
            index=df.index,
        )

    # Kombinér dato + tidskolonner til fulde timestamps
    res["start_clinic"] = _build_ts_column(s_clin_col)
    res["start_eeg"]    = _build_ts_column(s_eeg_col)
    res["end_clinic"]   = _build_ts_column(e_clin_col)
    res["end_eeg"]      = _build_ts_column(e_eeg_col)

    # Type/other hvis tilgængelig
    res["seizure_type"] = df[type_col] if type_col else None
    res["other"]        = df[other_col] if other_col else None

    # Sørg for at tidskolonner er rigtige datotider (Timestamp)
    time_cols = ["start_clinic", "start_eeg", "end_clinic", "end_eeg"]
    for c in time_cols:
        res[c] = pd.to_datetime(res[c], errors="coerce")

    # Hjælpekolonner til evt. statistikker
    for prefix in time_cols:
        res[f"{prefix}_seconds"] = res[prefix].apply(_seconds_since_midnight)
        res[f"{prefix}_hour"]    = res[prefix].dt.hour

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
      Patient 38a -> 'patient 38a.xls' eller 'Patient 38a.xlsx'
      Patient 5   -> 'patient 5.xls'   eller 'Patient 5.xlsx'
    """
    candidates = []

    exts = [".xls", ".xlsx"]

    if key.enrollment_id is not None:
        enr = key.enrollment_id
        for ext in exts:
            candidates.append(annotations_root / f"patient {key.patient_id}{enr}{ext}")
            candidates.append(annotations_root / f"Patient {key.patient_id}{enr}{ext}")

    for ext in exts:
        candidates.append(annotations_root / f"patient {key.patient_id}{ext}")
        candidates.append(annotations_root / f"Patient {key.patient_id}{ext}")

    for c in candidates:
        if c.exists():
            return c

    # ekstra fallback: glob i tilfælde af mærkelige mellemrum osv.
    if key.enrollment_id is not None:
        pattern = f"[Pp]atient {key.patient_id}{key.enrollment_id}.*"
    else:
        pattern = f"[Pp]atient {key.patient_id}.*"

    glob_candidates = sorted(annotations_root.glob(pattern))
    if glob_candidates:
        return glob_candidates[0]

    return None

