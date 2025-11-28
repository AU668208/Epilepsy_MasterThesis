# %%
# from src.hrv_epatch.io.tdms import extract_tdms_channel
from src.hrv_epatch.io.data_loader import Load_full_ecg_data

# from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from scipy.signal import welch
from scipy import stats
from datetime import timedelta, datetime as dt

from src.hrv_epatch.io.data_loader import Load_full_ecg_data


@dataclass
class SeizureEvent:
    seizure_id: int
    t0: float       # sekunder fra recording start
    t1: float       # sekunder fra recording start


# %%
data = Load_full_ecg_data("Patient 5")
patient_id = data["PatientID"]
ecg_df = data["ECG"]
seizure_df = data["Seizures"]
sample_rate = data["SampleRate"]


print(f"Loaded data for {patient_id}:")
print(f"- ECG signal shape: {ecg_df.shape}")
print(f"- Number of seizures: {seizure_df.shape[0]}")
print(f"- Sample rate: {sample_rate} Hz")

print(seizure_df.head())

start_ts = pd.to_datetime(data["StartTime"])
print("Recording start:", start_ts)

# Alternative
# start_ts = pd.to_datetime(ecg_df["Timestamp"].iloc[0])


# %%
seizure_events: List[SeizureEvent] = []

for _, row in seizure_df.iterrows():
    # Byg fulde datetime-stempler fra 'Dato' + tidstekst
    date_str = str(row["Dato"]).strip() if pd.notna(row["Dato"]) else None

    def parse_dt(time_str_col: str) -> Optional[pd.Timestamp]:
        if date_str is None or pd.isna(row[time_str_col]):
            return None
        timestr = str(row[time_str_col]).strip()
        # Tilpas format hvis nødvendigt (her antaget: dd.mm.yy HH:MM:SS)
        return pd.to_datetime(f"{date_str} {timestr}", format="%d.%m.%y %H:%M:%S", errors="coerce")

    start_klinisk = parse_dt("Anfaldsstart Klinisk (tt:mm:ss)")
    stop_klinisk  = parse_dt("Anfaldstop Klinisk (tt:mm:ss)")
    start_eeg     = parse_dt("Anfaldsstart EEG (tt:mm:ss)")
    stop_eeg      = parse_dt("Anfaldstop EEG (tt:mm:ss)")

    if start_klinisk is not None and stop_klinisk is not None:
        klinisk_duration = (stop_klinisk - start_klinisk).total_seconds()
    else:
        klinisk_duration = 0.0

    if start_eeg is not None and stop_eeg is not None:
        eeg_duration = (stop_eeg - start_eeg).total_seconds()
    else:
        eeg_duration = 0.0

    # Vælg hvilken annotation vi tror mest på (klinisk vs EEG)
    if klinisk_duration >= eeg_duration and start_klinisk is not None:
        t0_dt = start_klinisk
        duration = klinisk_duration
    elif start_eeg is not None:
        t0_dt = start_eeg
        duration = eeg_duration
    else:
        continue  # ingen valid annotation

    # Konverter til sekunder fra recording start
    t0 = (t0_dt - start_ts).total_seconds()
    t1 = t0 + duration

    if t1 <= t0:
        continue  # discard hvis duration er 0 eller negativ

    ev = SeizureEvent(
        seizure_id=int(row["Anfald nr."]),
        t0=t0,
        t1=t1,
    )
    seizure_events.append(ev)

print("Antal validerede seizure events:", len(seizure_events))
if seizure_events:
    print("Eksempler:", seizure_events[:3])