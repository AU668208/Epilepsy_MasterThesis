# DataLoader
import os
import pandas as pd

from src.hrv_epatch.io.tdms import (
    load_tdms_for_patient,
    build_ecg_dataframe,
    _find_base_dirs,   # ja, den er "privat", men det er fint internt i dit projekt
)
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import os
import random
    
DEFAULT_BASE_DIR = r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne"

def Load_full_ecg_data(patient_id: str, base_dir: str = DEFAULT_BASE_DIR):
    """
    Load fuldt EKG-signal + alle seizure-annoteringer for en patient.

    Returnerer en dict med:
        - "PatientID": patient_id
        - "ECG": DataFrame med ECG-signal og tidsakse
        - "Seizures": DataFrame med seizure-annoteringer (kan være tom)
        - "SampleRate": samplingfrekvens (float)
        - "StartTime": start-tidspunkt (naiv lokal datetime)
        - "TdmsPath": sti til den brugte TDMS-fil
    """
    # 1) Find og load TDMS via fælles loader
    signal, meta = load_tdms_for_patient(
        patient_id=patient_id,
        base_dir=base_dir,
        channel_hint="EKG",
        prefer_tz="Europe/Copenhagen",
        assume_source_tz="UTC",     # sæt evt. til None, hvis wf_start_time allerede er lokal tid
        prefer_naive_local=True,
    )

    fs = float(meta.fs)

    # 2) Byg EKG-DataFrame med tidsakse
    ecg_df = build_ecg_dataframe(signal, meta)

    # 3) Find og læs seizure-log (hvis den findes)
    data_dir, seizure_log_dir = _find_base_dirs(base_dir)
    seizure_df = pd.DataFrame()  # tom fallback

    if seizure_log_dir is not None and os.path.isdir(seizure_log_dir):
        # Find en Excel-fil i seizure-log-mappen, der matcher patient_id i filnavnet
        cand = [
            f for f in os.listdir(seizure_log_dir)
            if patient_id in f and f.lower().endswith((".xls", ".xlsx"))
        ]
        if cand:
            seizure_log_file = os.path.join(seizure_log_dir, cand[0])
            # Her kan du tilpasse skiprows/kolonner til din log-struktur
            raw = pd.read_excel(seizure_log_file, skiprows=5)
            raw.columns = raw.iloc[0]
            seizure_df = raw[1:].reset_index(drop=True)

    return {
        "PatientID": patient_id,
        "ECG": ecg_df,
        "Seizures": seizure_df,
        "SampleRate": fs,
        "StartTime": meta.start_time,
        "TdmsPath": meta.path,
    }


