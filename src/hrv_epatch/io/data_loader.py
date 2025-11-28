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
import re
    
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
        # Udtræk patientnummer fra "Patient 1", "Patient 14" osv.
        m_pid = re.search(r"\d+", patient_id)
        if m_pid is None:
            raise ValueError(f"Kunne ikke finde patientnummer i patient_id={patient_id!r}")
        patient_num = int(m_pid.group(0))

        candidate_files = []
        for f in os.listdir(seizure_log_dir):
            if not f.lower().endswith((".xls", ".xlsx")):
                continue

            # find "patient <tal>" i filnavnet
            m = re.search(r"patient\s*0*(\d+)", f, flags=re.IGNORECASE)
            if m and int(m.group(1)) == patient_num:
                candidate_files.append(f)

        print(f"[DEBUG] Potentielle annoteringsfiler for {patient_id}:")
        for cf in candidate_files:
            print("   ", cf)

        if candidate_files:
            # hvis der er flere (fx 1a/1b), kan du sortere eller filtrere yderligere
            candidate_files.sort()
            seizure_log_file = os.path.join(seizure_log_dir, candidate_files[0])
            print(f"  - Læser seizure-log fra: {seizure_log_file}")

            raw = pd.read_excel(seizure_log_file, skiprows=5)
            raw.columns = raw.iloc[0]
            seizure_df = raw[1:].reset_index(drop=True)
        else:
            print(f"  - Ingen annoteringsfil fundet for {patient_id}")

    return {
        "PatientID": patient_id,
        "ECG": ecg_df,
        "Seizures": seizure_df,
        "SampleRate": fs,
        "StartTime": meta.start_time,
        "TdmsPath": meta.path,
    }


