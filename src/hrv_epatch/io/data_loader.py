# DataLoader
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import os
import random
import re

from src.hrv_epatch.io.tdms import (
    load_tdms_from_path,
    build_ecg_dataframe,
)
    
DEFAULT_BASE_DIR = r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne"

def _find_base_dirs() -> tuple[Path, Path]:
    """
    Returner (tdms_root, annotations_root).

    TODO: Tilpas stierne herunder til din egen mappe-struktur.
    De er kun eksempler!

    Eksempel (som i dine andre notebooks):
      TDMS-root:   ...\ePatch data from Aarhus to Lausanne\Patients ePatch data
      ANN-root:    ...\Seizure log ePatch patients with seizures
    """
    tdms_root = Path(
        r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Patients ePatch data"
    )
    ann_root = Path(
        r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Seizure log ePatch patients with seizures - excluded seizures removed"
    )
    return tdms_root, ann_root


def _parse_patient_id(patient_label: str) -> int:
    """
    Ekstraher numerisk ID fra fx 'Patient 5' -> 5.
    """
    m = re.search(r"(\d+)", str(patient_label))
    if not m:
        raise ValueError(f"Could not extract numeric patient ID from: {patient_label!r}")
    return int(m.group(1))


def Load_full_ecg_data(patient_label: str) -> Dict[str, Any]:
    """
    Høj-niveau loader brugt i RPeak-Comparison notebooken.

    Returnerer et dict med nøgler:
      - "PatientID": fx 'Patient 5'
      - "ECG": DataFrame med EKG (tid + amplitude)
      - "Seizures": rå seizure-annoteringer (samme struktur som før)
      - "SampleRate": fs (float)
      - "StartTime": datetime for recording start (naiv lokal tid)
      - "TdmsPath": fuld sti til TDMS-filen
    """
    tdms_root, ann_root = _find_base_dirs()
    pid = _parse_patient_id(patient_label)

    # RPeak-Comparison er kun på recording 1 → Patient X_1.tdms
    rec_dir = tdms_root / f"Patient {pid}" / "recording 1"
    tdms_path = rec_dir / f"Patient {pid}_1.tdms"

    if not tdms_path.exists():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

    # Brug den nye TDMS-loader
    sig, meta = load_tdms_from_path(
        str(tdms_path),
        channel_hint="EKG",
        prefer_tz="Europe/Copenhagen",
        assume_source_tz="UTC",
        prefer_naive_local=True,
    )

    ecg_df = build_ecg_dataframe(sig, meta)
    # -> kolonner: ["Timestamp", "Value"] (eller ["TimeSec", "Value"] hvis ingen start_time)

    # ----- Seizure-annoteringer (så build_seizure_windows_from_res stadig virker) -----
    seizure_df = pd.DataFrame()

    # prøv et par normale navne: 'patient 5.xls', 'Patient 5.xls', også .xlsx
    candidates = [
        ann_root / f"patient {pid}.xls",
        ann_root / f"Patient {pid}.xls",
        ann_root / f"patient {pid}.xlsx",
        ann_root / f"Patient {pid}.xlsx",
    ]
    seizure_log_file = None
    for c in candidates:
        if c.exists():
            seizure_log_file = c
            break

    if seizure_log_file is not None:
        # ----------------------------------------------------------
        # Robust Excel loader med automatisk header-detektion
        # ----------------------------------------------------------
        raw = pd.read_excel(seizure_log_file, header=None)

        # Find række hvor kolonnenavne står
        header_row = None
        for i in range(min(10, len(raw))):   # kig de første 10 rækker
            row_vals = raw.iloc[i].astype(str).str.lower()
            if any("anfaldsstart" in v for v in row_vals) and any("dato" in v for v in row_vals):
                header_row = i
                break

        if header_row is None:
            raise ValueError(f"Could not find header row in seizure log: {seizure_log_file}")

        # Brug header-row som kolonnenavne
        df = pd.read_excel(seizure_log_file, header=header_row)

        # Fjern rækker som står under headeren men ikke indeholder reelle data
        df = df[df.iloc[:,0].notna()].reset_index(drop=True)

        seizure_df = df

    else:
        print(f"  - Ingen annoteringsfil fundet for {patient_label}")

    return {
        "PatientID": patient_label,
        "ECG": ecg_df,
        "Seizures": seizure_df,
        "SampleRate": float(meta.fs),
        "StartTime": meta.start_time,
        "TdmsPath": meta.path,
    }

# def Load_full_ecg_data(patient_id: str, base_dir: str = DEFAULT_BASE_DIR):
#     """
#     Load fuldt EKG-signal + alle seizure-annoteringer for en patient.

#     Returnerer en dict med:
#         - "PatientID": patient_id
#         - "ECG": DataFrame med ECG-signal og tidsakse
#         - "Seizures": DataFrame med seizure-annoteringer (kan være tom)
#         - "SampleRate": samplingfrekvens (float)
#         - "StartTime": start-tidspunkt (naiv lokal datetime)
#         - "TdmsPath": sti til den brugte TDMS-fil
#     """
#     # 1) Find og load TDMS via fælles loader
#     signal, meta = load_tdms_for_patient(
#         patient_id=patient_id,
#         base_dir=base_dir,
#         channel_hint="EKG",
#         prefer_tz="Europe/Copenhagen",
#         assume_source_tz="UTC",     # sæt evt. til None, hvis wf_start_time allerede er lokal tid
#         prefer_naive_local=True,
#     )

#     fs = float(meta.fs)

#     # 2) Byg EKG-DataFrame med tidsakse
#     ecg_df = build_ecg_dataframe(signal, meta)

#     # 3) Find og læs seizure-log (hvis den findes)
#     data_dir, seizure_log_dir = _find_base_dirs(base_dir)
#     seizure_df = pd.DataFrame()  # tom fallback

#     if seizure_log_dir is not None and os.path.isdir(seizure_log_dir):
#         # Udtræk patientnummer fra "Patient 1", "Patient 14" osv.
#         m_pid = re.search(r"\d+", patient_id)
#         if m_pid is None:
#             raise ValueError(f"Kunne ikke finde patientnummer i patient_id={patient_id!r}")
#         patient_num = int(m_pid.group(0))

#         candidate_files = []
#         for f in os.listdir(seizure_log_dir):
#             if not f.lower().endswith((".xls", ".xlsx")):
#                 continue

#             # find "patient <tal>" i filnavnet
#             m = re.search(r"patient\s*0*(\d+)", f, flags=re.IGNORECASE)
#             if m and int(m.group(1)) == patient_num:
#                 candidate_files.append(f)

#         print(f"[DEBUG] Potentielle annoteringsfiler for {patient_id}:")
#         for cf in candidate_files:
#             print("   ", cf)

#         if candidate_files:
#             # hvis der er flere (fx 1a/1b), kan du sortere eller filtrere yderligere
#             candidate_files.sort()
#             seizure_log_file = os.path.join(seizure_log_dir, candidate_files[0])
#             print(f"  - Læser seizure-log fra: {seizure_log_file}")

#             raw = pd.read_excel(seizure_log_file, skiprows=5)
#             raw.columns = raw.iloc[0]
#             seizure_df = raw[1:].reset_index(drop=True)
#         else:
#             print(f"  - Ingen annoteringsfil fundet for {patient_id}")

#     return {
#         "PatientID": patient_id,
#         "ECG": ecg_df,
#         "Seizures": seizure_df,
#         "SampleRate": fs,
#         "StartTime": meta.start_time,
#         "TdmsPath": meta.path,
#     }


