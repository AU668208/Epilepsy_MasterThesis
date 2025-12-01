# %%
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# >>> Tilpas disse tre variabler til din struktur:
DATA_ROOT = Path("E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Patients ePatch data")      # mappe med TDMS
ANNOT_ROOT = Path("E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne\Seizure log ePatch patients with seizures")        # mappe med annotationsfiler (csv/xlsx)
OUTPUT_ROOT = Path("E:\ML algoritme tl anfaldsdetektion vha HRV\LabView-Results\Validation_export")         # hvor vi skriver CSV + PNG

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Hvilke kolonnenavne forventer vi i annotationsfilen?
# Varianten her antager én række pr. anfald med enten absolutte tider (ISO8601) eller relative sekunder fra filstart.
ANNOT_COLS = dict(
    patient="Anfald nr.",
    file="Dato",              # navn på tdms-fil for den pågældende optagelse
    start_Clinic="Anfaldsstart Klinisk (tt:mm:ss)",        # enten absolut tid (ISO) eller float sekunder rel. til filstart
    end_Clinic="Anfaldstop Klinisk (tt:mm:ss)",            # samme format som 'start'
    start_eeg="Anfaldsstart EEG (tt:mm:ss)",                # enten absolut tid (ISO) eller float sekunder rel. til filstart
    end_eeg="Anfaldstop EEG (tt:mm:ss)",                    # samme format som 'start'
    is_absolute_time=True         # sæt til False hvis start/end er i sekunder relativt til signalstart
)

from typing import Optional, Tuple, Dict, Any
from nptdms import TdmsFile
import re

def load_ecg_tdms(tdms_path: Path,
                  group_hint: Optional[str] = None,
                  channel_hint: Optional[str] = None
                 ) -> Dict[str, Any]:
    """
    Indlæser 1. kanal fra en TDMS (eller den du angiver).
    Returnerer: dict(signal=np.array, fs=float, start_time=datetime, duration=float sek, name=str)
    Forsøger at udlede starttid fra almindelige NI/EPatch properties (wf_start_time, NI_ExpStartTimeStamp, name).
    """
    
    t = TdmsFile.read(tdms_path)

    # Vælg kanal
    if group_hint and channel_hint:
        ch = t[group_hint][channel_hint]
    else:
        # default: tag første kanal
        g = t.groups()[0]
        ch = g.channels()[0]

    x = ch[:]  # numpy array
    # Sample rate
    fs = ch.properties.get("wf_increment", None)
    if fs is not None:
        fs = 1.0 / fs
    else:
        # fallback – hvis ikke tilgængeligt, så kræv at du sætter fs manuelt senere
        raise ValueError("Kunne ikke finde fs i TDMS (wf_increment mangler).")

    # Starttid
    start_dt = None
    # typiske felter:
    cand = [
        ch.properties.get("wf_start_time", None),
        ch.properties.get("NI_ExpStartTimeStamp", None),
        ch.properties.get("NI_ExpTimeStamp", None),
        t.properties.get("wf_start_time", None),
        t.properties.get("NI_ExpStartTimeStamp", None),
        t.properties.get("NI_ExpTimeStamp", None),
    ]
    for c in cand:
        if c is None:
            continue
        # Nogle drivere returnerer allerede datetime, andre string
        if isinstance(c, datetime):
            start_dt = c
            break
        else:
            # Forsøg parse ISO/NI-format
            for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                        "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
                try:
                    # Hvis der mangler TZ, antag UTC (eller sæt din egen TZ)
                    dt = datetime.strptime(str(c), fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    start_dt = dt
                    break
                except Exception:
                    pass
        if start_dt:
            break

    # Fallback: forsøg file 'name' property som tidsstempel (hvis du bruger den)
    if start_dt is None and "name" in t.properties:
        try:
            dt = datetime.fromisoformat(str(t.properties["name"]))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            start_dt = dt
        except Exception:
            pass

    # Sidste fallback: brug filsystemets mtime (dårligere, men bedre end intet)
    if start_dt is None:
        start_dt = datetime.fromtimestamp(tdms_path.stat().st_mtime, tz=timezone.utc)

    return dict(
        signal=x.astype(float),
        fs=float(fs),
        start_time=start_dt,
        duration=len(x)/float(fs),
        name=tdms_path.stem,
        group=ch.group_name,
        channel=ch.name,
        props=dict(ch.properties)
    )


def time_to_index(t_abs: datetime, start_time: datetime, fs: float) -> int:
    """ Konverter absolut timestamp -> sampleindeks """
    dt = (t_abs - start_time).total_seconds()
    return max(0, int(round(dt * fs)))


def rels_to_slice(t0_rel_s: float, t1_rel_s: float, fs: float, n: int) -> slice:
    """ Rel. sekunder -> slice i [0, n). """
    i0 = max(0, int(round(t0_rel_s * fs)))
    i1 = min(n, int(round(t1_rel_s * fs)))
    if i1 <= i0:
        i1 = min(n, i0 + 1)
    return slice(i0, i1)

def export_segment(signal: np.ndarray,
                   fs: float,
                   start_time: datetime,
                   seg_slice: slice,
                   seizure_start_rel: float,
                   seizure_end_rel: float,
                   out_csv: Path,
                   out_png: Path,
                   title: str):
    """
    Gemmer CSV + figur for givne slice [seg_slice].
    seizure_*_rel er relativ tid (sek) inden for segmentet for vertikale markører.
    """
    seg = signal[seg_slice]
    n = len(seg)
    t_rel = np.arange(n) / fs  # sek rel. til segmentstart
    # Absolutte tidsstempler pr. sample (ISO)
    t_abs = [ (start_time + timedelta(seconds=float(seg_slice.start)/fs) + timedelta(seconds=float(tt))).isoformat() 
             for tt in t_rel ]

    # CSV
    df = pd.DataFrame({
        "t_abs": t_abs,
        "t_rel_s": t_rel,
        "ecg": seg
    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(12, 3))
    plt.plot(t_rel, seg, linewidth=0.8)
    ymin, ymax = float(np.min(seg)), float(np.max(seg))
    # Markér start/stop
    plt.axvline(seizure_start_rel, linestyle="--", linewidth=1.2)
    plt.axvline(seizure_end_rel, linestyle="--", linewidth=1.2)
    plt.ylim(ymin - 0.05*(ymax-ymin), ymax + 0.05*(ymax-ymin))
    plt.xlabel("Tid (s) relativt til segmentstart")
    plt.ylabel("EKG (arb. enhed)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def process_one_annotation(ecg_file: Path,
                           ann_row: pd.Series,
                           out_root: Path,
                           idx_in_patient: int):
    """
    Kører hele flowet for én annotation i én TDMS-fil.
    Forventer kolonner iht. ANNOT_COLS.
    """
    # Load ECG
    rec = load_ecg_tdms(ecg_file)
    x, fs, t0 = rec["signal"], rec["fs"], rec["start_time"]
    n = len(x)
    rec_end = t0 + timedelta(seconds=n/fs)

    # Parse annotation tider
    if ANNOT_COLS["is_absolute_time"]:
        t_start = pd.to_datetime(ann_row[ANNOT_COLS["start"]], utc=True).to_pydatetime()
        t_end   = pd.to_datetime(ann_row[ANNOT_COLS["end"]],   utc=True).to_pydatetime()
    else:
        # relative sekunder fra filstart
        t_start = t0 + timedelta(seconds=float(ann_row[ANNOT_COLS["start"]]))
        t_end   = t0 + timedelta(seconds=float(ann_row[ANNOT_COLS["end"]]))

    # Vindue: -60s .. +60s omkring anfaldsstart (eller omkring hele intervallet – her bruger vi start)
    win_start = t_start - timedelta(seconds=60)
    win_end   = t_start + timedelta(seconds=60)

    # Begræns til recordings grænser
    if win_start < t0: win_start = t0
    if win_end   > rec_end: win_end = rec_end

    # Slice for vinduet
    i0 = time_to_index(win_start, t0, fs)
    i1 = time_to_index(win_end,   t0, fs)
    seg_slice = slice(i0, i1)

    # Relativ markørplaceringer inden for segmentet
    seiz_start_rel = (t_start - win_start).total_seconds()
    seiz_end_rel   = (t_end   - win_start).total_seconds()

    # Output-stier
    base = out_root / rec["name"] / f"seizure_{idx_in_patient:02d}"
    csv_path = base / "ecg_window_±60s.csv"
    png_path = base / "ecg_window_±60s.png"

    export_segment(
        signal=x, fs=fs, start_time=t0, seg_slice=seg_slice,
        seizure_start_rel=seiz_start_rel, seizure_end_rel=seiz_end_rel,
        out_csv=csv_path, out_png=png_path,
        title=f"{rec['name']} – Anfald {idx_in_patient}  (±60s)"
    )

    # Find ikke-anfalds vindue ≈ 1 time efter anfaldets slutning
    candidate_start = t_end + timedelta(hours=1)
    not_ok_reason = None
    if candidate_start + timedelta(seconds=120) > rec_end:
        # Fallback: prøv 1 time før anfaldsstart (hvis muligt)
        candidate_start = t_start - timedelta(hours=1)
        if candidate_start < t0:
            # Sidste fallback: midt i optagelsen langt fra anfaldet
            mid = t0 + (rec_end - t0)/2
            candidate_start = max(t0, min(rec_end - timedelta(seconds=120), mid))
            not_ok_reason = "faldt tilbage til midt i optagelsen"
    # Brug 120 sekunder “sikker” ikke-anfaldsudsnit
    not_slice = slice(
        time_to_index(candidate_start, t0, fs),
        time_to_index(candidate_start + timedelta(seconds=120), t0, fs)
    )
    not_base = base / "nonseizure_+1h"
    export_segment(
        signal=x, fs=fs, start_time=t0, seg_slice=not_slice,
        seizure_start_rel=-1, seizure_end_rel=-1,   # ingen markører
        out_csv=not_base.with_suffix(".csv"),
        out_png=not_base.with_suffix(".png"),
        title=f"{rec['name']} – Ikke-anfaldsvindue (~+1h){' ('+not_ok_reason+')' if not_ok_reason else ''}"
    )

    return dict(
        ecg_file=str(ecg_file),
        seizure_csv=str(csv_path),
        seizure_png=str(png_path),
        nonseizure_csv=str(not_base.with_suffix('.csv')),
        nonseizure_png=str(not_base.with_suffix('.png')),
        fs=fs, start_time=str(t0)
    )

def load_annotations_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        # Excel: patient id is in row 2, data starts at row 7
        # Først læs de første 2 rækker for at hente patient-id
        preview = pd.read_excel(path, header=None, nrows=2)
        if len(preview) >= 2:
            patient_row = preview.iloc[1].dropna().astype(str).str.strip()
        else:
            patient_row = preview.iloc[0].dropna().astype(str).str.strip()

        if len(patient_row) == 0:
            patient_id = ""
        else:
            patient_id = patient_row.iloc[1]
            # Hvis cellen fx indeholder "Patient 5", prøv at udtrække tallet
            m = re.search(r'\d+', patient_id)
            if m:
                patient_id = m.group(0)
            print(f"[INFO] Fundet patient-id '{patient_id}' i filen '{path.name}'")

        # Læs selve dataene (starter på række 7 => skiprows=6)
        df = pd.read_excel(path, skiprows=6)

        # Sæt patient-kolonnen hvis den mangler
        if ANNOT_COLS["patient"] not in df.columns:
            df[ANNOT_COLS["patient"]] = patient_id
    else:
        df = pd.read_csv(path, skiprows=6)
    # Sikr obligatoriske kolonner (brug nøglerne defineret i ANNOT_COLS)
    need = [
        ANNOT_COLS["start_Clinic"],
        ANNOT_COLS["start_eeg"],
        ANNOT_COLS["end_Clinic"],
        ANNOT_COLS["end_eeg"],
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i annotationsfilen: {missing}")
    return df

def run_validation_pipeline(ann_path: Path,
                            data_root: Path = DATA_ROOT,
                            out_root: Path = OUTPUT_ROOT,
                            patient_filter: Optional[str] = None):
    df = load_annotations_table(ann_path)
    if patient_filter is not None:
        df = df[df[ANNOT_COLS["patient"]] == patient_filter].copy()

    results = []
    for (patient_id, ecg_file), sub in df.groupby([ANNOT_COLS["patient"], ANNOT_COLS["file"]]):
        tdms_path = (data_root / ecg_file)
        if not tdms_path.exists():
            print(f"[ADVARSEL] Fil findes ikke: {tdms_path}")
            continue
        sub = sub.sort_values(by=ANNOT_COLS["start"]).reset_index(drop=True)
        for k, row in sub.iterrows():
            try:
                res = process_one_annotation(tdms_path, row, out_root/(str(patient_id)), idx_in_patient=k+1)
                print(f"[OK] {patient_id} {ecg_file}  seizure#{k+1}  ->  {res['seizure_csv']}")
                results.append(res)
            except Exception as e:
                print(f"[FEJL] {patient_id} {ecg_file} seizure#{k+1}: {e}")
    return pd.DataFrame(results)


# %%
ann_file = ANNOT_ROOT / "Patient 5.xls"   # eller .xlsx
summary = run_validation_pipeline(ann_file, patient_filter=None)
# summary


# %%



