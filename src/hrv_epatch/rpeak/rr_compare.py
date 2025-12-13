
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from src.hrv_epatch.io.tdms import load_tdms_from_path
from src.hrv_epatch.io.labview import read_labview_rr, read_header_datetime_lvm

# ---------------------------------------------------------
# Konfiguration: én optagelse (patient + recording)
# ---------------------------------------------------------

@dataclass
class RecordingConfig:
    patient_id: int
    recording_id: int
    tdms_path: Path
    lvm_path: Path
    fs: float = 512.0
    channel_hint: str = "EKG"
    tz: str = "Europe/Copenhagen"

    # NYT:
    algo_id: str = "neurokit"        # hvilken NeuroKit-peak-metode
    trim_label: str | None = None    # fx "no_trim", "trim_table", "trim_30min"



# ---------------------------------------------------------
# Hjælpefunktioner til RR / tid / matching
# ---------------------------------------------------------


def rpeaks_to_times_and_rr(r_idx: np.ndarray, fs: float, t0_tdms_dt) -> Tuple[np.ndarray, np.ndarray]:
    r_idx = np.asarray(r_idx, dtype=float)
    if r_idx.size == 0:
        return np.array([]), np.array([])

    t0_epoch = _to_epoch_seconds(t0_tdms_dt)
    t_peaks = t0_epoch + r_idx / fs

    if t_peaks.size < 2:
        return t_peaks, np.array([])

    rr = np.diff(t_peaks)
    return t_peaks, rr



def reconstruct_labview_peak_times(rr_lv: np.ndarray, t0_lv_dt) -> Tuple[np.ndarray, np.ndarray]:
    rr_lv = np.asarray(rr_lv, dtype=float).ravel()
    if rr_lv.size == 0:
        return np.array([]), rr_lv

    t0_epoch = _to_epoch_seconds(t0_lv_dt)
    t_peaks = t0_epoch + np.cumsum(np.insert(rr_lv, 0, 0.0))
    return t_peaks, rr_lv



def reconstruct_labview_r_times(rr_lv: np.ndarray, t0_lv_dt):
    """
    Som før: starttid for hvert RR-interval (dvs. ved peak i).
    Bygger nu på peak-funktionen, så vi har en konsistent definition.
    """
    t_peaks, rr_lv = reconstruct_labview_peak_times(rr_lv, t0_lv_dt)
    if t_peaks.size <= 1:
        return np.array([]), rr_lv
    # starttid for interval i er ved peak i (ikke den sidste)
    return t_peaks[:-1], rr_lv



def match_rr_series_time_based(
    t_R_lv: np.ndarray,
    RR_lv: np.ndarray,
    t_R_py: np.ndarray,
    RR_py: np.ndarray,
    delta_s: float = 0.0,
    tol_s: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matcher RR-intervaller ved at align'e tidspunkter.

    t_R_lv, RR_lv: LabVIEW (reference)
    t_R_py, RR_py: Python
    delta_s: ekstra tids-skift der lægges til Python-tiderne
    tol_s: max tidsafstand mellem starttidspunkter for to RR-intervaller
           for at de betragtes som "samme" interval.

    Returnerer:
    - rr_lv_matched, rr_py_matched: arrays med matched RR'er
    """
    RR_lv = np.asarray(RR_lv, dtype=float)
    RR_py = np.asarray(RR_py, dtype=float)

    if RR_lv.size == 0 or RR_py.size == 0:
        return np.array([]), np.array([])

    # Starttid for RR-interval: LabVIEW: t_R_lv[i]
    # Python: RR_py[i] starter ved t_R_py[i]
    t_start_lv = np.asarray(t_R_lv, dtype=float)
    t_start_py = np.asarray(t_R_py[:-1], dtype=float) + delta_s

    rr_lv_matched = []
    rr_py_matched = []

    j = 0
    for i, t_lv in enumerate(t_start_lv):
        # Ryk j frem, til vi er tæt nok på t_lv
        while j < len(t_start_py) and t_start_py[j] < t_lv - tol_s:
            j += 1
        if j >= len(t_start_py):
            break

        if abs(t_start_py[j] - t_lv) <= tol_s:
            rr_lv_matched.append(RR_lv[i])
            rr_py_matched.append(RR_py[j])
            j += 1

    return np.asarray(rr_lv_matched), np.asarray(rr_py_matched)


def find_best_delta(
    t_R_lv: np.ndarray,
    RR_lv: np.ndarray,
    t_R_py: np.ndarray,
    RR_py: np.ndarray,
    delta_range_s: Tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_s: float = 0.04,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Finder det delta_s der giver lavest MAE mellem matchende RR-intervaller
    (enkel grid search).

    Returnerer:
    - best_delta (sekunder)
    - best_rr_lv, best_rr_py: matched RR-serier for dette delta
    """
    deltas = np.arange(delta_range_s[0], delta_range_s[1] + delta_step_s, delta_step_s)

    best_delta = 0.0
    best_score = -np.inf
    best_rr_lv: Optional[np.ndarray] = None
    best_rr_py: Optional[np.ndarray] = None

    for delta in deltas:
        lv_m, py_m = match_rr_series_time_based(
            t_R_lv, RR_lv, t_R_py, RR_py, delta_s=delta, tol_s=tol_s
        )
        if lv_m.size < 5:
            continue  # for lidt data

        diff = py_m - lv_m
        mae = np.mean(np.abs(diff))
        score = -mae  # lav MAE = god => høj score

        if score > best_score:
            best_score = score
            best_delta = delta
            best_rr_lv = lv_m
            best_rr_py = py_m

    if best_rr_lv is None or best_rr_py is None:
        # Fald tilbage: ingen god delta fundet
        return 0.0, np.array([]), np.array([])

    return best_delta, best_rr_lv, best_rr_py


def compute_rr_metrics(
    rr_ref: np.ndarray,
    rr_test: np.ndarray,
) -> Dict[str, float]:
    """
    RR-ref = LabVIEW (”sandhed”), RR-test = Python-algoritme.
    """
    rr_ref = np.asarray(rr_ref, dtype=float)
    rr_test = np.asarray(rr_test, dtype=float)

    if rr_ref.size == 0 or rr_test.size == 0:
        return {
            "n_common": 0,
            "rr_mae_ms": np.nan,
            "rr_rmse_ms": np.nan,
            "rr_corr": np.nan,
            "mean_hr_ref_bpm": np.nan,
            "mean_hr_test_bpm": np.nan,
            "mean_hr_diff_bpm": np.nan,
        }

    n = min(rr_ref.size, rr_test.size)
    rr_ref = rr_ref[:n]
    rr_test = rr_test[:n]

    diff = rr_test - rr_ref

    mae_ms = float(np.mean(np.abs(diff)) * 1000.0)
    rmse_ms = float(np.sqrt(np.mean(diff**2)) * 1000.0)
    if n > 1:
        rr_corr = float(np.corrcoef(rr_ref, rr_test)[0, 1])
    else:
        rr_corr = np.nan

    mean_hr_ref = 60.0 / float(np.mean(rr_ref))
    mean_hr_test = 60.0 / float(np.mean(rr_test))
    mean_hr_diff = float(mean_hr_test - mean_hr_ref)

    return {
        "n_common": int(n),
        "rr_mae_ms": mae_ms,
        "rr_rmse_ms": rmse_ms,
        "rr_corr": rr_corr,
        "mean_hr_ref_bpm": mean_hr_ref,
        "mean_hr_test_bpm": mean_hr_test,
        "mean_hr_diff_bpm": mean_hr_diff,
    }


import numpy as np
import neurokit2 as nk

# Alle clean-metoder, som ecg_clean kender
SUPPORTED_CLEAN_METHODS = {
    "neurokit",
    "biosppy",
    "pantompkins1985",
    "hamilton2002",
    "elgendi2010",
    "engzeemod2012",
    "vg",
}

# Mapping: hvilken clean-metode der passer bedst til hvilken peak-metode
PEAK_TO_CLEAN = {
    # 1: samme navn → samme pipeline
    "neurokit": "neurokit",
    "pantompkins1985": "pantompkins1985",
    "hamilton2002": "hamilton2002",
    "elgendi2010": "elgendi2010",
    "engzeemod2012": "engzeemod2012",

    # 2: Emrich/FastNVG → bruger 'vg'-cleaning ifølge docs
    "emrich2023": "vg",

    # 3: resten (zong, martinez, …, promac) får fallback til 'neurokit'
    # (ingen entries her → default nedenfor)
}


def detect_rpeaks_python(
    ecg: np.ndarray,
    fs: float,
    method: str = "neurokit",
) -> np.ndarray:
    """
    R-peak detection using NeuroKit2.

    - Cleaning method is chosen to match the peak detector when possible.
    - For peak methods without a dedicated cleaning pipeline, 'neurokit'
      cleaning is used as a generic fallback.
    """
    ecg = np.asarray(ecg, dtype=float).ravel()
    method_l = method.lower()

    # NeuroKit forventer i praksis at sampling_rate er et heltal
    fs_int = int(round(fs))

    # Vælg clean-metode ud fra peak-metoden
    clean_method = PEAK_TO_CLEAN.get(method_l, "neurokit")
    if clean_method not in SUPPORTED_CLEAN_METHODS:
        clean_method = "neurokit"

    # 1) Cleaning
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs_int, method=clean_method)

    # 2) R-peaks med ønsket algoritme
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs_int, method=method_l)

    rpeaks_idx = info.get("ECG_R_Peaks", None)
    if rpeaks_idx is None:
        rpeaks_idx = np.where(info["ECG_R_Peaks"] == 1)[0]

    return np.asarray(rpeaks_idx, dtype=np.int64)


# def detect_rpeaks_python(ecg: np.ndarray, fs: float, method: str = "neurokit") -> np.ndarray:
#     """
#     R-peak detektion med NeuroKit2.

#     method:
#       - bruges som 'method' i nk.ecg_peaks, fx:
#         "neurokit", "pantompkins1985", "hamilton2002",
#         "christov2004", "elgendi2010", "kalidas2017", ...

#     Vi holder cleaning fast til 'neurokit' for nu.
#     """
#     ecg = np.asarray(ecg, dtype=float).ravel()

#     # Clean
#     ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")

#     # Peaks med valgt metode
#     _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method=method)

#     rpeaks_idx = info.get("ECG_R_Peaks", None)
#     if rpeaks_idx is None:
#         # fallback hvis nøglen mod forventning mangler
#         # (afhænger lidt af NeuroKit-version)
#         rpeaks_idx = np.where(info["ECG_R_Peaks"] == 1)[0]

#     return np.asarray(rpeaks_idx, dtype=np.int64)

from pathlib import Path
import numpy as np

def _rpeak_cache_path(
    cache_dir: Path,
    patient_id: int,
    recording_id: int,
    algo_id: str,
    fs: float,
    max_duration_s: float | None,
    recording_uid: str | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fs_tag = int(round(fs))
    dur_tag = "full" if max_duration_s is None else f"{int(round(max_duration_s))}s"
    base = recording_uid if recording_uid is not None else f"p{patient_id:02d}_r{recording_id:02d}"
    fname = f"{base}__{algo_id}__fs{fs_tag}__{dur_tag}__rpeaks_idx.npz"
    return cache_dir / fname


def get_or_compute_rpeaks_idx(
    ecg: np.ndarray,
    fs: float,
    cfg,
    max_duration_s: float | None,
    cache_dir: Path | None = None,
    force_recompute: bool = False,
    recording_uid: str | None = None,
) -> np.ndarray:
    """
    Returnerer rpeaks sample-indeks (np.int64).
    Hvis cache_dir er angivet, forsøger den at loade/gemme.
    """
    if cache_dir is not None:
        p = _rpeak_cache_path(
            cache_dir=cache_dir,
            patient_id=cfg.patient_id,
            recording_id=cfg.recording_id,
            algo_id=cfg.algo_id,
            fs=fs,
            max_duration_s=max_duration_s,
            recording_uid=recording_uid,
        )
        if (not force_recompute) and p.exists():
            data = np.load(p, allow_pickle=False)
            return data["r_idx"].astype(np.int64)

    # compute
    r_idx = detect_rpeaks_python(ecg, fs=fs, method=cfg.algo_id)

    if cache_dir is not None:
        np.savez_compressed(p, r_idx=r_idx.astype(np.int32))  # int32 er rigeligt til sample-indeks

    return r_idx

def sync_peak_trains(t_lv, t_py, max_search=10):
        # find første fælles peak i starten
        for i in range(max_search):
            j = np.argmin(np.abs(t_py - t_lv[i]))
            if abs(t_py[j] - t_lv[i]) < 0.2:  # 200 ms
                return t_lv[i:], t_py[j:]
        raise RuntimeError("No common peak found")

import pandas as pd

def _to_epoch_seconds(dt, tz="Europe/Copenhagen") -> float:
    """
    Robust datetime -> epoch seconds.
    Ensretter timezone for både tz-aware og tz-naive datetimes.
    """
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return float(ts.timestamp())

import numpy as np

def crop_tdms_to_lvm_window(
    r_idx_py: np.ndarray,
    fs: float,
    tdms_start_epoch: float,
    lv_peak_times_epoch: np.ndarray,
):
    """
    Begræns Python peaks til LVM-vinduet (mellem første og sidste LV peak).
    Returnerer:
      - r_idx_py_crop (sample-indeks i ORIGINAL TDMS)
      - t_py_epoch_crop (epoch tider for de crop’ede Python peaks)
      - lv_window_start_epoch, lv_window_end_epoch
      - start_idx, end_idx (TDMS sample indices for vinduet)
    """
    lv_window_start = float(lv_peak_times_epoch[0])
    lv_window_end   = float(lv_peak_times_epoch[-1])

    start_idx = int(np.ceil((lv_window_start - tdms_start_epoch) * fs))
    end_idx   = int(np.floor((lv_window_end   - tdms_start_epoch) * fs))

    # Safety
    start_idx = max(start_idx, 0)
    end_idx   = max(end_idx, start_idx + 1)

    r_idx_py = np.asarray(r_idx_py, dtype=np.int64)
    keep = (r_idx_py >= start_idx) & (r_idx_py <= end_idx)
    r_idx_py_crop = r_idx_py[keep]

    t_py_epoch_crop = tdms_start_epoch + (r_idx_py_crop.astype(float) / fs)

    return r_idx_py_crop, t_py_epoch_crop, lv_window_start, lv_window_end, start_idx, end_idx

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Du har allerede disse i rr_compare.py (navne kan variere hos dig)
# from src.hrv_epatch.io.tdms import load_tdms_from_path
# from src.hrv_epatch.io.labview import read_labview_rr, read_header_datetime_lvm
# from src.hrv_epatch.rpeak.detect import detect_rpeaks_python  (eller hvor den ligger)
# from src.hrv_epatch.rpeak.rr_compare import find_best_delta, compute_rr_metrics, ...
# from src.hrv_epatch.rpeak.rr_compare import match_rpeaks_time_based_global, compute_peak_metrics

def _reconstruct_lv_peak_times_epoch(rr_lv_s: np.ndarray, lv_header_dt) -> np.ndarray:
    """Returnér LabVIEW peak tider i epoch sekunder, baseret på header start + kumulativ sum af RR."""
    rr_lv_s = np.asarray(rr_lv_s, float).ravel()
    t0_epoch = float(lv_header_dt.timestamp())
    t_rel = np.cumsum(np.insert(rr_lv_s, 0, 0.0))  # første peak ved t0
    return t0_epoch + t_rel

# def _cache_path_for_rpeaks(
#     cache_dir: Path,
#     tdms_path: Path,
#     algo_id: str,
#     max_duration_s: Optional[float],
# ) -> Path:
#     """
#     Cache filnavn: baseret på TDMS filnavn + algo + evt max_duration.
#     (Du kan gøre det endnu mere robust med recording_uid hvis du vil.)
#     """
#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     stem = tdms_path.stem
#     dur_tag = "full" if max_duration_s is None else f"dur{int(max_duration_s)}s"
#     fname = f"{stem}__{algo_id}__{dur_tag}__rpeaks.npy"
#     return cache_dir / fname

def _crop_python_peaks_to_lv_window(
    r_idx_py: np.ndarray,
    fs: float,
    tdms_start_epoch: float,
    lv_peak_times_epoch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """
    Begræns Python peaks til LVM-vinduet (mellem første og sidste LV peak).

    Returnerer:
      r_idx_py_crop, t_py_epoch_crop, lv_window_start_epoch, lv_window_end_epoch, start_idx, end_idx
    """
    lv_window_start = float(lv_peak_times_epoch[0])
    lv_window_end = float(lv_peak_times_epoch[-1])

    start_idx = int(np.ceil((lv_window_start - tdms_start_epoch) * fs))
    end_idx = int(np.floor((lv_window_end - tdms_start_epoch) * fs))

    # safety
    start_idx = max(start_idx, 0)
    end_idx = max(end_idx, start_idx + 1)

    r_idx_py = np.asarray(r_idx_py, dtype=np.int64)
    keep = (r_idx_py >= start_idx) & (r_idx_py <= end_idx)
    r_idx_py_crop = r_idx_py[keep]

    t_py_epoch_crop = tdms_start_epoch + (r_idx_py_crop.astype(float) / fs)

    return r_idx_py_crop, t_py_epoch_crop, lv_window_start, lv_window_end, start_idx, end_idx

def _crop_peaks_to_overlap(
    t_lv_epoch: np.ndarray,
    t_py_epoch: np.ndarray,
    best_delta_s: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Crop LV og PY peaks til fælles overlap EFTER at PY er shifted med best_delta.
    Returnerer: t_lv_ov, t_py_ov (ikke-shifted!), overlap_start, overlap_end
    """
    t_lv_epoch = np.asarray(t_lv_epoch, float).ravel()
    t_py_epoch = np.asarray(t_py_epoch, float).ravel()

    t_py_shift = t_py_epoch + float(best_delta_s)

    overlap_start = max(float(t_lv_epoch[0]), float(t_py_shift[0]))
    overlap_end = min(float(t_lv_epoch[-1]), float(t_py_shift[-1]))

    mask_lv = (t_lv_epoch >= overlap_start) & (t_lv_epoch <= overlap_end)
    mask_py = (t_py_shift >= overlap_start) & (t_py_shift <= overlap_end)

    return t_lv_epoch[mask_lv], t_py_epoch[mask_py], overlap_start, overlap_end

import numpy as np

def _estimate_delta_from_peaks(
    t_ref: np.ndarray,
    t_test: np.ndarray,
    max_abs_dt_s: float = 1.0,
) -> float:
    """
    Estimér global tidsforskydning (delta) så t_test + delta ≈ t_ref.
    Bruger nearest-neighbour med 2-pointer og tager median af dt.
    """
    t_ref = np.asarray(t_ref, float)
    t_test = np.asarray(t_test, float)
    if t_ref.size == 0 or t_test.size == 0:
        return 0.0

    i = 0
    dts = []
    for tr in t_ref:
        # flyt i så t_test[i] er nær tr
        while i + 1 < t_test.size and abs(t_test[i + 1] - tr) <= abs(t_test[i] - tr):
            i += 1
        dt = tr - t_test[i]  # delta som skal lægges på test for at ramme ref
        if abs(dt) <= max_abs_dt_s:
            dts.append(dt)

    if len(dts) == 0:
        return 0.0
    return float(np.median(dts))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _win_idx_from_epoch(t_epoch: float, tdms_start_epoch: float, win_s: float) -> int:
    """Map absolut epoch-tid til 10s window_idx (0-based) relativt til TDMS start."""
    return int(np.floor((t_epoch - tdms_start_epoch) / win_s))

def attach_window_context_for_events(
    df_events: pd.DataFrame,
    df_win: pd.DataFrame,
    win_s: float = 10.0,
) -> pd.DataFrame:
    """
    df_events: skal have ['t_event_epoch','tdms_start_epoch','patient_id','recording_id','algo_id','event_type']
    df_win: din windows tabel (fx df_win_with_seiz) med mindst:
        ['patient_id','recording_id','window_idx','ours_bad', 'is_noiseburst','is_flatline','is_clipping',
         'nk_averageQRS','nk_zhao2018','is_seizure_window','context' ...]
    """
    df = df_events.copy()
    df["window_idx"] = df.apply(
        lambda r: _win_idx_from_epoch(r["t_event_epoch"], r["tdms_start_epoch"], win_s),
        axis=1
    )

    # Join på window_idx + (patient_id, recording_id, algo_id hvis du har pr-metode vinduer)
    # Ofte er df_win ikke algo-specifik; så join kun på patient+rec+window.
    keys_left = ["patient_id", "recording_id", "window_idx"]
    keys_right = ["patient_id", "recording_id", "window_idx"]

    out = df.merge(df_win, on=keys_left, how="left", suffixes=("", "_win"))
    return out

def build_peak_event_table(
    t_peaks_lv: np.ndarray,
    t_peaks_py: np.ndarray,
    tp_lv_idx: np.ndarray,
    tp_py_idx: np.ndarray,
    fn_lv_idx: np.ndarray,
    fp_py_idx: np.ndarray,
    *,
    patient_id: int,
    recording_id: int,
    algo_id: str,
    tdms_start_epoch: float,
    best_delta_s: float,
) -> pd.DataFrame:
    """
    Konvention (vigtigt):
      - LabVIEW peaks er reference (LV)
      - Python peaks er test (PY)
      - best_delta_s er den delta du bruger til at align'e PY til LV (samme som i match-funktionen)

    Returnerer DataFrame med én række pr event:
      event_type in {'FN','FP','TP'}
      t_event_epoch = event-time i absolut epoch (aligned til LV tidsakse)
    """
    rows = []

    # TP: brug LV tid som "event time" (og gem residual også)
    for i_lv, i_py in zip(tp_lv_idx, tp_py_idx):
        t_lv = float(t_peaks_lv[i_lv])
        t_py_aligned = float(t_peaks_py[i_py] + best_delta_s)
        rows.append({
            "event_type": "TP",
            "t_event_epoch": t_lv,
            "dt_ms": (t_py_aligned - t_lv) * 1e3,
        })

    # FN: peaks i LV som PY ikke fandt
    for i_lv in fn_lv_idx:
        rows.append({
            "event_type": "FN",
            "t_event_epoch": float(t_peaks_lv[i_lv]),
            "dt_ms": np.nan,
        })

    # FP: peaks i PY som LV ikke har
    for i_py in fp_py_idx:
        rows.append({
            "event_type": "FP",
            "t_event_epoch": float(t_peaks_py[i_py] + best_delta_s),  # align til LV
            "dt_ms": np.nan,
        })

    df = pd.DataFrame(rows)
    df.insert(0, "patient_id", patient_id)
    df.insert(1, "recording_id", recording_id)
    df.insert(2, "algo_id", algo_id)
    df["tdms_start_epoch"] = float(tdms_start_epoch)
    df["best_delta_s"] = float(best_delta_s)
    return df

def plot_peak_audit_examples(
    ecg: np.ndarray,
    fs: float,
    tdms_start_epoch: float,
    df_events_ctx: pd.DataFrame,
    *,
    n_each: int = 10,
    pre_s: float = 1.5,
    post_s: float = 1.5,
    title_prefix: str = "",
):
    """
    Plots af FP/FN events. df_events_ctx bør være output fra attach_window_context_for_events().
    """
    ecg = np.asarray(ecg, float).ravel()

    def _plot_one(ax, t_epoch, row):
        t_rel = t_epoch - tdms_start_epoch
        i0 = int(max(0, np.floor((t_rel - pre_s) * fs)))
        i1 = int(min(len(ecg), np.ceil((t_rel + post_s) * fs)))
        t = (np.arange(i0, i1) / fs) + (i0 / fs * 0)  # relative axis not needed; use seconds rel window
        t = np.arange(i0, i1) / fs
        ax.plot(t, ecg[i0:i1])

        ax.axvline(t_rel, linestyle="--")
        ax.set_xlabel("Seconds since TDMS start")
        ax.set_ylabel("ECG")

        # Mini-label med window flags
        flags = []
        for c in ["ours_bad", "is_noiseburst", "is_flatline", "is_clipping", "is_seizure_window"]:
            if c in row and pd.notna(row[c]):
                flags.append(f"{c}={row[c]}")
        if "nk_averageQRS" in row and pd.notna(row["nk_averageQRS"]):
            flags.append(f"nk_avgQRS={row['nk_averageQRS']:.3f}")
        if "nk_zhao2018" in row and pd.notna(row["nk_zhao2018"]):
            flags.append(f"nk_zhao={row['nk_zhao2018']}")

        ax.set_title(" | ".join(flags)[:140])

    # sample events
    fps = df_events_ctx[df_events_ctx["event_type"] == "FP"].sample(
        min(n_each, (df_events_ctx["event_type"] == "FP").sum()), random_state=1
    ) if (df_events_ctx["event_type"] == "FP").any() else df_events_ctx.iloc[0:0]

    fns = df_events_ctx[df_events_ctx["event_type"] == "FN"].sample(
        min(n_each, (df_events_ctx["event_type"] == "FN").sum()), random_state=2
    ) if (df_events_ctx["event_type"] == "FN").any() else df_events_ctx.iloc[0:0]

    to_plot = pd.concat([fps.assign(_kind="FP"), fns.assign(_kind="FN")], ignore_index=True)

    if len(to_plot) == 0:
        print("No FP/FN events to plot.")
        return

    n = len(to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, to_plot.iterrows()):
        _plot_one(ax, float(row["t_event_epoch"]), row)
        ax.text(0.01, 0.85, f"{row['_kind']}  |  w={row.get('window_idx', 'NA')}", transform=ax.transAxes)

    fig.suptitle(f"{title_prefix} Peak audit FP/FN examples (n={n})", y=0.995)
    fig.tight_layout()
    plt.show()

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class AuditEvent:
    kind: str          # "FN" eller "FP"
    t_event: float     # epoch-sekunder (reference for FN, test for FP)
    t_lv: Optional[float] = None
    t_py: Optional[float] = None
    dt_s: Optional[float] = None
    window_idx: Optional[int] = None
    ours_bad: Optional[bool] = None
    nk_bad: Optional[bool] = None


def _epoch_to_tdms_sample(t_epoch: float, t0_tdms_epoch: float, fs: float) -> int:
    return int(np.round((t_epoch - t0_tdms_epoch) * fs))


def audit_peak_mismatches(
    *,
    ecg_full: np.ndarray,
    fs: float,
    t0_tdms_epoch: float,
    t_peaks_lv_ov: np.ndarray,
    t_peaks_py_ov: np.ndarray,
    best_delta_s: float,
    delta_lv: float,
    tp_lv_idx: np.ndarray,
    tp_py_idx: np.ndarray,
    fn_lv_idx: np.ndarray,
    fp_py_idx: np.ndarray,
    out_dir: Path,
    uid: str,
    algo_id: str,
    n_events_each: int = 20,
    pad_s: float = 2.0,
    df_win: Optional[pd.DataFrame] = None,  # skal have win_abs_start/win_abs_end/ours_bad/nk_bad
    rng_seed: int = 123,
) -> pd.DataFrame:
    """
    Gemmer PNG-plots omkring FP/FN mismatch-events og returnerer tabel med events.

    Bemærk:
      - t_peaks_py_ov er i epoch, men matcher med best_delta_s (dvs. test+delta vs ref).
      - For FP bruger vi test-peak epoch (uden delta) som event-tidspunkt, men plot markerer delta-korrigeret position.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ecg_full = np.asarray(ecg_full, float).ravel()
    rng = np.random.default_rng(rng_seed)

    events: list[AuditEvent] = []

    # samle FN (ref peaks uden match)
    fn_times = t_peaks_lv_ov[fn_lv_idx] if fn_lv_idx.size else np.array([])
    fp_times = t_peaks_py_ov[fp_py_idx] if fp_py_idx.size else np.array([])

    fn_pick = fn_times if fn_times.size <= n_events_each else rng.choice(fn_times, n_events_each, replace=False)
    fp_pick = fp_times if fp_times.size <= n_events_each else rng.choice(fp_times, n_events_each, replace=False)

    # helper: find nærmeste match (til annotering)
    def nearest(arr: np.ndarray, x: float) -> tuple[Optional[float], Optional[float]]:
        if arr.size == 0:
            return None, None
        j = int(np.argmin(np.abs(arr - x)))
        return float(arr[j]), float(arr[j] - x)

    # optional window lookup
    def attach_window_info(ev: AuditEvent):
        if df_win is None or df_win.empty:
            return
        # forventer epoch timestamps i win_abs_start/end
        m = (df_win["win_abs_start"].astype("int64")/1e9 <= ev.t_event) & (df_win["win_abs_end"].astype("int64")/1e9 >= ev.t_event)
        if not m.any():
            return
        row = df_win.loc[m].iloc[0]
        ev.window_idx = int(row.get("window_idx")) if "window_idx" in row else None
        if "ours_bad" in row:
            ev.ours_bad = bool(row["ours_bad"])
        if "nk_bad" in row:
            ev.nk_bad = bool(row["nk_bad"])

    # plotting
    pad_n = int(np.round(pad_s * fs))

    def plot_event(ev: AuditEvent, k: int):
        s_center = _epoch_to_tdms_sample(ev.t_event, t0_tdms_epoch, fs)
        s0 = max(0, s_center - pad_n)
        s1 = min(ecg_full.size, s_center + pad_n)

        x = (np.arange(s0, s1) / fs)  # sekunder relativt til TDMS start
        y = ecg_full[s0:s1]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(x, y, linewidth=0.8)

        # vertikale markører (lokalt)
        # LV peak (ref) = ev.t_lv ; PY peak (test) = ev.t_py ; test-aligned = ev.t_py + best_delta_s
        # LV (shifted to TDMS/Python axis)
        if ev.t_lv is not None:
            ax.axvline((ev.t_lv + delta_lv - t0_tdms_epoch), linestyle="--", linewidth=1.0)

        # PY (fixed)
        if ev.t_py is not None:
            ax.axvline((ev.t_py - t0_tdms_epoch), linestyle=":", linewidth=1.0)

        title = f"{uid} | {algo_id} | {ev.kind} | t={ev.t_event:.3f}"
        if ev.window_idx is not None:
            title += f" | win={ev.window_idx}"
        if ev.ours_bad is not None:
            title += f" | ours_bad={ev.ours_bad}"
        if ev.nk_bad is not None:
            title += f" | nk_bad={ev.nk_bad}"
        ax.set_title(title)
        ax.set_xlabel("Seconds since TDMS start")
        ax.set_ylabel("ECG")
        ax.grid(True, alpha=0.2)

        fname = f"{uid}__{algo_id}__{ev.kind}__{k:03d}.png"
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)

    # --- FN events (manglende python-peak for LV peak)
    for i, t_lv in enumerate(fn_pick):
        # find nærmeste python (aligned) for reference
        # vi sammenligner mod (py + best_delta)
        t_py_near, dt = nearest(t_peaks_py_ov + best_delta_s, float(t_lv))
        ev = AuditEvent(
            kind="FN",
            t_event=float(t_lv),
            t_lv=float(t_lv),
            t_py=(t_py_near - best_delta_s) if t_py_near is not None else None,
            dt_s=dt,
        )
        attach_window_info(ev)
        events.append(ev)
        plot_event(ev, i)

    # --- FP events (python-peak uden match i LV)
    for i, t_py in enumerate(fp_pick):
        # find nærmeste LV peak til aligned python peak
        t_lv_near, dt = nearest(t_peaks_lv_ov, float(t_py + best_delta_s))
        ev = AuditEvent(
            kind="FP",
            t_event=float(t_py),
            t_lv=t_lv_near,
            t_py=float(t_py),
            dt_s=dt,
        )
        attach_window_info(ev)
        events.append(ev)
        plot_event(ev, i + 1000)

    # return table + gem CSV
    df = pd.DataFrame([e.__dict__ for e in events])
    df.to_csv(out_dir / f"{uid}__{algo_id}__audit_events.csv", index=False)
    return df



# ---------------------------------------------------------
# Samlet process_recording-funktion
# ---------------------------------------------------------


from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path

from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

# def _to_epoch_seconds(dt) -> float:
#     """Robust: datetime -> epoch seconds (float)."""
#     return float(dt.timestamp())

def _cache_path_for_rpeaks(
    cache_dir: Path,
    tdms_path: Path,
    method: str,
    fs: float,
    max_duration_s: Optional[float],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "full" if max_duration_s is None else f"{int(max_duration_s)}s"
    # Gør filnavn stabilt men rimeligt unikt
    stem = tdms_path.stem.replace(" ", "_")
    fname = f"{stem}__{method}__fs{int(round(fs))}__{tag}__rpeaks.npy"
    return cache_dir / fname

# def _load_or_compute_rpeaks(
#     ecg: np.ndarray,
#     fs: float,
#     tdms_path: Path,
#     method: str,
#     rpeak_cache_dir: Optional[Path],
#     max_duration_s: Optional[float],
#     force_recompute: bool,
# ) -> np.ndarray:
#     """Returnerer rpeak-indices (samples) for Python-metoden, evt. fra cache."""
#     if rpeak_cache_dir is None:
#         return detect_rpeaks_python(ecg, fs=fs, method=method)

#     cache_path = _cache_path_for_rpeaks(rpeak_cache_dir, tdms_path, method, fs, max_duration_s)

#     if (not force_recompute) and cache_path.exists():
#         try:
#             arr = np.load(cache_path)
#             return np.asarray(arr, dtype=np.int64)
#         except Exception:
#             # hvis cache er korrupt -> recompute
#             pass

#     r_idx = detect_rpeaks_python(ecg, fs=fs, method=method)
#     np.save(cache_path, np.asarray(r_idx, dtype=np.int64))
#     return np.asarray(r_idx, dtype=np.int64)

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

def process_recording(
    cfg: "RecordingConfig",
    save_aligned_path: Optional[Path] = None,
    delta_range_s: Tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_s: float = 0.04,
    max_duration_s: Optional[float] = None,
    rpeak_cache_dir: Optional[Path] = None,
    force_recompute: bool = False,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Sammenlign LabVIEW RR/peaks mod Python-peaks (NeuroKit metode).
    Nøgleidé:
      - brug LVM header-tid som 'ankerpkt' (efter LabVIEW warmup)
      - crop TDMS til præcis samme tidsvindue som LVM dækker
      - estimer best_delta direkte fra peak trains (robust median dt)
    """
    # -------------------------
    # 1) Load TDMS
    # -------------------------
    sig, meta = load_tdms_from_path(
        cfg.tdms_path,
        channel_hint=getattr(cfg, "channel_hint", "EKG"),
        prefer_tz=getattr(cfg, "tz", "Europe/Copenhagen"),
    )
    ecg_full = np.asarray(sig, dtype=float).ravel()
    fs = float(cfg.fs) if hasattr(cfg, "fs") and cfg.fs is not None else float(meta.fs)

    if meta.start_time is None:
        raise ValueError("TDMS metadata mangler start_time.")
    
    t0_tdms_dt = meta.start_time
    t0_tdms_epoch = _to_epoch_seconds(t0_tdms_dt, tz=getattr(cfg, "tz", "Europe/Copenhagen"))

    # evt. max_duration_s (hurtig testkørsel)
    if max_duration_s is not None:
        nmax = int(round(max_duration_s * fs))
        ecg_full = ecg_full[:nmax]

    # -------------------------
    # 2) Load LabVIEW RR + header tid
    # -------------------------
    rr_lv = np.asarray(read_labview_rr(str(cfg.lvm_path)), float).ravel()
    if rr_lv.size == 0:
        raise ValueError("LabVIEW RR er tom.")
    
    t0_lv_dt = read_header_datetime_lvm(str(cfg.lvm_path))
    if t0_lv_dt is None:
        raise ValueError(f"Kunne ikke læse LabVIEW header datetime: {cfg.lvm_path}")

    t0_lv_epoch = _to_epoch_seconds(t0_lv_dt, tz=getattr(cfg, "tz", "Europe/Copenhagen"))

    # LabVIEW peak tider (epoch): første peak ved header-tid
    t_peaks_lv = t0_lv_epoch + np.cumsum(np.insert(rr_lv, 0, 0.0))

    # -------- 3) Definér LVM vindue relativt til TDMS og crop TDMS --------
    offset0_s = t0_lv_epoch - t0_tdms_epoch  # typisk ~17.15s (warmup+filter)
    lv_win_start_rel_s = offset0_s
    lv_win_end_rel_s = offset0_s + float(np.sum(rr_lv))  # ca sidste peak-tid

    crop_start = int(np.round(lv_win_start_rel_s * fs))
    crop_end = int(np.round(lv_win_end_rel_s * fs))

    crop_start = max(0, crop_start)
    crop_end = min(len(ecg_full), crop_end)

    if crop_end - crop_start < int(5 * fs):
        raise ValueError("Efter crop er signalet for kort til meningsfuld sammenligning.")

    # epoch tid for crop-start (TDMS)
    t0_crop_epoch = t0_tdms_epoch + (crop_start / fs)

    if debug:
        print(f"[DEBUG] TDMS start dt   : {t0_tdms_dt} (epoch={t0_tdms_epoch:.6f})")
        print(f"[DEBUG] LVM  header dt : {t0_lv_dt} (epoch={t0_lv_epoch:.6f})")
        print(f"[DEBUG] offset0_s (LVM header - TDMS start): {offset0_s:.6f} s")
        print(f"[DEBUG] LVM window start/end (rel. TDMS): {lv_win_start_rel_s:.3f}s  -> {lv_win_end_rel_s:.3f}s")
        print(f"[DEBUG] TDMS crop idx start/end: {crop_start} -> {crop_end}  (len(ecg_full)={len(ecg_full)})")

    # -------- 4) Python peaks (cache på fuld TDMS) + crop til LVM vindue --------
    recording_uid = getattr(cfg, "recording_uid", None)  # hvis du senere tilføjer den til cfg
    r_idx_py_full = get_or_compute_rpeaks_idx(
        ecg=ecg_full,
        fs=fs,
        cfg=cfg,
        max_duration_s=max_duration_s,
        cache_dir=rpeak_cache_dir,
        force_recompute=force_recompute,
        recording_uid=recording_uid,
    )

    # vælg de peaks der ligger indenfor crop-vinduet (ORIG tdms indices)
    m = (r_idx_py_full >= crop_start) & (r_idx_py_full < crop_end)
    r_idx_py = (r_idx_py_full[m] - crop_start).astype(np.int64)  # lokale indices i crop
    t_peaks_py = t0_crop_epoch + (r_idx_py.astype(float) / fs)

    if debug:
        print(f"[DEBUG] #py_peaks full={int(r_idx_py_full.size)}  cropped={int(r_idx_py.size)}  #lv_peaks={int(t_peaks_lv.size)}")

    if t_peaks_py.size == 0:
        raise ValueError("Ingen Python-peaks i LVM-vinduet efter crop.")


    # -------- 5) overlap (fair sammenligning) --------
    # NB: her er der endnu IKKE tilføjet best_delta
    overlap_start = max(float(t_peaks_lv[0]), float(t_peaks_py[0]))
    overlap_end = min(float(t_peaks_lv[-1]), float(t_peaks_py[-1]))
    if overlap_end <= overlap_start:
        raise ValueError("Ingen overlappende tidsinterval LV vs PY efter crop.")

    lv_ov = t_peaks_lv[(t_peaks_lv >= overlap_start) & (t_peaks_lv <= overlap_end)]
    py_ov = t_peaks_py[(t_peaks_py >= overlap_start) & (t_peaks_py <= overlap_end)]

    ## Extra check - Sanity check
    def _refine_delta_lv_by_matched_median(
        t_lv: np.ndarray,
        t_py: np.ndarray,
        delta_lv: float,
        tol_s: float = 0.04,
        min_tp: int = 200,
        max_adjust_s: float = 0.2,
    ) -> tuple[float, dict]:
        """
        Forfin delta_lv ved at bruge median signed dt på de matchede peaks.
        Flytter KUN LabVIEW (delta_lv), aldrig Python.

        Returnerer (delta_lv_refined, debug_info)
        """
        tp_lv_idx, tp_py_idx, fn_lv_idx, fp_py_idx = match_rpeaks_time_based_global(
            t_peaks_ref=t_lv,
            t_peaks_test=t_py,
            delta_s=delta_lv,
            tol_s=tol_s,
            shift_ref=True,
        )

        info = {
            "tp": int(len(tp_lv_idx)),
            "fp": int(len(fp_py_idx)),
            "fn": int(len(fn_lv_idx)),
            "median_dt_ms": np.nan,
            "p95_abs_dt_ms": np.nan,
            "delta_lv_before": float(delta_lv),
            "delta_lv_after": float(delta_lv),
        }

        if len(tp_lv_idx) < min_tp:
            return delta_lv, info

        dt = t_py[tp_py_idx] - (t_lv[tp_lv_idx] + delta_lv)  # signed seconds
        med = float(np.median(dt))
        info["median_dt_ms"] = med * 1e3
        info["p95_abs_dt_ms"] = float(np.percentile(np.abs(dt), 95)) * 1e3

        # undgå voldsomme “justeringer” pga. et skævt match
        med = float(np.clip(med, -max_adjust_s, max_adjust_s))

        delta_lv_ref = delta_lv + med
        info["delta_lv_after"] = float(delta_lv_ref)

        return delta_lv_ref, info



    # -------- 6) estimer lille delta fra peaks --------
    best_delta = _estimate_delta_from_peaks(lv_ov, py_ov, max_abs_dt_s=1.0)

    # ... du har allerede estimeret best_delta og sat delta_lv ...
    delta_lv = -best_delta

    # 2C: ultrahurtig “lag check” / refinement (kun for Hamilton hvis du vil)
    if str(cfg.algo_id) == "hamilton2002":
        delta_lv_ref, info = _refine_delta_lv_by_matched_median(
            t_lv=lv_ov,
            t_py=py_ov,
            delta_lv=delta_lv,
            tol_s=0.04,
            min_tp=500,  # justér evt
        )
        if debug:
            print(f"[DEBUG] Hamilton refine: median_dt={info['median_dt_ms']:.3f} ms "
                f"(p95|dt|={info['p95_abs_dt_ms']:.3f} ms) "
                f"delta_lv {info['delta_lv_before']:.6f} -> {info['delta_lv_after']:.6f}")

        delta_lv = delta_lv_ref  # <-- brug den refined fremover

    # match igen (nu med refined delta_lv)
    tp_lv_idx, tp_py_idx, fn_lv_idx, fp_py_idx = match_rpeaks_time_based_global(
        t_peaks_ref=lv_ov,
        t_peaks_test=py_ov,
        delta_s=delta_lv,
        tol_s=0.04,
        shift_ref=True,
    )

    # dt for matched peaks (PY fixed, LV shifted by delta_lv)
    if len(tp_lv_idx) > 0:
        dt_ms = (py_ov[tp_py_idx] - (lv_ov[tp_lv_idx] + delta_lv)) * 1e3
        median_dt_ms = float(np.median(dt_ms))
        p95_abs_dt_ms = float(np.percentile(np.abs(dt_ms), 95))
    else:
        median_dt_ms = np.nan
        p95_abs_dt_ms = np.nan


    if debug:
        print(f"[DEBUG] best_delta_s (py->lv): {best_delta:.6f}")
        print(f"[DEBUG] delta_lv_s  (lv->py): {delta_lv:.6f}")
        print(f"[DEBUG] overlap_start/end epoch: {overlap_start:.3f} -> {overlap_end:.3f}")
        print(f"[DEBUG] matched peaks: {len(tp_lv_idx)}")
        if len(tp_lv_idx) > 0:
            dt_ms = (py_ov[tp_py_idx] - (lv_ov[tp_lv_idx] + delta_lv)) * 1e3
            print(f"[DEBUG] median |dt| (ms): {np.median(np.abs(dt_ms)):.3f}")
            print(f"[DEBUG] p95 |dt| (ms): {np.percentile(np.abs(dt_ms), 95):.3f}")

    if debug and (len(fn_lv_idx) + len(fp_py_idx) > 0):
        audit_dir = Path(r"E:\Speciale - Results\study3_audit") / f"{cfg.patient_id:02d}_{cfg.recording_id:02d}"
        uid = getattr(cfg, "recording_uid", f"p{cfg.patient_id:02d}_r{cfg.recording_id:02d}")

        # Hvis du har df_win for denne recording, så giv den med.
        # Ellers: df_win=None (audit virker stadig).
        audit_peak_mismatches(
            ecg_full=ecg_full,   # fuld TDMS (ikke kun cropped)
            fs=fs,
            t0_tdms_epoch=t0_tdms_epoch,
            t_peaks_lv_ov=lv_ov,
            t_peaks_py_ov=py_ov,
            best_delta_s=best_delta,
            delta_lv=delta_lv,
            tp_lv_idx=tp_lv_idx,
            tp_py_idx=tp_py_idx,
            fn_lv_idx=fn_lv_idx,
            fp_py_idx=fp_py_idx,
            out_dir=audit_dir,
            uid=uid,
            algo_id=str(cfg.algo_id),
            n_events_each=20,
            pad_s=2.0,
            df_win=None,  # <-- kan du koble på senere
        )

    # -------- 6b) first-R diagnostics (Python stays fixed, LabVIEW shifts) --------
    tdms_start_epoch = float(t0_tdms_epoch)

    lv_first_r_epoch_raw = float(t_peaks_lv[0]) if t_peaks_lv.size else np.nan
    lv_first_r_epoch = float(t_peaks_lv[0] + delta_lv) if t_peaks_lv.size else np.nan  # <-- shifted LV

    py_first_r_epoch = float(t_peaks_py[0]) if t_peaks_py.size else np.nan             # <-- NOT shifted

    lv_first_r_rel_s = lv_first_r_epoch - tdms_start_epoch if np.isfinite(lv_first_r_epoch) else np.nan
    py_first_r_rel_s = py_first_r_epoch - tdms_start_epoch if np.isfinite(py_first_r_epoch) else np.nan
    first_r_rel_diff_s = py_first_r_rel_s - lv_first_r_rel_s if (
        np.isfinite(py_first_r_rel_s) and np.isfinite(lv_first_r_rel_s)
    ) else np.nan



    peak_metrics = compute_peak_metrics(tp=len(tp_lv_idx), fp=len(fp_py_idx), fn=len(fn_lv_idx))

    # -------- 7) RR metrics --------
    lv_ov_shift = lv_ov + delta_lv
    py_ov_fix = py_ov

    rr_lv_ov = np.diff(lv_ov_shift)     # samme som diff(lv_ov)
    rr_py_ov = np.diff(py_ov_fix)

    best_delta_rr, rr_lv_m, rr_py_m = find_best_delta(
        t_R_lv=lv_ov_shift[:-1],
        RR_lv=rr_lv_ov,
        t_R_py=py_ov_fix[:-1],
        RR_py=rr_py_ov,
        delta_range_s=delta_range_s,
        delta_step_s=delta_step_s,
        tol_s=tol_s,
    )

    rr_metrics = compute_rr_metrics(rr_ref=rr_lv_m, rr_test=rr_py_m)

    # -------- 8) output --------
    out: Dict[str, float] = {}
    out.update(rr_metrics)
    out.update(
        {
            "patient_id": int(cfg.patient_id),
            "recording_id": int(cfg.recording_id),
            "algo_id": str(cfg.algo_id),
            "trim_label": getattr(cfg, "trim_label", None),

            "offset0_s": float(offset0_s),
            "best_delta_s": float(best_delta),
            "best_delta_rr_s": float(best_delta_rr),

            "overlap_start": float(overlap_start),
            "overlap_end": float(overlap_end),

            "n_peaks_labview_total": int(lv_ov.size),
            "n_peaks_python_total": int(py_ov.size),

            "n_peaks_tp": int(peak_metrics["tp"]),
            "n_peaks_fp": int(peak_metrics["fp"]),
            "n_peaks_fn": int(peak_metrics["fn"]),
            "peak_sens": float(peak_metrics["sens"]),
            "peak_ppv": float(peak_metrics["ppv"]),
            "peak_f1": float(peak_metrics["f1"]),

            "raw_tdms_path": str(cfg.tdms_path),
            "raw_lvm_path": str(cfg.lvm_path),

            "tdms_start_epoch": tdms_start_epoch,

            "labview_first_r_epoch_raw": lv_first_r_epoch_raw,
            "labview_first_r_epoch": lv_first_r_epoch,     # shifted
            "python_first_r_epoch": py_first_r_epoch,       # unshifted

            "labview_first_r_rel_s": lv_first_r_rel_s,
            "python_first_r_rel_s": py_first_r_rel_s,
            "first_r_rel_diff_s": first_r_rel_diff_s,

            "median_dt_ms": median_dt_ms,
            "p95_abs_dt_ms": p95_abs_dt_ms,
        }
    )

    # -------- 9) save aligned RR (valgfrit) --------
    if save_aligned_path is not None:
        save_aligned_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"RR_labview_s": rr_lv_m, "RR_python_s": rr_py_m}).to_csv(save_aligned_path, index=False)

    return out




def plot_rr_alignment(t_R_lv, RR_lv, t_R_py, RR_py, delta_s, tol_s=0.04):
    """
    Visualiserer forskellen mellem LabVIEW og Python RR over tid.

    Lav to plots:
      1) RR over tid (sek)
      2) forskel i RR (ms)
    """
    # Match én gang (samme som i process_recording)
    rr_lv_m, rr_py_m = match_rr_series_time_based(
        t_R_lv,
        RR_lv,
        t_R_py,
        RR_py,
        delta_s=delta_s,
        tol_s=tol_s,
    )
    
    # Lav tidsakse (brug LabVIEW's R-peak tider)
    t = t_R_lv[:len(rr_lv_m)]  # tidsakse for matched intervaller

    diff_ms = (rr_py_m - rr_lv_m) * 1000.0

    fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axs[0].plot(t, rr_py_m, label="Python RR (s)", alpha=0.7)
    axs[0].plot(t, rr_lv_m, label="LabVIEW RR (s)")
    axs[0].legend()
    axs[0].set_ylabel("RR (s)")
    axs[0].set_title("RR comparison over time")

    axs[1].plot(t, diff_ms)
    axs[1].set_ylabel("Difference (ms)")
    axs[1].set_xlabel("Time (epoch seconds)")
    axs[1].set_title("RR difference (Python - LabVIEW)")

    plt.tight_layout()
    return fig

import numpy as np


def match_rpeaks_time_based_global(
    t_peaks_ref: np.ndarray,
    t_peaks_test: np.ndarray,
    delta_s: float,
    tol_s: float = 0.04,
    shift_ref: bool = False,
):
    """
    Global peak alignment.

    - t_peaks_ref  : reference peak times (seconds since epoch)
    - t_peaks_test : test peak times (seconds since epoch)
    - delta_s      : time shift applied to either test (default) or ref (if shift_ref=True)
    - tol_s        : max difference for peaks to be considered a match

    Returns:
      tp_idx_ref, tp_idx_test, fn_idx_ref, fp_idx_test
      (indices refer to ORIGINAL input arrays)
    """
    t_peaks_ref = np.asarray(t_peaks_ref, dtype=float)
    t_peaks_test = np.asarray(t_peaks_test, dtype=float)

    # Apply delta to the chosen series
    if shift_ref:
        t_ref = t_peaks_ref + float(delta_s)
        t_test = t_peaks_test
    else:
        t_ref = t_peaks_ref
        t_test = t_peaks_test + float(delta_s)

    n_ref = t_ref.size
    n_test = t_test.size

    if n_ref == 0 or n_test == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.arange(n_ref, dtype=int),
            np.arange(n_test, dtype=int),
        )

    # Two-pointer matching on sorted times
    i = j = 0
    tp_ref = []
    tp_test = []
    fn_ref = []
    fp_test = []

    while i < n_ref and j < n_test:
        dt = t_test[j] - t_ref[i]
        if abs(dt) <= tol_s:
            tp_ref.append(i)
            tp_test.append(j)
            i += 1
            j += 1
        elif dt < -tol_s:
            fp_test.append(j)
            j += 1
        else:
            fn_ref.append(i)
            i += 1

    # leftovers
    if i < n_ref:
        fn_ref.extend(range(i, n_ref))
    if j < n_test:
        fp_test.extend(range(j, n_test))

    return (
        np.asarray(tp_ref, dtype=int),
        np.asarray(tp_test, dtype=int),
        np.asarray(fn_ref, dtype=int),
        np.asarray(fp_test, dtype=int),
    )




def compute_peak_metrics(tp: int, fp: int, fn: int):
    """
    tp, fp, fn: heltal
    Returnerer dict med sensitivity, PPV, F1 osv.
    """
    tp = int(tp)
    fp = int(fp)
    fn = int(fn)

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # recall / sensitivity
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan   # precision / PPV
    f1 = 2 * sens * ppv / (sens + ppv) if np.isfinite(sens) and np.isfinite(ppv) and (sens + ppv) > 0 else np.nan

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sens": sens,
        "ppv": ppv,
        "f1": f1,
    }


def plot_alignment_summary(
    t_R_lv: np.ndarray,
    RR_lv: np.ndarray,
    t_R_py: np.ndarray,
    RR_py: np.ndarray,
    best_delta_s: float,
    tol_s: float = 0.04,
    big_thresh_ms: float = 50.0,
    title_suffix: str = "",
):
    """
    Lav et samlet alignment-resume:
      - RR over tid (LabVIEW vs Python)
      - RR-difference (Python - LabVIEW) i ms
      - scatter over hvor |diff| > big_thresh_ms

    t_R_lv, RR_lv : LabVIEW RR-starttider og intervaller (sek / sek)
    t_R_py, RR_py : Python peak-tider og RR-intervaller (sek / sek)
    best_delta_s  : tidsforskydning lagt til Python (fra find_best_delta)
    """
    # Match én gang (samme som i process_recording)
    rr_lv_m, rr_py_m = match_rr_series_time_based(
        t_R_lv,
        RR_lv,
        t_R_py,
        RR_py,
        delta_s=best_delta_s,
        tol_s=tol_s,
    )

    if rr_lv_m.size == 0:
        raise ValueError("No matched RR intervals – check inputs.")

    # Tidsakse (relative minutter fra LV første RR)
    t = t_R_lv[: len(rr_lv_m)]
    t_rel_min = (t - t[0]) / 60.0

    diff_ms = (rr_py_m - rr_lv_m) * 1000.0
    big_err_mask = np.abs(diff_ms) > big_thresh_ms

    fig = plt.figure(figsize=(16, 10))

    # 1) RR over tid
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_rel_min, rr_py_m, label="Python RR (s)", alpha=0.7)
    ax1.plot(t_rel_min, rr_lv_m, label="LabVIEW RR (s)")
    ax1.set_ylabel("RR (s)")
    ax1.set_title(f"RR over time (relative minutes){title_suffix}")
    ax1.legend(loc="upper right")

    # 2) RR difference
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t_rel_min, diff_ms)
    ax2.axhline(0, linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Difference (ms)")
    ax2.set_title("RR difference (Python - LabVIEW)")

    # 3) Hvor er |diff| > threshold
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.scatter(t_rel_min, diff_ms, s=2, label="All", alpha=0.3)
    ax3.scatter(
        t_rel_min[big_err_mask],
        diff_ms[big_err_mask],
        s=5,
        label=f"|diff| > {big_thresh_ms:.0f} ms",
    )
    ax3.axhline(0, linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Time (min from LabVIEW first RR)")
    ax3.set_ylabel("Diff (ms)")
    ax3.set_title("Where do we disagree more than threshold?")
    ax3.legend(loc="upper right")

    plt.tight_layout()
    return fig


def load_index_csv(index_path: Path) -> List[RecordingConfig]:
    """
    Læser et index-CSV med kolonner:
      patient_id, recording_id, tdms_path, lvm_path, fs

    og returnerer en liste af RecordingConfig.
    """
    df = pd.read_csv(index_path)

    configs: List[RecordingConfig] = []
    for _, row in df.iterrows():
        cfg = RecordingConfig(
            patient_id=int(row["patient_id"]),
            recording_id=int(row["recording_id"]),
            tdms_path=Path(row["tdms_path"]),
            lvm_path=Path(row["lvm_path"]),
            fs=float(row.get("fs", 512.0)),
            # channel_hint og tz kan evt. overrides i CSV senere
        )
        configs.append(cfg)

    return configs


def run_rr_comparison(
    index_csv: Path,
    out_metrics_csv: Path,
    aligned_dir: Optional[Path] = None,
    patient_filter: Optional[List[int]] = None,
    delta_range_s: tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_s: float = 0.04,
) -> pd.DataFrame:
    """
    Kører RR- og peak-sammenligning for alle optagelser i index_csv.

    - index_csv     : CSV med patient_id, recording_id, tdms_path, lvm_path, fs
    - out_metrics_csv : hvor den samlede metrics-tabel gemmes
    - aligned_dir   : hvis angivet, gemmes én aligned RR-fil per optagelse her
    - patient_filter: hvis angivet, kun disse patient_id'er (liste af ints)
    """
    configs = load_index_csv(index_csv)

    if patient_filter is not None:
        configs = [c for c in configs if c.patient_id in patient_filter]

    rows = []
    if aligned_dir is not None:
        aligned_dir = Path(aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

    for cfg in tqdm(configs, desc="RR/peak comparison"):
        # vælg filnavn for aligned RR (hvis ønsket)
        save_path = None
        if aligned_dir is not None:
            save_path = aligned_dir / f"p{cfg.patient_id:02d}_r{cfg.recording_id:02d}_rr_aligned.csv"

        metrics = process_recording(
            cfg,
            save_aligned_path=save_path,
            delta_range_s=delta_range_s,
            delta_step_s=delta_step_s,
            tol_s=tol_s,
        )
        rows.append(metrics)

    df_metrics = pd.DataFrame(rows)
    out_metrics_csv = Path(out_metrics_csv)
    out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_metrics_csv, index=False)

    return df_metrics


def run_rr_comparison_from_df(
    df_index: pd.DataFrame,
    methods: list[str],
    aligned_dir: Path | None = None,
    patient_filter: list[int] | None = None,
    delta_range_s: tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_s: float = 0.04,
    max_duration_s: float | None = None,  # NEW
    rpeak_cache_dir: Path | None = None,
    force_recompute: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """
    df_index forventes at have mindst:
      - patient_id
      - recording_id
      - tdms_path
      - lvm_file
      - fs
    Valgfrit:
      - recording_uid
      - trim_label   (fx "no_trim", "trim_table", "trim_30min")

    methods:
      - liste over NeuroKit-peak-metoder, fx
        ["neurokit", "pantompkins1985", "hamilton2002"]
    """
    df = df_index.copy()
    if patient_filter is not None:
        df = df[df["patient_id"].isin(patient_filter)]

    if aligned_dir is not None:
        aligned_dir = Path(aligned_dir)
        aligned_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    total_iters = len(df) * len(methods)

    with tqdm(total=total_iters, desc=f"RR/peak comparison for:") as pbar:
        for _, row in df.iterrows():
            trim_label = row["trim_label"] if "trim_label" in row else None

            for method in methods:
                cfg = RecordingConfig(
                    patient_id=int(row["patient_id"]),
                    recording_id=int(row["recording_id"]),
                    tdms_path=Path(row["tdms_path"]),
                    lvm_path=Path(row["lvm_file"]),
                    fs=float(row["fs"]),
                    algo_id=method,
                    trim_label=trim_label,
                )

                save_path = None
                if aligned_dir is not None:
                    if "recording_uid" in row:
                        base = f"{row['recording_uid']}"
                    else:
                        base = f"p{cfg.patient_id:02d}_r{cfg.recording_id:02d}"
                    suffix_trim = f"_{trim_label}" if trim_label is not None else ""
                    fname = f"{base}_{method}{suffix_trim}_rr_aligned.csv"
                    save_path = aligned_dir / fname

                metrics = process_recording(
                    cfg,
                    save_aligned_path=save_path,
                    delta_range_s=delta_range_s,
                    delta_step_s=delta_step_s,
                    tol_s=tol_s,
                    max_duration_s=max_duration_s,
                    rpeak_cache_dir=rpeak_cache_dir,
                    force_recompute=force_recompute,
                    debug=debug,
                )


                rows.append(metrics)
                pbar.update(1)

    df_metrics = pd.DataFrame(rows)
    return df_metrics


def load_labview_peaks(lvm_path: str | Path):
    """
    Læs en LabVIEW-LVM-fil og returnér:
      - t_peaks : absolutte R-peak tider (sekunder siden epoch)
      - RR      : RR-intervaller i sekunder
    """
    lvm_path = Path(lvm_path)
    rr_lv = read_labview_rr(str(lvm_path))
    t0_lv_dt = read_header_datetime_lvm(str(lvm_path))
    if t0_lv_dt is None:
        raise ValueError(f"Kunne ikke læse header-timestamp fra {lvm_path}")

    # Vi genbruger reconstruct_labview_peak_times fra rr_compare.py
    # from src.hrv_epatch.study3.rr_compare import reconstruct_labview_peak_times

    t_peaks_lv, RR_lv = reconstruct_labview_peak_times(rr_lv, t0_lv_dt)
    return t_peaks_lv, RR_lv


def compare_labview_versions(
    lvm_ref_path: str | Path,
    lvm_test_path: str | Path,
    ref_label: str = "ref",
    test_label: str = "test",
    tol_peak_s: float = 0.04,   # 40 ms peak tolerance
    delta_range_s: tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_rr_s: float = 0.15,     # 150 ms RR-tolerance (til matching)
) -> Dict[str, float]:
    """
    Sammenlign to LabVIEW-R-peak serier (to forskellige LVM-filer)
    for samme optagelse.

    lvm_ref_path  : fx "uden trim" version
    lvm_test_path : fx "trim_table" eller "trim_30min"

    ref_label/test_label bruges kun som tekst i output, så du kan skelne.
    """
    # 1) Load peaks + RR for begge
    t_ref, rr_ref = load_labview_peaks(lvm_ref_path)
    t_test, rr_test = load_labview_peaks(lvm_test_path)

    # 2) RR-baseret alignment: find lille delta der giver bedst RR-match
    #    (bare for at korrigere evt. global timestamp-offset mellem LVM-filerne)
    best_delta, rr_ref_m, rr_test_m = find_best_delta(
        t_R_lv=t_ref[:-1],    # starttid for RR-interval = peak i
        RR_lv=rr_ref,
        t_R_py=t_test[:-1],   # her bruger vi 'py'-navnet som "test", men begge er LV
        RR_py=rr_test,
        delta_range_s=delta_range_s,
        delta_step_s=delta_step_s,
        tol_s=tol_rr_s,
    )

    rr_metrics = compute_rr_metrics(rr_ref=rr_ref_m, rr_test=rr_test_m)

    # 3) Peak-level matching indenfor ref-vinduet, med best_delta
    tp_ref_idx, tp_test_idx, fn_ref_idx, fp_test_idx = match_rpeaks_time_based_global(
        t_peaks_ref=t_ref,
        t_peaks_test=t_test,
        delta_s=best_delta,
        tol_s=tol_peak_s,
    )


    peak_metrics = compute_peak_metrics(
        tp=len(tp_ref_idx),
        fp=len(fp_test_idx),
        fn=len(fn_ref_idx),
    )

    # 4) Pak det hele i et samlet dict
    out: Dict[str, float] = {}
    out.update(rr_metrics)
    out.update(
        {
            "ref_label": ref_label,
            "test_label": test_label,
            "n_peaks_ref_total": int(t_ref.size),
            "n_peaks_test_total": int(t_test.size),
            "n_peaks_tp": int(peak_metrics["tp"]),
            "n_peaks_fp": int(peak_metrics["fp"]),
            "n_peaks_fn": int(peak_metrics["fn"]),
            "peak_sens": float(peak_metrics["sens"]),
            "peak_ppv": float(peak_metrics["ppv"]),
            "peak_f1": float(peak_metrics["f1"]),
            "best_delta_s": float(best_delta),
            "lvm_ref_path": str(lvm_ref_path),
            "lvm_test_path": str(lvm_test_path),
        }
    )
    return out