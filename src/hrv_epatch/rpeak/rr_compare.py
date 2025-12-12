
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


def rpeaks_to_times_and_rr(
    r_idx: np.ndarray,
    fs: float,
    t0_tdms_dt,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Konverterer R-peak sampleindeks til absolutte tidspunkter (sekunder siden epoch)
    og RR-intervaller i sekunder.

    r_idx: R-peak sampleindeks (0-baseret)
    fs: sampling frequency
    t0_tdms_dt: datetime (timezone-aware eller naive) for TDMS start
    """
    r_idx = np.asarray(r_idx, dtype=float)
    if r_idx.size == 0:
        return np.array([]), np.array([])

    t0_epoch = t0_tdms_dt.timestamp()
    t_R_py = t0_epoch + r_idx / fs

    if t_R_py.size < 2:
        return t_R_py, np.array([])

    RR_py = np.diff(t_R_py)
    return t_R_py, RR_py


def reconstruct_labview_peak_times(rr_lv: np.ndarray, t0_lv_dt):
    """
    rr_lv : RR-intervaller i sekunder (LabVIEW)
    t0_lv_dt : datetime for første R-peak i LabVIEW-sekvensen (fra header)

    Returnerer:
      t_peaks_lv : absolutte R-peak tidspunkter (sekunder siden epoch), længde = len(rr_lv) + 1
      RR_lv      : uændret RR-array
    """
    rr_lv = np.asarray(rr_lv, dtype=float)
    if rr_lv.size == 0:
        return np.array([]), rr_lv

    t0_epoch = t0_lv_dt.timestamp()

    # R-peak tider: t0, t0 + rr0, t0 + rr0 + rr1, ...
    t_peaks = [t0_epoch]
    for rr in rr_lv:
        t_peaks.append(t_peaks[-1] + rr)
    t_peaks = np.asarray(t_peaks)

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
    tol_s: float = 0.15,
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
    tol_s: float = 0.15,
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


# ---------------------------------------------------------
# Samlet process_recording-funktion
# ---------------------------------------------------------


from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path

def process_recording(
    cfg: RecordingConfig,
    save_aligned_path: Optional[Path] = None,
    delta_range_s: Tuple[float, float] = (-2.0, 2.0),
    delta_step_s: float = 0.05,
    tol_s: float = 0.15,
    max_duration_s: float | None = None,  # NEW: begræns længden
    rpeak_cache_dir: Path | None = None,
    force_recompute: bool = False,
) -> Dict[str, float]:
    """
    Kører hele RR-sammenligningen for én optagelse.

    Trin:
    1) Load u-trimmet TDMS og hent EKG + starttid.
    2) Kør Python-R-peak detektion på hele signalet (eller første max_duration_s).
    3) Læs LabVIEW-RR + første R-peak timestamp fra LVM.
    4) Konvertér begge til absolut tid + RR.
    5) Find bedste tids-offset (delta) via simpel grid search.
    6) Beregn metrics (MAE, RMSE, korrelation, HR-diff).
    7) Gem evt. aligned RR-serier til CSV.

    Hvis max_duration_s er sat (fx 3*3600), bruges kun de første
    max_duration_s af TDMS-signalet til Python-R-peaks.
    """
    # ---------------------------------------------
    # 1) TDMS: load signal + metadata
    # ---------------------------------------------
    sig, meta = load_tdms_from_path(
        cfg.tdms_path,
        channel_hint=cfg.channel_hint,
        prefer_tz=cfg.tz,
    )

    fs = float(cfg.fs)

    # antag EKG i første kolonne
    ecg_full = np.asarray(sig)[:]

    # NEW: begræns længde hvis ønsket
    if max_duration_s is not None:
        n_max = int(max_duration_s * fs)
        ecg = ecg_full[:n_max]
    else:
        ecg = ecg_full

    # TDMS starttid som datetime
    t0_tdms_dt = meta.start_time
    if t0_tdms_dt is None:
        raise ValueError("TDMS metadata indeholder ikke 'start_time'.")

    # ---------------------------------------------
    # 2) Python R-peak detektion (med cache)
    r_idx_py = get_or_compute_rpeaks_idx(
        ecg=ecg,
        fs=fs,
        cfg=cfg,
        max_duration_s=max_duration_s,
        cache_dir=rpeak_cache_dir,      # ny parameter til process_recording
        force_recompute=force_recompute # ny parameter til process_recording
    )
    t_R_py, RR_py = rpeaks_to_times_and_rr(r_idx_py, fs=fs, t0_tdms_dt=t0_tdms_dt)


    # ---------------------------------------------
    # 3) LabVIEW: RR + første R-peak timestamp
    # ---------------------------------------------
    rr_lv = read_labview_rr(str(cfg.lvm_path))
    t0_lv_dt = read_header_datetime_lvm(str(cfg.lvm_path))
    if t0_lv_dt is None:
        raise ValueError("Kunne ikke læse LabVIEW starttid (header datetime er None).")

    t_R_lv, RR_lv = reconstruct_labview_r_times(rr_lv, t0_lv_dt=t0_lv_dt)

    # ---------------------------------------------
    # 4) Finjustér alignment: find bedste delta
    # ---------------------------------------------
    best_delta, rr_lv_m, rr_py_m = find_best_delta(
        t_R_lv=t_R_lv,
        RR_lv=RR_lv,
        t_R_py=t_R_py,
        RR_py=RR_py,
        delta_range_s=delta_range_s,
        delta_step_s=delta_step_s,
        tol_s=tol_s,
    )

    # ---------------------------------------------
    # 5) Beregn RR-metrics
    # ---------------------------------------------
    metrics = compute_rr_metrics(rr_ref=rr_lv_m, rr_test=rr_py_m)

    # ---------------------------------------------
    # 5b) Peak-level matching og metrics (TP/FP/FN)
    # ---------------------------------------------
    # 1) LabVIEW peak-tider (sekunder siden epoch)
    t_peaks_lv, _ = reconstruct_labview_peak_times(rr_lv, t0_lv_dt)

    # 2) Python peak-tider (t_R_py er allerede peaks) – brug samme best_delta
    t_peaks_py = t_R_py  # rename for klarhed

    tp_lv_idx, tp_py_idx, fn_lv_idx, fp_py_idx = match_rpeaks_time_based_global(
        t_peaks_ref=t_peaks_lv,
        t_peaks_test=t_peaks_py,
        delta_s=best_delta,
        tol_s=0.04,
    )

    # ---------------------------------------------
    # 5c) Relative tider for første R-peak (til TDMS-start)
    # ---------------------------------------------
    tdms_start_epoch = float(t0_tdms_dt.timestamp())
    lv_first_r_epoch = float(t_peaks_lv[0]) if t_peaks_lv.size > 0 else np.nan
    py_first_r_epoch = float(t_peaks_py[0] + best_delta) if t_peaks_py.size > 0 else np.nan

    lv_first_r_rel_s = lv_first_r_epoch - tdms_start_epoch if np.isfinite(lv_first_r_epoch) else np.nan
    py_first_r_rel_s = py_first_r_epoch - tdms_start_epoch if np.isfinite(py_first_r_epoch) else np.nan
    first_r_rel_diff_s = py_first_r_rel_s - lv_first_r_rel_s if (
        np.isfinite(py_first_r_rel_s) and np.isfinite(lv_first_r_rel_s)
    ) else np.nan

    peak_metrics = compute_peak_metrics(
        tp=len(tp_lv_idx),
        fp=len(fp_py_idx),
        fn=len(fn_lv_idx),
    )

    # ---------------------------------------------
    # 6) Saml alle metrics + metadata
    # ---------------------------------------------
    metrics.update(
        {
            "patient_id": cfg.patient_id,
            "recording_id": cfg.recording_id,
            "algo_id": cfg.algo_id,
            "trim_label": cfg.trim_label,
            "n_rr_labview_total": int(RR_lv.size),
            "n_rr_python_total": int(RR_py.size),
            "n_rr_matched": int(metrics["n_common"]),
            "best_delta_s": float(best_delta),
            "tdms_start_epoch": tdms_start_epoch,
            "labview_first_r_epoch": lv_first_r_epoch,
            "python_first_r_epoch": py_first_r_epoch,
            "labview_first_r_rel_s": lv_first_r_rel_s,
            "python_first_r_rel_s": py_first_r_rel_s,
            "first_r_rel_diff_s": first_r_rel_diff_s,
            "raw_tdms_path": str(cfg.tdms_path),
            "raw_lvm_path": str(cfg.lvm_path),
            "n_peaks_labview_total": int(t_peaks_lv.size),
            "n_peaks_python_total": int(t_peaks_py.size),
            "n_peaks_tp": int(peak_metrics["tp"]),
            "n_peaks_fp": int(peak_metrics["fp"]),
            "n_peaks_fn": int(peak_metrics["fn"]),
            "peak_sens": float(peak_metrics["sens"]),
            "peak_ppv": float(peak_metrics["ppv"]),
            "peak_f1": float(peak_metrics["f1"]),
        }
    )

    # ---------------------------------------------
    # 7) Gem aligned RR-serier (valgfrit)
    # ---------------------------------------------
    if save_aligned_path is not None:
        save_aligned_path.parent.mkdir(parents=True, exist_ok=True)
        df_aligned = pd.DataFrame(
            {
                "RR_labview_s": rr_lv_m,
                "RR_python_s": rr_py_m,
            }
        )
        df_aligned.to_csv(save_aligned_path, index=False)

    return metrics


def plot_rr_alignment(t_R_lv, RR_lv, t_R_py, RR_py, delta_s, tol_s=0.15):
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
    tol_s: float = 0.05,
):
    """
    Global peak alignment:

    - t_peaks_ref  : reference peak times (seconds since epoch)
    - t_peaks_test : test peak times (seconds since epoch, BEFORE delta)
    - delta_s      : time shift applied to test (typically small, from RR alignment)
    - tol_s        : max difference for peaks to be considered a match (e.g. 0.05s)

    Steps:
      1) Shift test times with delta_s
      2) Compute global overlap window between ref and shifted test
      3) Restrict both series to this overlap
      4) Run greedy two-pointer matching inside the overlap

    Returns:
      tp_idx_ref, tp_idx_test, fn_idx_ref, fp_idx_test
      (all indices refer to ORIGINAL arrays, not cropped subarrays)
    """
    t_peaks_ref = np.asarray(t_peaks_ref, dtype=float)
    t_peaks_test = np.asarray(t_peaks_test, dtype=float) + delta_s

    n_ref = t_peaks_ref.size
    n_test = t_peaks_test.size

    if n_ref == 0 or n_test == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.arange(n_ref, dtype=int),
            np.arange(n_test, dtype=int),
        )

    # 1) Find global overlap window
    window_start = max(t_peaks_ref[0], t_peaks_test[0])
    window_end = min(t_peaks_ref[-1], t_peaks_test[-1])

    if window_end <= window_start:
        # No temporal overlap at all
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.arange(n_ref, dtype=int),
            np.arange(n_test, dtype=int),
        )

    # 2) Mask peaks inside overlap
    ref_mask = (t_peaks_ref >= window_start) & (t_peaks_ref <= window_end)
    test_mask = (t_peaks_test >= window_start) & (t_peaks_test <= window_end)

    t_ref_win = t_peaks_ref[ref_mask]
    t_test_win = t_peaks_test[test_mask]

    ref_idx_win = np.where(ref_mask)[0]
    test_idx_win = np.where(test_mask)[0]

    # 3) Greedy matching in overlap
    tp_ref = []
    tp_test = []
    fn_ref = []
    fp_flags = np.zeros_like(t_test_win, dtype=bool)

    i = 0  # ref
    j = 0  # test

    while i < len(t_ref_win) and j < len(t_test_win):
        t_r = t_ref_win[i]
        t_t = t_test_win[j]

        if t_t < t_r - tol_s:
            # Test-peak too early → FP
            fp_flags[j] = True
            j += 1
        elif t_t > t_r + tol_s:
            # Ref-peak too early → FN
            fn_ref.append(i)
            i += 1
        else:
            # |t_t - t_r| <= tol_s → match (TP)
            tp_ref.append(i)
            tp_test.append(j)
            i += 1
            j += 1

    # Remaining ref-peaks in overlap → FN
    while i < len(t_ref_win):
        fn_ref.append(i)
        i += 1

    # Remaining test-peaks in overlap → FP
    while j < len(t_test_win):
        fp_flags[j] = True
        j += 1

    tp_ref = ref_idx_win[np.asarray(tp_ref, dtype=int)]
    tp_test = test_idx_win[np.asarray(tp_test, dtype=int)]
    fn_ref = ref_idx_win[np.asarray(fn_ref, dtype=int)]
    fp_test = test_idx_win[fp_flags]

    return tp_ref, tp_test, fn_ref, fp_test


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
    tol_s: float = 0.15,
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
    tol_s: float = 0.15,
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
    tol_s: float = 0.15,
    max_duration_s: float | None = None,  # NEW
    rpeak_cache_dir: Path | None = None,
    force_recompute: bool = False,
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

    with tqdm(total=total_iters, desc="RR/peak comparison") as pbar:
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
    tol_peak_s: float = 0.05,   # 50 ms peak tolerance
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