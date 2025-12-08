import numpy as np
import pandas as pd
import neurokit2 as nk
from typing import Dict, Iterable, List


def compute_hrv_features_for_segment(
    ecg: np.ndarray,
    fs: float,
    min_peaks: int = 3,
) -> Dict[str, float]:
    """
    Compute short-term HRV features for a single ECG segment using NeuroKit2.

    Fokus: kun korttids-time-domain HRV + SD1/SD2 (Poincaré),
    som giver mening for 40–60 s vinduer.
    """
    ecg = np.asarray(ecg).astype(float)

    # Clean ECG
    try:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs)
    except Exception:
        return _empty_hrv_features()

    # Detect R-peaks
    try:
        _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        rpeaks = info.get("ECG_R_Peaks", np.array([], dtype=int))
    except Exception:
        return _empty_hrv_features()

    if rpeaks is None or len(rpeaks) < min_peaks:
        return _empty_hrv_features()

    # RR intervals in seconds
    rr_sec = np.diff(rpeaks) / fs
    if len(rr_sec) < 2:
        return _empty_hrv_features()

    # Time-domain HRV via NeuroKit (kun korttids-ting)
    try:
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=fs, show=False)
    except Exception:
        hrv_time = pd.DataFrame([{}])

    feat = {}
    if not hrv_time.empty:
        feat.update(hrv_time.iloc[0].to_dict())

    # Beregn SD1, SD2, SD1/SD2 manuelt fra RR-serien
    sdnn = np.std(rr_sec, ddof=1)
    diff_rr = np.diff(rr_sec)
    sdsd = np.std(diff_rr, ddof=1)

    sd1 = np.sqrt(0.5) * sdsd
    # pas på negative hvis meget få punkter – clamp til 0
    sd2_sq = 2 * (sdnn**2) - 0.5 * (sdsd**2)
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
    sd1sd2 = sd1 / sd2 if sd2 > 0 else np.nan

    feat["SD1_manual"] = sd1
    feat["SD2_manual"] = sd2
    feat["SD1SD2_manual"] = sd1sd2

    # Prefix alle nøgler
    prefixed = {f"HRV_{k}": v for k, v in feat.items()}

    if len(prefixed) == 0:
        return _empty_hrv_features()

    return prefixed


def compute_hrv_for_many_segments(
    segments: Iterable[np.ndarray],
    fs: float,
    min_peaks: int = 3,
) -> List[Dict[str, float]]:
    features_list = []
    for seg in segments:
        feats = compute_hrv_features_for_segment(
            ecg=np.asarray(seg),
            fs=fs,
            min_peaks=min_peaks,
        )
        features_list.append(feats)
    return features_list


def _empty_hrv_features() -> Dict[str, float]:
    """
    Return a dict with et lille sæt HRV-nøgler fyldt med NaN,
    så downstream kode kan håndtere manglende HRV.
    """
    base_keys = [
        "HRV_HRV_MeanNN",
        "HRV_HRV_SDNN",
        "HRV_HRV_RMSSD",
        "HRV_HRV_SDSD",
        "HRV_SD1_manual",
        "HRV_SD2_manual",
        "HRV_SD1SD2_manual",
    ]
    return {k: np.nan for k in base_keys}


def compute_hrv_for_many_segments(
    segments: Iterable[np.ndarray],
    fs: float,
    min_peaks: int = 3,
) -> List[Dict[str, float]]:
    """
    Compute HRV features for many ECG segments.

    Parameters
    ----------
    segments : iterable of np.ndarray
        Iterable of ECG segments (1D).
    fs : float
        Sampling frequency in Hz (assumed constant across segments).
    min_peaks : int
        Minimum number of R-peaks required per segment.

    Returns
    -------
    features_list : list of dict
        One dict of HRV features per segment.
    """
    features_list = []
    for seg in segments:
        feats = compute_hrv_features_for_segment(
            ecg=np.asarray(seg),
            fs=fs,
            min_peaks=min_peaks,
        )
        features_list.append(feats)
    return features_list