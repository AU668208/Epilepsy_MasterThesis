"""
sqi_bukhari.py

Single-lead ECG signal quality indices inspired by:

    Bukhari et al., "Automated detection of non-physiological artifacts
    on ECG signal: UK Biobank and CRIC", Computers in Biology and Medicine, 2025.

Formål:
    - At kunne beregne simple, robuste SQI-features på et givent ECG-segment,
      f.eks. omkring et anfald (seizure) eller baseline.
    - At kunne estimere thresholds på populationsniveau og efterfølgende
      klassificere nye segmenter som "god" vs "artefakt-tung" ud fra
      amplitude- og frekvensbaserede outliers.

Afhængigheder:
    - numpy
    - scipy (kun signal.welch)

Brug:
    1) Beregn features for alle dine segmenter (seizure/non-seizure)
    2) Estimér thresholds på baggrund af disse
    3) Klassificér nye segmenter eller genbrug features til statistik/plots

Alle funktioner er IO-uafhængige. Du kan kalde dem direkte fra din
seizure-segmenteringskode, hvor du i forvejen har et 1D numpy-array
med ECG-data og sampling rate fs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Konfigurations-objekt
# ---------------------------------------------------------------------------


@dataclass
class SQIConfig:
    """
    Konfiguration for SQI-beregninger.

    Attributes
    ----------
    fs : float
        Sampling rate [Hz].
    flat_window_s : float
        Længde af sub-vinduer til flatline-detektion (rAmpdiff) [s].
    welch_nperseg : Optional[int]
        nperseg til Welch-PSD. Hvis None bruges en default baseret på
        vinduets længde.
    lf_band : Tuple[float, float]
        Lavfrekvent bånd til baseline-wander indeks [Hz].
    hf_band : Tuple[float, float]
        Højfrekvent bånd til EMG-/bevægelsesstøj indeks [Hz].
    mains_freq : float
        Netfrekvens (50 eller 60 Hz).
    mains_bandwidth : float
        +/- bånd omkring netfrekvens til mains-indeks [Hz].
    """

    fs: float
    flat_window_s: float = 2.0
    welch_nperseg: Optional[int] = None
    lf_band: Tuple[float, float] = (0.0, 0.5)
    hf_band: Tuple[float, float] = (40.0, 100.0)
    mains_freq: float = 50.0
    mains_bandwidth: float = 1.0


# ---------------------------------------------------------------------------
# Hjælpefunktioner (internt)
# ---------------------------------------------------------------------------


def _safe_1d(x: np.ndarray) -> np.ndarray:
    """Konverter til 1D float64 og fjern NaNs."""
    arr = np.asarray(x, dtype=float).ravel()
    if np.isnan(arr).any():
        arr = arr[~np.isnan(arr)]
    return arr


def _ampdiff(x: np.ndarray) -> float:
    """Peak-to-peak amplitude (max - min)."""
    if x.size == 0:
        return np.nan
    return float(np.max(x) - np.min(x))


def _rampdiff(x: np.ndarray) -> float:
    """Peak-to-peak amplitude på det absolutte signal."""
    if x.size == 0:
        return np.nan
    ax = np.abs(x)
    return float(np.max(ax) - np.min(ax))


def _band_power(
    f: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
) -> float:
    """Integreret power i givet frekvensbånd."""
    lo, hi = band
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return 0.0
    # Trapz integrering
    return float(np.trapz(psd[mask], f[mask]))


def _mean_frequency(f: np.ndarray, psd: np.ndarray) -> float:
    """Mean frequency = sum(f * P(f)) / sum(P(f))."""
    total_power = np.trapz(psd, f)
    if total_power <= 0:
        return 0.0
    return float(np.trapz(f * psd, f) / total_power)


def _compute_psd(
    x: np.ndarray, fs: float, nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD med fornuftige defaults."""
    if nperseg is None:
        # fx 4 sekunders vindue som default
        target_len = int(4 * fs)
        nperseg = min(len(x), max(128, target_len))
    f, psd = welch(x, fs=fs, nperseg=nperseg)
    return f, psd


def _iter_flat_windows(
    x: np.ndarray, fs: float, window_s: float
) -> Iterable[np.ndarray]:
    """Del signalet op i ikke-overlappende vinduer af længde window_s."""
    n = len(x)
    win_len = int(round(window_s * fs))
    if win_len <= 0:
        yield x
        return
    for start in range(0, n, win_len):
        end = min(start + win_len, n)
        if end > start:
            yield x[start:end]


# ---------------------------------------------------------------------------
# Hovedfunktion: beregn SQI-features for ét segment
# ---------------------------------------------------------------------------


def compute_sqi_features_for_segment(
    ecg: np.ndarray,
    config: SQIConfig,
) -> Dict[str, float]:
    """
    Beregn single-lead SQI-features for ét ECG-segment.

    Parametre
    ---------
    ecg : np.ndarray
        1D array med ECG-samples (samme enhed som resten af din pipeline,
        typisk mV eller V - bare du er konsekvent).
    config : SQIConfig
        Konfiguration med sampling rate og bånddefinitioner.

    Returnerer
    ----------
    features : Dict[str, float]
        Ordbog med numeriske features, bl.a.:
            - ampdiff
            - rampdiff
            - mean_freq
            - lf_power, hf_power, mains_power, total_power
            - lf_rel_power, hf_rel_power, mains_rel_power
            - flat_rampdiff_min, flat_rampdiff_p10
    """
    x = _safe_1d(ecg)
    fs = float(config.fs)

    if x.size == 0:
        # Returnér NaNs, hvis segmentet er tomt
        return {
            "ampdiff": np.nan,
            "rampdiff": np.nan,
            "mean_freq": np.nan,
            "lf_power": np.nan,
            "hf_power": np.nan,
            "mains_power": np.nan,
            "total_power": np.nan,
            "lf_rel_power": np.nan,
            "hf_rel_power": np.nan,
            "mains_rel_power": np.nan,
            "flat_rampdiff_min": np.nan,
            "flat_rampdiff_p10": np.nan,
        }

    # 1) Amplitude-baserede mål
    ampdiff = _ampdiff(x)
    rampdiff = _rampdiff(x)

    # 2) Frekvens-baserede mål (PSD + mean frequency + bånd-power)
    f, psd = _compute_psd(x, fs=fs, nperseg=config.welch_nperseg)
    total_power = float(np.trapz(psd, f))
    lf_power = _band_power(f, psd, config.lf_band)
    hf_power = _band_power(f, psd, config.hf_band)
    mains_band = (
        config.mains_freq - config.mains_bandwidth,
        config.mains_freq + config.mains_bandwidth,
    )
    mains_power = _band_power(f, psd, mains_band)
    mean_freq = _mean_frequency(f, psd)

    if total_power > 0:
        lf_rel = lf_power / total_power
        hf_rel = hf_power / total_power
        mains_rel = mains_power / total_power
    else:
        lf_rel = hf_rel = mains_rel = 0.0

    # 3) Flatline-detektion i 2s-vinduer (rAmpdiff pr. vindue)
    flat_ramps: List[float] = []
    for w in _iter_flat_windows(x, fs, config.flat_window_s):
        flat_ramps.append(_rampdiff(w))

    if len(flat_ramps) == 0:
        flat_min = flat_p10 = np.nan
    else:
        flat_arr = np.array(flat_ramps, dtype=float)
        flat_min = float(np.min(flat_arr))
        flat_p10 = float(np.percentile(flat_arr, 10.0))

    return {
        "ampdiff": ampdiff,
        "rampdiff": rampdiff,
        "mean_freq": mean_freq,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "mains_power": mains_power,
        "total_power": total_power,
        "lf_rel_power": lf_rel,
        "hf_rel_power": hf_rel,
        "mains_rel_power": mains_rel,
        "flat_rampdiff_min": flat_min,
        "flat_rampdiff_p10": flat_p10,
    }


# ---------------------------------------------------------------------------
# Threshold-estimering på populationsniveau
# (inspireret af modificeret Grubbs + percentiler)
# ---------------------------------------------------------------------------


@dataclass
class SQIThresholds:
    """
    Thresholds til klassifikation af artefaktiske segmenter.

    Disse kan estimeres fra en større samling segmenter (både seizure og
    non-seizure) og derefter bruges til nye segmenter.

    Vi gemmer både øvre/lower grænser for relevante features samt
    simple "Z-score" lignende parametre.
    """

    # Amplitude
    ampdiff_high: float
    rampdiff_low: float

    # Frekvens
    mean_freq_high: float
    hf_rel_high: float
    lf_rel_high: float

    # Flatline
    flat_rampdiff_low: float


def estimate_sqi_thresholds_from_population(
    feature_dicts: Iterable[Dict[str, float]],
    amp_high_percentile: float = 99.0,
    ramp_low_percentile: float = 5.0,
    mean_freq_high_percentile: float = 99.0,
    hf_rel_high_percentile: float = 99.0,
    lf_rel_high_percentile: float = 99.0,
    flat_rampdiff_low_percentile: float = 5.0,
) -> SQIThresholds:
    """
    Estimér thresholds fra en samling features (population).

    I stedet for at implementere Grubbs 1:1 bruger vi robuste percentiler,
    som er meget tæt på idéen i artiklen: at finde ekstremt høje/lave
    værdier i fordelingen.

    Parametre
    ---------
    feature_dicts : Iterable[Dict[str, float]]
        Liste/iterator af features (output fra compute_sqi_features_for_segment).
    amp_high_percentile : float
        Percentil for high-amp threshold (fx 99.0).
    ramp_low_percentile : float
        Percentil for lav rAmpdiff (flatline) (fx 5.0).
    mean_freq_high_percentile : float
        Percentil for mean frequency.
    hf_rel_high_percentile : float
        Percentil for høj relativ HF-power.
    lf_rel_high_percentile : float
        Percentil for høj relativ LF-power (kraftig baseline wander).
    flat_rampdiff_low_percentile : float
        Percentil for flatline-mål.

    Returnerer
    ----------
    SQIThresholds
    """
    # Saml relevante features i arrays
    ampdiffs: List[float] = []
    rampdiffs: List[float] = []
    mean_freqs: List[float] = []
    hf_rels: List[float] = []
    lf_rels: List[float] = []
    flat_mins: List[float] = []

    for feat in feature_dicts:
        amp = feat.get("ampdiff", np.nan)
        rmp = feat.get("rampdiff", np.nan)
        mf = feat.get("mean_freq", np.nan)
        hf_rel = feat.get("hf_rel_power", np.nan)
        lf_rel = feat.get("lf_rel_power", np.nan)
        flat_min = feat.get("flat_rampdiff_min", np.nan)

        if not np.isnan(amp):
            ampdiffs.append(amp)
        if not np.isnan(rmp):
            rampdiffs.append(rmp)
        if not np.isnan(mf):
            mean_freqs.append(mf)
        if not np.isnan(hf_rel):
            hf_rels.append(hf_rel)
        if not np.isnan(lf_rel):
            lf_rels.append(lf_rel)
        if not np.isnan(flat_min):
            flat_mins.append(flat_min)

    def _percentile_safe(values: List[float], q: float, default: float) -> float:
        if len(values) == 0:
            return default
        return float(np.percentile(np.array(values, dtype=float), q))

    ampdiff_high = _percentile_safe(ampdiffs, amp_high_percentile, np.inf)
    rampdiff_low = _percentile_safe(rampdiffs, ramp_low_percentile, 0.0)
    mean_freq_high = _percentile_safe(mean_freqs, mean_freq_high_percentile, np.inf)
    hf_rel_high = _percentile_safe(hf_rels, hf_rel_high_percentile, np.inf)
    lf_rel_high = _percentile_safe(lf_rels, lf_rel_high_percentile, np.inf)
    flat_rampdiff_low = _percentile_safe(flat_mins, flat_rampdiff_low_percentile, 0.0)

    return SQIThresholds(
        ampdiff_high=ampdiff_high,
        rampdiff_low=rampdiff_low,
        mean_freq_high=mean_freq_high,
        hf_rel_high=hf_rel_high,
        lf_rel_high=lf_rel_high,
        flat_rampdiff_low=flat_rampdiff_low,
    )


# ---------------------------------------------------------------------------
# Klassifikation af ét segment ud fra thresholds
# ---------------------------------------------------------------------------


def classify_segment_with_thresholds(
    features: Dict[str, float],
    thresholds: SQIThresholds,
) -> Dict[str, object]:
    """
    Klassificér ét segment som "godt" vs "artefaktisk" ud fra thresholds.

    Returnerer en ordbog med:
        - flags for hver feature (True = indenfor acceptable grænser)
        - samlet "is_good" flag
        - en simpel SQI-score i [0, 1] (andel af kriterier opfyldt)
    """
    amp = features.get("ampdiff", np.nan)
    rmp = features.get("rampdiff", np.nan)
    mf = features.get("mean_freq", np.nan)
    hf_rel = features.get("hf_rel_power", np.nan)
    lf_rel = features.get("lf_rel_power", np.nan)
    flat_min = features.get("flat_rampdiff_min", np.nan)

    flags: Dict[str, bool] = {}

    # Amplitude for høj?
    flags["amp_ok"] = not (amp > thresholds.ampdiff_high)

    # For lav amplitudeforskel (flatline)
    flags["ramp_ok"] = not (rmp < thresholds.rampdiff_low)

    # Frekvens for høj (meget HF-støj)
    flags["mean_freq_ok"] = not (mf > thresholds.mean_freq_high)

    # Relativ HF-power for høj
    flags["hf_rel_ok"] = not (hf_rel > thresholds.hf_rel_high)

    # Relativ LF-power for høj (kraftig baseline wander)
    flags["lf_rel_ok"] = not (lf_rel > thresholds.lf_rel_high)

    # Mindste rAmpdiff i 2s-vinduer for lav (lokal flatline)
    flags["flat_ok"] = not (flat_min < thresholds.flat_rampdiff_low)

    # Samlet score
    valid_flags = [v for v in flags.values() if v is not None]
    if len(valid_flags) == 0:
        sqi_score = np.nan
    else:
        sqi_score = float(sum(bool(v) for v in valid_flags) / len(valid_flags))

    is_good = bool(all(flags.values()))

    result: Dict[str, object] = {
        "is_good": is_good,
        "sqi_score": sqi_score,
    }
    # Merge flags og original features ind i ét dict, hvis man vil gemme alt.
    result.update({f"flag_{k}": v for k, v in flags.items()})
    result.update({f"feat_{k}": v for k, v in features.items()})
    return result


# ---------------------------------------------------------------------------
# Convenience-funktioner til batchede segmenter
# ---------------------------------------------------------------------------


def compute_sqi_for_many_segments(
    segments: Iterable[np.ndarray],
    config: SQIConfig,
) -> List[Dict[str, float]]:
    """
    Beregn SQI-features for en samling segmenter.

    Parameters
    ----------
    segments : Iterable[np.ndarray]
        Iterator/liste af 1D ECG-segmenter.
    config : SQIConfig
        Konfiguration med sampling rate, bånd etc.

    Returns
    -------
    List[Dict[str, float]]
        Liste af feature-ordbøger (én per segment).
    """
    out: List[Dict[str, float]] = []
    for seg in segments:
        out.append(compute_sqi_features_for_segment(seg, config=config))
    return out


def classify_many_segments(
    segments: Iterable[np.ndarray],
    config: SQIConfig,
    thresholds: SQIThresholds,
) -> List[Dict[str, object]]:
    """
    Kombi-funktion: beregn features + klassifikation for flere segmenter.

    Praktisk hvis du fx har en liste af seizure-segmenter og vil have en
    samlet oversigt.

    Returns
    -------
    List[Dict[str, object]]
        Liste af results (se classify_segment_with_thresholds).
    """
    results: List[Dict[str, object]] = []
    for seg in segments:
        feats = compute_sqi_features_for_segment(seg, config=config)
        res = classify_segment_with_thresholds(feats, thresholds=thresholds)
        results.append(res)
    return results
