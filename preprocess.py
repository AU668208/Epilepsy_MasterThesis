from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import firwin, filtfilt, resample_poly, freqz, find_peaks
from nptdms import TdmsFile

# ---------- Utils ----------
def secs_to_samples(t, fs): 
    return int(round(t * fs))

def time_axis(x: np.ndarray, fs: float) -> np.ndarray:
    return np.arange(len(x)) / fs

def moving_average(x: np.ndarray, half_width: int = 4) -> np.ndarray:
    """Glidende middel med rektangulært vindue; half_width=4 => 8-punkts."""
    w = 2 * max(0, half_width)
    if w < 1:
        return x.copy()
    kernel = np.ones(w) / w
    pad = w
    xpad = np.r_[x[pad:0:-1], x, x[-2:-pad-2:-1]]
    y = np.convolve(xpad, kernel, mode="same")[pad:-pad]
    return y

# ---------- Filter design ----------
def design_highpass_fir(fs: float, fp: float = 1.0, fs_stop: float = 0.5, numtaps: int = 257) -> np.ndarray:
    """Linear-phase FIR highpass (baseline-fjernelse)."""
    return firwin(numtaps, cutoff=fp, window="hann", pass_zero=False, fs=fs)

def design_lowpass_fir(fs: float, fp: float = 32.0, fs_stop: float = 50.0, numtaps: int = 257) -> np.ndarray:
    """Linear-phase FIR lowpass (~QRS fokus)."""
    return firwin(numtaps, cutoff=fp, window="hann", pass_zero=True, fs=fs)

def filtfilt_fir(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return filtfilt(b, [1.0], x, padlen=min(3 * (len(b) - 1), max(1, len(x)//2 - 1)))

def fir_freq_response(b: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(frekvens [Hz], magnitude [dB])"""
    w, h = freqz(b, worN=4096)
    f = w * fs / (2*np.pi)
    db = 20*np.log10(np.maximum(np.abs(h), 1e-12))
    return f, db

# ---------- I/O ----------
def load_tdms_first_channel(path_tdms: str, fs_override: Optional[float] = None) -> Dict[str, object]:
    """
    Indlæser første kanal fra en TDMS-fil.
    Returnerer et dict med:
      - "signal": numpy array (float)
      - "fs": float (Hz, samplingfrekvens)
      - "start_time": wf_start_time (datetime eller None)
    """
    td = TdmsFile.read(path_tdms)
    ch = td.groups()[0].channels()[0]

    # Data
    x = ch.data.astype(float)

    # Samplingfrekvens
    inc = ch.properties.get("wf_increment", None)
    fs = fs_override if fs_override is not None else (None if inc is None else 1.0 / inc)
    if fs is None:
        raise ValueError("Samplingfrekvens ukendt – angiv fs_override")

    # Starttid
    start_time = ch.properties.get("wf_start_time", None)

    return {
        "signal": x,
        "fs": float(fs),
        "start_time": start_time
    }

# ---------- Pre steps ----------
@dataclass
class PreParams:
    drop_start_s: float = 0.0
    drop_end_s: float = 0.0
    target_fs: Optional[float] = 256.0  # resample hvis ønsket
    hp_pass_hz: float = 1.0             # baseline
    hp_stop_hz: float = 0.5
    lp_pass_hz: float = 32.0            # QRS-fokus
    lp_stop_hz: float = 50.0
    taps_hp: int = 257
    taps_lp: int = 257
    smooth_half_width: int = 4          # 8-punkts moving average (0=off)

def trim_signal(x: np.ndarray, fs: float, drop_start_s: float, drop_end_s: float) -> np.ndarray:
    i0 = secs_to_samples(drop_start_s, fs)
    i1 = len(x) - secs_to_samples(drop_end_s, fs)
    return x[i0:max(i1, i0+1)]

def maybe_resample(x: np.ndarray, fs_in: float, target_fs: Optional[float]) -> Tuple[np.ndarray, float]:
    if target_fs is None or abs(target_fs - fs_in) < 1e-6:
        return x, fs_in
    from math import gcd
    up = int(target_fs * 1000)
    dn = int(fs_in * 1000)
    g = gcd(up, dn)
    y = resample_poly(x, up//g, dn//g)
    return y, float(target_fs)

def prefilter_pipeline(x: np.ndarray, fs: float, p: PreParams) -> Dict[str, np.ndarray]:
    """Returnerer alle mellemtrin til visuel QA."""
    stages = {"raw": x}
    # trim
    xt = trim_signal(stages["raw"], fs, p.drop_start_s, p.drop_end_s)
    stages["trim"] = xt
    # resample
    xr, fsr = maybe_resample(xt, fs, p.target_fs)
    stages["resampled"] = xr
    # highpass
    bhp = design_highpass_fir(fsr, fp=p.hp_pass_hz, fs_stop=p.hp_stop_hz, numtaps=p.taps_hp)
    xhp = filtfilt_fir(bhp, xr)
    stages["highpass"] = xhp
    # lowpass
    blp = design_lowpass_fir(fsr, fp=p.lp_pass_hz, fs_stop=p.lp_stop_hz, numtaps=p.taps_lp)
    xlp = filtfilt_fir(blp, xhp)
    stages["lowpass"] = xlp
    # smoothing
    xs = moving_average(xlp, half_width=max(0, p.smooth_half_width))
    stages["smoothed"] = xs
    # gem fs og filtre
    stages["_fs"] = np.array([fsr])
    stages["_bhp"] = bhp
    stages["_blp"] = blp
    return stages

# ---------- Peaks & QC ----------
@dataclass
class PeakParams:
    min_rr_s: float = 0.25   # min afstand mellem peaks
    prominence: float = 0.8  # relativ vht. std (skaleres)
    height_std: float = 0.0  # ekstra højdekrav i std-enheder (0=none)

def detect_r_peaks(x: np.ndarray, fs: float, p: PeakParams = PeakParams()) -> np.ndarray:
    """Returnerer indeks for R-toppe i forbehandlet signal."""
    xz = x - np.median(x)
    s = np.std(xz) + 1e-9
    distance = int(round(p.min_rr_s * fs))
    kwargs = dict(distance=max(distance, 1))
    if p.prominence is not None:
        kwargs["prominence"] = float(p.prominence * s)
    if p.height_std and p.height_std > 0:
        kwargs["height"] = float(p.height_std * s)
    peaks, _ = find_peaks(xz, **kwargs)
    return peaks

def qc_metrics(x_raw: np.ndarray, x_proc: np.ndarray) -> dict:
    """Enkle indikatorer: RMS, std, 'flatline'-procent."""
    def stats(x):
        dif = np.diff(x)
        flat = np.mean(np.abs(dif) < (1e-9 + 1e-6*np.std(x))) * 100.0
        return dict(rms=float(np.sqrt(np.mean(x**2))), std=float(np.std(x)), flat_pct=float(flat))
    out = {f"raw_{k}": v for k,v in stats(x_raw).items()}
    out.update({f"proc_{k}": v for k,v in stats(x_proc).items()})
    return out
