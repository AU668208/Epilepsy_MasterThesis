# denoise_metrics.py
import numpy as np
import pandas as pd
from scipy.signal import welch, iirnotch, filtfilt, butter

# ---------- PSD / bandpower ----------
def _psd(sig, fs, nperseg=None, overlap=0.5):
    sig = np.asarray(sig)

    # default vindueslængde ~8 sekunder
    if nperseg is None:
        nperseg = int(8 * fs)

    # cap nperseg til signalets længde
    nperseg = min(nperseg, len(sig))

    # hvis signalet er meget kort, giver PSD ikke rigtig mening
    if nperseg < 4:
        # fald tilbage til noget helt simpelt: 1 frekvens-bin
        f = np.array([0.0])
        Pxx = np.array([0.0])
        return f, Pxx

    noverlap = int(nperseg * overlap)

    # sørg for at noverlap < nperseg
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    f, Pxx = welch(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )
    return f, Pxx


def _bandpower(f, Pxx, f1, f2):
    m = (f >= f1) & (f <= f2)
    if not np.any(m): return 0.0
    return float(np.trapz(Pxx[m], f[m]))

# ---------- Notch-comb ved 50 Hz + harmoniske ----------
def apply_notch_comb(sig, fs, line_freq=50.0, Q=30.0, harmonics=3):
    """
    Kaskade af smalle notch-filtre ved 50, 100, 150... Hz.
    Q≈30 er et godt kompromis (smalt, men stabilt).
    """
    x = np.asarray(sig, float)
    y = x.copy()
    for k in range(1, harmonics+1):
        f0 = k * line_freq
        if f0 >= fs/2: break
        b, a = iirnotch(f0/(fs/2), Q)
        y = filtfilt(b, a, y)
    return y

# ---------- Lavpas til “signalbevarelse” (≤40 Hz) ----------
def _butter_lowpass(data, fs, cutoff=40.0, order=4):
    if cutoff >= fs/2: return data
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, data)

# ---------- Støjmålinger + SNR-proxy ----------
def noise_metrics(sig, fs):
    """
    Returnerer dict med PSD-baserede mål + bandpower tabel.
    """
    f, Pxx = _psd(sig, fs)
    bp_total = _bandpower(f, Pxx, f[0], f[-1])
    bp_qrs   = _bandpower(f, Pxx, 5.0, 40.0)
    bp_line  = _bandpower(f, Pxx, 48.0, 52.0)
    bp_hf    = _bandpower(f, Pxx, 40.0, 100.0)
    bp_ulf   = _bandpower(f, Pxx, 0.0, 0.5)

    # ratioer (jo lavere line/hf-ratio, jo bedre kvalitet)
    line_noise_ratio = bp_line / (bp_qrs + 1e-20)
    hf_noise_ratio   = bp_hf  / (bp_qrs + 1e-20)

    # SNR-proxy: “nyttebåndet” (QRS) ift. resten
    noise_power = max(bp_total - bp_qrs, 1e-20)
    snr_qrs = bp_qrs / noise_power

    # RMS
    rms = float(np.sqrt(np.mean(np.asarray(sig, float)**2)))

    bands = pd.DataFrame([
        ("ULF", 0.0, 0.5, bp_ulf,   bp_ulf/(bp_total+1e-20)),
        ("QRS", 5.0, 40.0, bp_qrs,  bp_qrs/(bp_total+1e-20)),
        ("HF",  40.0, 100.0, bp_hf, bp_hf /(bp_total+1e-20)),
        ("Line",48.0, 52.0, bp_line,bp_line/(bp_total+1e-20)),
        ("Total", f[0], f[-1], bp_total, 1.0),
    ], columns=["band","f_low","f_high","power","rel_power"])

    return {
        "rms": rms,
        "line_noise_ratio": float(line_noise_ratio),
        "hf_noise_ratio": float(hf_noise_ratio),
        "snr_qrs": float(snr_qrs),
        "bands": bands,
        "psd": (f, Pxx),
    }

# ---------- Bevarelse: før/efter-sammenligning ----------
def preservation_metrics(sig_pre, sig_post, fs, lowpass_hz=40.0):
    """
    Måler, hvor godt lavfrekvent (≤lowpass_hz) indhold bevares efter filtrering.
    Returnerer korrelation og normeret RMSE.
    """
    x = _butter_lowpass(sig_pre,  fs, cutoff=lowpass_hz)
    y = _butter_lowpass(sig_post, fs, cutoff=lowpass_hz)
    x = np.asarray(x, float); y = np.asarray(y, float)

    # korrelation
    x0 = x - np.mean(x); y0 = y - np.mean(y)
    denom = (np.linalg.norm(x0) * np.linalg.norm(y0) + 1e-20)
    corr = float(np.dot(x0, y0) / denom)

    # nRMSE (normaliseret med dynamik i pre-signalet)
    rmse = np.sqrt(np.mean((y - x)**2))
    dyn  = np.percentile(np.abs(x), 95)  # robust normalisering
    nrmse = float(rmse / (dyn + 1e-20))
    return {"lowband_corr": corr, "lowband_nrmse": nrmse}

# ---------- (Valgfrit) R-peak-match hvis du har peaks ----------
def rpeak_match_rate(peaks_pre, peaks_post, fs, tol_ms=40):
    """
    Andel R-toppe i pre der findes i post indenfor tolerance.
    """
    if len(peaks_pre) == 0 or len(peaks_post) == 0:
        return np.nan
    tol = int(tol_ms * fs / 1000.0)
    j = 0; hits = 0
    for i in peaks_pre:
        while j < len(peaks_post) and peaks_post[j] < i - tol:
            j += 1
        if j < len(peaks_post) and abs(peaks_post[j] - i) <= tol:
            hits += 1; j += 1
    return float(hits / max(len(peaks_pre), 1))
