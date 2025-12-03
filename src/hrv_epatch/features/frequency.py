# --- freq_tools.py ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, stft, find_peaks

# ---------- Welch PSD ----------
def compute_psd(sig, fs, nperseg=None, overlap=0.5, detrend="constant"):
    """
    Robust PSD-beregning med welch, der fungerer også for korte segmenter.
    """
    sig = np.asarray(sig)

    # standard: ~8 sekunders vindue
    if nperseg is None:
        nperseg = int(8 * fs)

    # cap nperseg til signalets længde
    nperseg = min(nperseg, len(sig))

    # hvis signalet er ekstremt kort, giver PSD ikke mening
    if nperseg < 4:
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
        detrend=detrend,
    )
    return f, Pxx


def plot_psd(sig, fs, fmax=None, line_freq=50.0, harmonics=4, annotate=True):
    """
    Plot PSD i dB. Marker evt. 50 Hz og harmoniske.
    Returnerer (f, Pxx).
    """
    f, Pxx = compute_psd(sig, fs)
    if fmax is None:
        fmax = min(150.0, f[-1])
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(f[m], 10*np.log10(Pxx[m] + 1e-20))
    if annotate and line_freq is not None:
        for k in range(1, harmonics+1):
            lf = k*line_freq
            if lf <= fmax:
                ax.axvline(lf, linestyle='--')
    ax.set_xlabel('Frekvens (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title('Welch PSD')
    plt.tight_layout()
    plt.show()
    return f, Pxx

# ---------- Bandpower ----------
def bandpower(f, Pxx, f1, f2):
    m = (f>=f1) & (f<=f2)
    if not np.any(m):
        return 0.0
    return float(np.trapz(Pxx[m], f[m]))

def summarize_psd_bands(f, Pxx, bands=None):
    """
    Returnerer DataFrame med (band, f_low, f_high, power, rel_power).
    """
    if bands is None:
        bands = [
            ('ULF', 0.0, 0.5),
            ('LF', 0.5, 5.0),
            ('QRS-ish', 5.0, 40.0),
            ('HF(40-100)', 40.0, 100.0),
            ('Line(48-52)', 48.0, 52.0),
        ]
    rows = []
    total = bandpower(f, Pxx, f[0], f[-1])
    for name, a, b in bands:
        bp = bandpower(f, Pxx, a, b)
        rows.append({'band': name, 'f_low': a, 'f_high': b, 'power': bp, 'rel_power': bp/(total+1e-20)})
    return pd.DataFrame(rows)

# ---------- Spectrogram (STFT) ----------
def plot_spectrogram(sig, fs, win_sec=4.0, hop_sec=0.25, fmax=None):
    """
    Plot |STFT| i dB som spektrogram.
    """
    nperseg = int(win_sec*fs)
    noverlap = int((win_sec-hop_sec)*fs)
    f, t, Z = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', padded=False, boundary=None)
    S = np.abs(Z)
    if fmax is None:
        fmax = min(100.0, f[-1])
    m = f <= fmax
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(20*np.log10(S[m, :] + 1e-20), aspect='auto', origin='lower',
              extent=[t[0], t[-1], f[m][0], f[m][-1]])
    ax.set_xlabel('Tid (s)')
    ax.set_ylabel('Frekvens (Hz)')
    ax.set_title('Spectrogram (|STFT|, dB)')
    plt.tight_layout()
    plt.show()
    return f, t, S

# ---------- Modulation spectrum ----------
def modulation_spectrum(sig, fs, win_sec=8.0, hop_sec=0.5):
    """
    1) STFT -> S(f,t) = |Z|
    2) FFT over tid for hver frekvens -> M(f, fm)
    3) Returner fm, f, M2D (power), samt (f, t, S)
    """
    nperseg = int(win_sec*fs)
    noverlap = int((win_sec-hop_sec)*fs)
    f, t, Z = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   window='hann', padded=False, boundary=None)
    S = np.abs(Z)  # [F x T]
    M = np.fft.rfft(S, axis=1)          # [F x Fm]
    Mpow = np.abs(M)**2
    fm = np.fft.rfftfreq(S.shape[1], d=(t[1]-t[0])) if len(t)>1 else np.array([0.0])
    return fm, f, Mpow, (f, t, S)

def plot_modulation_spectrum_1d(sig, fs, win_sec=8.0, hop_sec=0.5, fm_max=5.0):
    """
    1D modulation: sum over carrier-frekvenser → M1(fm).
    Plotter relativ power.
    """
    fm, f, M2D, _ = modulation_spectrum(sig, fs, win_sec, hop_sec)
    M1 = M2D.sum(axis=0)
    if fm_max is None:
        fm_max = min(6.0, fm[-1])
    m = fm <= fm_max
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(fm[m], M1[m] / (M1[m].sum() + 1e-20))
    ax.set_xlabel('Modulationsfrekvens (Hz)')
    ax.set_ylabel('Rel. modulation power')
    ax.set_title('Modulation spectrum (1D)')
    plt.tight_layout()
    plt.show()
    return fm, M1

def plot_modulation_spectrum_2d(sig, fs, win_sec=8.0, hop_sec=0.5, fm_max=5.0, fmax=100.0):
    """
    2D modulation: carrier-frekvens (y) vs. modulationsfrekvens (x) i log-power.
    """
    fm, f, M2D, _ = modulation_spectrum(sig, fs, win_sec, hop_sec)
    m_fm = fm <= fm_max
    m_f  = f  <= fmax
    Mshow = np.log10(M2D[np.ix_(m_f, m_fm)] + 1e-20)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(Mshow, aspect='auto', origin='lower',
              extent=[fm[m_fm][0], fm[m_fm][-1], f[m_f][0], f[m_f][-1]])
    ax.set_xlabel('Modulationsfrekvens (Hz)')
    ax.set_ylabel('Frekvens (Hz)')
    ax.set_title('Modulation spectrum (2D, log-power)')
    plt.tight_layout()
    plt.show()
    return fm, f, M2D

# ---------- “Noise fingerprint” ----------
def noise_fingerprint(sig, fs, window_sec=8.0, hop_sec=2.0, fmax=150.0, top_n=8, prominence_db=6.0):
    """
    Gennemsnit PSD over overlappende vinduer og find gennemgående peak-frekvenser.
    Returnerer (peaks_df, (f, meanP)).
    """
    n = len(sig)
    step = int(hop_sec*fs)
    win  = int(window_sec*fs)
    psds = []
    for i0 in range(0, n-win+1, step):
        seg = sig[i0:i0+win]
        f, Pxx = compute_psd(seg, fs)
        psds.append(Pxx)
    if not psds:
        f, Pxx = compute_psd(sig, fs)
        meanP = Pxx
    else:
        meanP = np.mean(np.vstack(psds), axis=0)
    m = f <= fmax
    y = 10*np.log10(meanP[m] + 1e-20)
    peaks, props = find_peaks(y, prominence=prominence_db)
    prominences = props.get('prominences', np.zeros_like(peaks))
    order = np.argsort(prominences)[::-1]
    peaks = peaks[order][:top_n]
    prominences = prominences[order][:top_n]
    peak_freqs = f[m][peaks]
    peak_db = y[peaks]
    df = pd.DataFrame({'freq_hz': peak_freqs, 'level_db': peak_db, 'prominence_db': prominences})
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(f[m], y)
    for fhz, lvl in zip(peak_freqs, peak_db):
        ax.axvline(fhz, linestyle='--')
    ax.set_xlabel('Frekvens (Hz)')
    ax.set_ylabel('Gns. PSD (dB)')
    ax.set_title('Gennemsnitlig PSD med dominerende støj-peakfrekvenser')
    plt.tight_layout()
    plt.show()
    return df, (f, meanP)
