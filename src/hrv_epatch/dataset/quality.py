import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, welch
from src.hrv_epatch.io.tdms import load_tdms_from_path

def plot_all_psds(df_rec, outpath, channel_hint="EKG", max_freq=100.0, *, k_outliers=8):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    all_psd = []
    all_uids = []
    freqs_ref = None

    # For outlier-score: netstøj-bånd (tilpas hvis du vil)
    band = (45.0, 55.0)

    plt.figure(figsize=(12, 8))

    for _, row in df_rec.iterrows():
        uid = row["recording_uid"]
        tdms = row["tdms_path"]
        fs   = row["fs"]

        try:
            sig, _ = load_tdms_from_path(tdms, channel_hint=channel_hint)
            sig = np.asarray(sig)
            if sig.ndim == 2:
                sig = sig[:, 0]
        except Exception:
            continue

        freqs, psd = welch(sig, fs, nperseg=4096)

        # limit freq range
        m = freqs <= max_freq
        freqs = freqs[m]
        psd = psd[m]

        if freqs_ref is None:
            freqs_ref = freqs
        else:
            # skip if grid mismatch (shouldn't happen with same fs/nperseg, but safe)
            if len(freqs) != len(freqs_ref) or not np.allclose(freqs, freqs_ref):
                continue

        all_psd.append(psd)
        all_uids.append(uid)

        # plot all as background
        plt.semilogy(freqs, psd, alpha=0.12, linewidth=0.8)

    if not all_psd:
        plt.close()
        return

    P = np.vstack(all_psd)  # shape (n_rec, n_freq)

    # summary curves
    p10 = np.nanpercentile(P, 10, axis=0)
    p50 = np.nanpercentile(P, 50, axis=0)
    p90 = np.nanpercentile(P, 90, axis=0)

    plt.semilogy(freqs_ref, p50, linewidth=2.2, label="Median")
    plt.fill_between(freqs_ref, p10, p90, alpha=0.15, label="10–90% band")

    # outliers by band power
    band_mask = (freqs_ref >= band[0]) & (freqs_ref <= band[1])
    band_power = np.trapz(P[:, band_mask], freqs_ref[band_mask], axis=1)
    idx = np.argsort(band_power)[::-1][:k_outliers]

    for j in idx:
        plt.semilogy(freqs_ref, P[j], linewidth=2.0, alpha=0.9, label=f"outlier uid={all_uids[j]}")

    plt.title("PSD of all recordings (background + median + outliers)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power spectral density")
    plt.xlim(0, max_freq)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



def compute_spectral_uniqueness(df_rec, channel_hint="EKG"):
    """
    Returns a DataFrame with a uniqueness score for each recording,
    based on deviation from the median PSD across all recordings.
    """

    psd_list = []
    freqs = None

    # Step 1: compute PSDs for all
    for _, row in df_rec.iterrows():
        try:
            sig, _ = load_tdms_from_path(row["tdms_path"], channel_hint="EKG")
        except:
            psd_list.append(None)
            continue

        sig = np.asarray(sig)
        if sig.ndim == 2:
            sig = sig[:, 0]

        f, p = welch(sig, row["fs"], nperseg=4096)

        if freqs is None:
            freqs = f
        psd_list.append(p)

    # Step 2: compute median PSD
    psd_array = np.vstack([p for p in psd_list if p is not None])
    median_psd = np.median(psd_array, axis=0)

    # Step 3: compute uniqueness score
    scores = []
    for uid, p in zip(df_rec["recording_uid"], psd_list):
        if p is None:
            scores.append(np.nan)
            continue

        # deviation in log-space
        diff = np.log10(p + 1e-12) - np.log10(median_psd + 1e-12)
        score = float(np.sqrt(np.mean(diff**2)))
        scores.append(score)

    df_unique = pd.DataFrame({
        "recording_uid": df_rec["recording_uid"],
        "spectral_uniqueness_score": scores
    })

    return df_unique


def _preprocess_for_quality(
    sig,
    fs,
    trim_start_s=60.0,
    trim_end_s=60.0,
    bp_low=0.5,
    bp_high=40.0,
    notch_freq=50.0,
    notch_q=30.0,
):
    """
    Trimmer kanter og laver en standard ECG-preprocessing:
    - trimmer de første/sidste X sekunder
    - bandpass 0.5–40 Hz
    - 50 Hz notch

    Bruges KUN til kvalitetsvurdering (ikke til dine egentlige analyser).
    """

    sig = np.asarray(sig).astype(float)
    if sig.ndim == 2:
        sig = sig[:, 0]

    n = sig.size
    if n < 10:
        return sig

    # --- trim start/slut ---
    n_start = int(trim_start_s * fs)
    n_end = int(trim_end_s * fs)

    start_idx = min(n_start, n // 2)  # hvis meget kort, trim maks halvdelen
    end_idx = max(n - n_end, start_idx + 1)

    sig = sig[start_idx:end_idx]

    # --- bandpass 0.5–40 Hz ---
    nyq = fs / 2.0
    low = bp_low / nyq
    high = bp_high / nyq
    if high >= 1.0:
        high = 0.99
    b_bp, a_bp = butter(2, [low, high], btype="band")
    sig = filtfilt(b_bp, a_bp, sig)

    # --- 50 Hz notch ---
    w0 = notch_freq / nyq
    if 0 < w0 < 1:
        b_notch, a_notch = iirnotch(w0, notch_q)
        sig = filtfilt(b_notch, a_notch, sig)

    return sig

def _apply_trim(sig: np.ndarray, fs: float, trim_start_s: float, trim_end_s: float) -> np.ndarray:
    n = len(sig)
    i0 = int(round(trim_start_s * fs))
    i1 = n - int(round(trim_end_s * fs))
    i0 = max(0, min(i0, n))
    i1 = max(i0, min(i1, n))
    return sig[i0:i1]

def compute_recording_quality(
    df_rec: pd.DataFrame,
    channel_hint: str = "EKG",
    prefer_tz: str = "Europe/Copenhagen",
    window_s: float = 10.0,
    flatline_std_thresh: float = 1.0,
    noise_std_factor: float = 5.0,
    clip_range_thresh: float = 0.9,
    trim_df: pd.DataFrame | None = None, 
    axis: str = "raw"
) -> pd.DataFrame:
    """
    Recording-level kvalitetsvurdering med:
      - preprocessing (trim + 0.5–40 Hz + 50 Hz notch)
      - vinduesvise STD / range / flatline / noisebursts
      - PSD-baseret QRS-energi vs HF-støj

    Returnerer én række per recording.
    """

    trim_map = None
    if trim_df is not None:
        trim_map = (
            trim_df[["recording_uid","trim_start_s","trim_end_s"]]
            .drop_duplicates("recording_uid")
            .set_index("recording_uid")
        )

    rows = []

    for _, row in tqdm(df_rec.iterrows(), total=len(df_rec)):
        rec_uid = row.get("recording_uid")
        patient_id = int(row["patient_id"])
        recording_id = int(row["recording_id"])
        fs = float(row["fs"])
        tdms_path = Path(row["tdms_path"])

        # ---- load råsignal ----
        try:
            sig_raw, _ = load_tdms_from_path(
                tdms_path, channel_hint=channel_hint, prefer_tz=prefer_tz
            )
            # Apply trim BEFORE preprocessing if axis="trim"
            if axis == "trim" and trim_map is not None and rec_uid in trim_map.index:
                ts = float(trim_map.loc[rec_uid, "trim_start_s"])
                te = float(trim_map.loc[rec_uid, "trim_end_s"])
                sig_raw = _apply_trim(np.asarray(sig_raw), fs, ts, te)
            
        except Exception as e:
            rows.append(
                {
                    "recording_uid": rec_uid,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "load_error": str(e),
                }
            )
            continue

        # ---- preprocess til kvalitetsvurdering ----
        sig = _preprocess_for_quality(sig_raw, fs)
        sig = np.asarray(sig).astype(float)
        if sig.ndim == 2:
            sig = sig[:, 0]

        n = sig.size
        if n < int(window_s * fs):
            rows.append(
                {
                    "recording_uid": rec_uid,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "load_error": "too_short_after_trim",
                }
            )
            continue

        # -------------------------------------------------
        # TIME-DOMAIN window metrics
        # -------------------------------------------------
        win_len = int(window_s * fs)
        n_win = n // win_len
        sig_win = sig[: n_win * win_len].reshape(n_win, win_len)

        std_win = np.std(sig_win, axis=1)
        max_win = np.max(sig_win, axis=1)
        min_win = np.min(sig_win, axis=1)
        range_win = max_win - min_win
        diff_abs_median_win = np.median(np.abs(np.diff(sig_win, axis=1)), axis=1)

        std_med = float(np.median(std_win))
        std_p01 = float(np.percentile(std_win, 1))
        std_p99 = float(np.percentile(std_win, 99))

        range_med = float(np.median(range_win))
        range_p99 = float(np.percentile(range_win, 99))

        diff_med_global = float(np.median(diff_abs_median_win))

        sig_min = float(sig.min())
        sig_max = float(sig.max())
        sig_range = sig_max - sig_min

        frac_flatline = float(np.mean(std_win < flatline_std_thresh))
        frac_noiseburst = float(np.mean(std_win > std_med * noise_std_factor))
        frac_clipping = float(np.mean(range_win > clip_range_thresh * sig_range))

        # -------------------------------------------------
        # FREQUENCY-DOMAIN (på det filtrerede midterstykke)
        # -------------------------------------------------
        # PSD med Welch
        freqs, psd = welch(sig, fs=fs, nperseg=min(4096, n))

        # indbygget bånd: low (0.5–5), QRS (5–25), HF (25–40)
        f = freqs
        p = psd
        total_power = float(np.trapz(p, f) + 1e-12)

        band_low = float(
            np.trapz(p[(f >= 0.5) & (f < 5.0)], f[(f >= 0.5) & (f < 5.0)])
        )
        band_qrs = float(
            np.trapz(p[(f >= 5.0) & (f < 25.0)], f[(f >= 5.0) & (f < 25.0)])
        )
        band_hf = float(
            np.trapz(p[(f >= 25.0) & (f < 40.0)], f[(f >= 25.0) & (f < 40.0)])
        )

        qrs_power_ratio = band_qrs / total_power
        hf_ratio = band_hf / total_power
        lf_ratio = band_low / total_power

        # hvor "spids" er PSD'en?
        psd_kurt = float(
            np.mean((p - np.mean(p)) ** 4) / ((np.std(p) ** 4) + 1e-12)
        )

        rows.append(
            {
                "recording_uid": rec_uid,
                "patient_id": patient_id,
                "recording_id": recording_id,
                "fs": fs,
                "n_samples_used": n,

                # time-domain
                "win_std_median": std_med,
                "win_std_p01": std_p01,
                "win_std_p99": std_p99,
                "win_range_median": range_med,
                "win_range_p99": range_p99,
                "frac_flatline_windows": frac_flatline,
                "frac_noiseburst_windows": frac_noiseburst,
                "frac_clipping_windows": frac_clipping,
                "diff_abs_median_global": diff_med_global,
                "sig_min_proc": sig_min,
                "sig_max_proc": sig_max,
                "sig_range_proc": sig_range,

                # frequency-domain
                "total_power": total_power,
                "band_low": band_low,
                "band_qrs": band_qrs,
                "band_hf": band_hf,
                "qrs_power_ratio": qrs_power_ratio,
                "hf_ratio": hf_ratio,
                "lf_ratio": lf_ratio,
                "psd_kurtosis": psd_kurt,

                "load_error": "",
            }
        )

    return pd.DataFrame(rows)


def classify_recordings(
    df_qual: pd.DataFrame,
    qrs_good_min: float = 0.10,       # før 0.10 — beholdes
    qrs_bad_max: float = 0.05,        # før 0.05 — beholdes
    std_good_min: float = 30.0,       # NY (før 50)
    std_bad_max: float = 20.0,        # NY (før 30)
    std_good_max: float = 800.0,
    flatline_good_max: float = 0.05,
    flatline_bad_min: float = 0.30,
    noise_good_max: float = 0.40,     # øget fra 0.30 → 0.40 (mere realistisk)
    noise_bad_min: float = 0.60
):
    """
    Version 4.1 af recording-level klassifikation.
    - lidt mere tolerant STD-thresholds
    - stadig stringent ift. QRS-power
    """

    labels = []
    reasons = []
    include_flags = []

    for _, r in df_qual.iterrows():
        if r.get("load_error", ""):
            labels.append("bad")
            reasons.append(f"load_error: {r['load_error']}")
            include_flags.append(False)
            continue

        qrs_ratio = r["qrs_power_ratio"]
        std_med = r["win_std_median"]
        frac_flat = r["frac_flatline_windows"]
        frac_noise = r["frac_noiseburst_windows"]

        # ---------- BAD ----------
        if (
            qrs_ratio < qrs_bad_max
            or std_med < std_bad_max
            or frac_flat > flatline_bad_min
            or frac_noise > noise_bad_min
        ):
            labels.append("bad")
            include_flags.append(False)

            reasons_bad = []
            if qrs_ratio < qrs_bad_max:
                reasons_bad.append(f"very low QRS power ({qrs_ratio:.3f})")
            if std_med < std_bad_max:
                reasons_bad.append(f"very low median STD ({std_med:.1f})")
            if frac_flat > flatline_bad_min:
                reasons_bad.append(f"many flatline windows ({frac_flat:.2f})")
            if frac_noise > noise_bad_min:
                reasons_bad.append(f"extreme noise bursts ({frac_noise:.2f})")

            reasons.append("; ".join(reasons_bad))
            continue

        # ---------- GOOD ----------
        if (
            qrs_ratio >= qrs_good_min
            and std_med >= std_good_min
            and std_med <= std_good_max
            and frac_flat <= flatline_good_max
            and frac_noise <= noise_good_max
        ):
            labels.append("good")
            include_flags.append(True)
            reasons.append("clear QRS, acceptable noise")
            continue

        # ---------- BORDERLINE ----------
        labels.append("borderline")
        include_flags.append(False)
        reasons.append("mixed quality, below good thresholds")

    df_out = df_qual.copy()
    df_out["quality_label"] = labels
    df_out["quality_reason"] = reasons
    df_out["include_for_rr"] = include_flags
    return df_out

def compute_window_quality_vs_seizure(
    df_rec,
    df_evt,
    *,
    trim_df=None,
    use_trimmed_axis=True,   # True => use t0_trim/t1_trim and trim signal
    channel_hint="EKG",
    prefer_tz="Europe/Copenhagen",
    window_s=10.0,
    flatline_std_thresh=1.0,
    noise_std_factor=5.0,
    clip_range_thresh=0.9,
):
    """
    Beregn vindues-baserede kvalitetsmål og label hver 10 s window
    som 'baseline' eller 'seizure' afhængigt af overlap med annoterede anfald.
    Returnerer én række per window.
    """
    rows = []

    for _, rec in tqdm(df_rec.iterrows(), total=len(df_rec), desc="Window quality"):
        rec_uid = rec["recording_uid"]
        patient_id = int(rec["patient_id"])
        recording_id = int(rec["recording_id"])
        fs = float(rec["fs"])
        tdms_path = Path(rec["tdms_path"])

        # relevant seizures for denne recording
        evt_rec = df_evt[df_evt["recording_uid"] == rec_uid].copy()

        try:
            sig_raw, _ = load_tdms_from_path(
                tdms_path, channel_hint=channel_hint, prefer_tz=prefer_tz
            )
        except Exception as e:
            # hvis der er load-fejl, spring recording over
            continue

        trim_start_s = 0.0
        trim_end_s = 0.0
        if use_trimmed_axis and trim_df is not None:
            trow = trim_df.loc[trim_df["recording_uid"] == rec_uid]
            if not trow.empty:
                trim_start_s = float(pd.to_numeric(trow["trim_start_s"].iloc[0], errors="coerce") or 0.0)
                trim_end_s   = float(pd.to_numeric(trow["trim_end_s"].iloc[0], errors="coerce") or 0.0)

        # apply trim (recording-specific)
        sig_raw = np.asarray(sig_raw)
        if sig_raw.ndim == 2:
            sig_raw = sig_raw[:, 0]

        i0 = int(round(trim_start_s * fs))
        i1 = sig_raw.shape[0] - int(round(trim_end_s * fs))
        i0 = max(0, min(i0, sig_raw.shape[0]))
        i1 = max(i0, min(i1, sig_raw.shape[0]))
        sig_raw_trim = sig_raw[i0:i1]


        sig = _preprocess_for_quality(sig_raw_trim, fs)
        sig = np.asarray(sig).astype(float)
        if sig.ndim == 2:
            sig = sig[:, 0]

        n = sig.size
        win_len = int(window_s * fs)
        if n < win_len:
            continue

        # opdel i ikke-overlappende vinduer
        n_win = n // win_len
        sig_win = sig[: n_win * win_len].reshape(n_win, win_len)

        # basale features per window
        std_win = np.std(sig_win, axis=1)
        max_win = np.max(sig_win, axis=1)
        min_win = np.min(sig_win, axis=1)
        range_win = max_win - min_win
        diff_abs_med = np.median(np.abs(np.diff(sig_win, axis=1)), axis=1)

        # globale referencer til noise/flatline-def
        std_med = float(np.median(std_win))
        sig_range = float(np.max(sig) - np.min(sig) + 1e-12)

        # thresholds som i compute_recording_quality
        noise_std_thresh = noise_std_factor * std_med

        # tidsskala for vinduer (relativt til trimmede signal)
        # t=0 svarer til start efter trimming, for Study 2 er det nok
        win_starts_s = np.arange(n_win) * window_s
        win_ends_s = win_starts_s + window_s

        # label hver window ift. seizures i evt_rec (t0, t1 i sekunder fra recording-start)
        # her antages, at t0/t1 er relative til recording-start; trimming betyder,
        # at absolute offset kan ignoreres, hvis vi kun bruger relative mønstre.
        # Hvis du vil være helt præcis, kan du korrigere for trim_start_s.
        if use_trimmed_axis and ("t0_trim" in evt_rec.columns) and ("t1_trim" in evt_rec.columns):
            seizure_intervals = evt_rec[["t0_trim", "t1_trim"]].to_numpy(dtype=float)
        else:
            seizure_intervals = evt_rec[["t0", "t1"]].to_numpy(dtype=float)


        context_labels = []
        for ws, we in zip(win_starts_s, win_ends_s):
            has_overlap = False
            for t0, t1 in seizure_intervals:
                if (we > t0) and (ws < t1):
                    has_overlap = True
                    break
            context_labels.append("seizure" if has_overlap else "baseline")


        # simple artefakt-flags per window
        is_flat = std_win < flatline_std_thresh
        is_noise = std_win > noise_std_thresh
        is_clip = range_win > (clip_range_thresh * sig_range)

        for i in range(n_win):
            rows.append(
                {
                    "recording_uid": rec_uid,
                    "patient_id": patient_id,
                    "recording_id": recording_id,
                    "window_idx": i,
                    "win_start_s": win_starts_s[i],
                    "win_end_s": win_ends_s[i],
                    "context": context_labels[i],          # baseline vs seizure
                    "std": float(std_win[i]),
                    "range": float(range_win[i]),
                    "diff_abs_med": float(diff_abs_med[i]),
                    "is_flatline": bool(is_flat[i]),
                    "is_noiseburst": bool(is_noise[i]),
                    "is_clipping": bool(is_clip[i]),
                }
            )

    return pd.DataFrame(rows)

def build_recording_quality_tables(
    df_quality: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    rec_uid_col: str = "recording_uid",
    frac_cols=("frac_flatline_windows", "frac_noiseburst_windows", "frac_clipping_windows"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_big:  per-recording table (good for appendix)
      df_sum:  dataset-level summary table (good for main report)
    """
    df = df_quality.copy()

    # keep only successful rows
    if "load_error" in df.columns:
        df = df[df["load_error"].fillna("") == ""].copy()

    # enforce numeric
    for c in frac_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Per-recording table (appendix)
    keep_cols = [
        rec_uid_col, patient_col, "recording_id",
        "n_samples_used",
        *frac_cols,
        "win_std_median", "win_range_median",
        "qrs_power_ratio", "hf_ratio",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df_big = df[keep_cols].sort_values([patient_col, "recording_id"]).reset_index(drop=True)

    # Dataset summary (weighted by analyzable windows / samples)
    # If you have window count, use it; else approximate weighting by n_samples_used
    w = pd.to_numeric(df.get("n_samples_used", 1), errors="coerce").fillna(1).to_numpy()
    w = np.maximum(w, 1)

    def wmean(x):
        x = pd.to_numeric(x, errors="coerce").to_numpy()
        m = np.isfinite(x)
        if not np.any(m):
            return np.nan
        return float(np.average(x[m], weights=w[m]))

    summary_rows = []
    for c in frac_cols:
        summary_rows.append({
            "metric": c,
            "mean_unweighted": float(df[c].mean(skipna=True)),
            "median": float(df[c].median(skipna=True)),
            "p10": float(df[c].quantile(0.10)),
            "p90": float(df[c].quantile(0.90)),
            "mean_weighted_by_samples": wmean(df[c]),
        })

    # also include how many recordings were processed
    df_sum = pd.DataFrame(summary_rows)
    df_sum.insert(0, "n_recordings", int(len(df_big)))

    return df_big, df_sum

def export_quality_tables(
    df_big: pd.DataFrame,
    df_sum: pd.DataFrame,
    out_dir: Path,
    *,
    prefix: str = "recording_quality",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_big.to_csv(out_dir / f"{prefix}_per_recording.csv", index=False)
    df_sum.to_csv(out_dir / f"{prefix}_summary.csv", index=False)

    # Optional: LaTeX versions (nice for appendix/report)
    df_sum.to_latex(out_dir / f"{prefix}_summary.tex", index=False, float_format="%.3f")
    # NB: per-recording LaTeX can get huge; usually keep as CSV for appendix
