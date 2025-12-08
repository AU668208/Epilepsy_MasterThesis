import numpy as np
import pandas as pd
import neurokit2 as nk

from src.hrv_epatch.io.tdms import load_tdms_from_path


# -------------------------------------------------------------------
# NeuroKit Quality
# -------------------------------------------------------------------
def compute_nk_quality(sig_window, fs):
    try:
        ecg = nk.ecg_clean(sig_window, sampling_rate=fs)
        _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=fs)
        quality = nk.ecg_quality(ecg, rpeaks=rpeaks, sampling_rate=fs)
        q = float(np.nanmean(quality))
        rpeaks = rpeaks["ECG_R_Peaks"]
    except Exception:
        q = 0.0
        rpeaks = np.array([])
    return q, rpeaks


# -------------------------------------------------------------------
# RR stability
# -------------------------------------------------------------------
def rr_stability(rpeaks, fs, rr_thr=0.08):
    """Return True if BAD window."""
    if len(rpeaks) < 3:
        return True  # too few R-peaks -> bad
    rr = np.diff(rpeaks) / fs
    sdnn = np.std(rr)
    return (sdnn > rr_thr)  # BAD if RR variance > threshold (seconds)


# -------------------------------------------------------------------
# Window classifier (weighted)
# -------------------------------------------------------------------
def classify_window(
    sig_window,
    fs,
    rms_thr,
    quality_thr=0.8,
    rr_thr=0.08,
    w_rms=0.5,
    w_nk=0.3,
    w_rr=0.2,
    score_thr=0.4,
    use_nk=True,
):
    # RMS check
    rms = float(np.sqrt(np.mean(sig_window**2)))
    rms_bad = rms > rms_thr

    if not use_nk:
        # Brug KUN RMS i midten af signalet
        q = 1.0          # “god” kvalitet
        nk_bad = False
        rr_bad = False
    else:
        # NeuroKit quality + RR
        q, rpeaks = compute_nk_quality(sig_window, fs)
        nk_bad = (q < quality_thr)
        rr_bad = rr_stability(rpeaks, fs, rr_thr=rr_thr)

    # Weighted combination
    score = w_rms * rms_bad + w_nk * nk_bad + w_rr * rr_bad
    combined_bad = score > score_thr

    return combined_bad, rms, q, rms_bad, nk_bad, rr_bad



# -------------------------------------------------------------------
# Main trim estimator
# -------------------------------------------------------------------
def estimate_trim_from_rms(
    sig,
    fs,
    win_s=10.0,
    step_s=5.0,
    rms_factor=3.0,
    min_clean_run_start_s=60.0,
    min_clean_run_end_s=60.0,
    edge_start_span_s=20*60.0,
    edge_end_span_s=20*60.0,
    pre_clean_before_start_s=10.0,   # rent lige før trim_start
    post_clean_after_end_s=10.0,     # rent lige efter trim_end
):
    n = len(sig)
    win = int(win_s * fs)
    step = int(step_s * fs)

    if n < win:
        return 0.0, 0.0

    # --- 1) RMS over hele recording ---
    rms_vals = []
    t_centers = []
    for start in range(0, n - win + 1, step):
        w = sig[start:start+win]
        rms_vals.append(float(np.sqrt(np.mean(w**2))))
        t_centers.append((start + win/2) / fs)

    rms_vals = np.array(rms_vals)
    t_centers = np.array(t_centers)

    med_rms = np.median(rms_vals)
    rms_thr = med_rms * rms_factor

    bad = rms_vals > rms_thr
    clean = ~bad

    def find_first_run(mask, min_len):
        count = 0
        for i, val in enumerate(mask):
            if val:
                count += 1
                if count >= min_len:
                    return i - count + 1
            else:
                count = 0
        return None

    def find_last_run(mask, min_len):
        count = 0
        for i in range(len(mask) - 1, -1, -1):
            if mask[i]:
                count += 1
                if count >= min_len:
                    # i er start-index for sidste run
                    return i, i + count - 1
            else:
                count = 0
        return None, None

    step_s = step_s = step / fs  # sekunder per vindue
    total_s = n / fs

    # --- 2) START: kig kun i første edge_start_span_s ---
    start_region = t_centers <= edge_start_span_s
    clean_start = clean & start_region

    min_run_start = int(np.ceil(min_clean_run_start_s / step_s))
    first_idx = find_first_run(clean_start, min_run_start)

    if first_idx is None:
        trim_start_s = 0.0
    else:
        # hvor mange vinduer ren tid vil vi have før cutoff?
        pre_win = int(np.floor(pre_clean_before_start_s / step_s))
        # vi kan max flytte os inde i run'et
        cut_idx = first_idx + pre_win
        trim_start_s = t_centers[cut_idx] - win_s/2

    # --- 3) SLUT: kig kun i sidste edge_end_span_s ---
    end_region = t_centers >= (total_s - edge_end_span_s)
    clean_end = clean & end_region

    min_run_end = int(np.ceil(min_clean_run_end_s / step_s))
    # find sidste run af mindst min_run_end længde
    run_start, run_end = find_last_run(clean_end, min_run_end)

    if run_start is None:
        trim_end_s = 0.0
    else:
        # vi vil have post_clean_after_end_s ren tid EFTER cutoff,
        # før støjen starter (som vi alligevel fjerner)
        post_win = int(np.floor(post_clean_after_end_s / step_s))

        # cut_idx bliver lidt tidligere end run_end,
        # så der er et lille rent stykke efter i det vi skærer væk.
        cut_idx = max(run_start, run_end - post_win)

        # vi beholder alt FØR cut_idx' vindue:
        trim_end_s = max(total_s - (t_centers[cut_idx] + win_s/2), 0.0)

    return float(trim_start_s), float(trim_end_s)





# -------------------------------------------------------------------
# Build trim table
# -------------------------------------------------------------------
def build_trim_table(df_rec, **kwargs):
    rows = []
    for _, row in df_rec.iterrows():
        tdms_path = row["tdms_path"]
        sig, meta = load_tdms_from_path(
            tdms_path,
            channel_hint="EKG",
            prefer_tz="Europe/Copenhagen",
            assume_source_tz="UTC",
            prefer_naive_local=True,
        )
        fs = meta.fs

        trim_start_s, trim_end_s = estimate_trim_from_rms(sig, fs, **kwargs)

        rows.append({
            "recording_uid": row["recording_uid"],
            "patient_id": row["patient_id"],
            "recording_id": row["recording_id"],
            "trim_start_s": trim_start_s,
            "trim_end_s": trim_end_s,
            "duration_s": len(sig) / fs,
        })

    return pd.DataFrame(rows)

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

def plot_trim_diagnostics(sig, fs, trim_start_s=None, trim_end_s=None,
                          edge_span_s=3600.0, win_s=10.0, step_s=5.0):
    n = len(sig)
    total_s = n / fs

    # ---- Første edge (0 -> edge_span_s) ----
    t_edge1_max = min(edge_span_s, total_s/2)
    n_edge1 = int(t_edge1_max * fs)
    sig1 = sig[:n_edge1]
    t1 = np.arange(len(sig1)) / fs

    # RMS
    win = int(win_s * fs)
    step = int(step_s * fs)
    rms1 = []
    t1_centers = []
    for start in range(0, len(sig1)-win+1, step):
        w = sig1[start:start+win]
        rms1.append(float(np.sqrt(np.mean(w**2))))
        t1_centers.append((start+win/2)/fs)
    rms1 = np.array(rms1)
    t1_centers = np.array(t1_centers)

    # NeuroKit process på første edge
    signals1, info1 = nk.ecg_process(sig1, sampling_rate=fs)
    q1_full = np.asarray(signals1["ECG_Quality"])

    # downsample quality til samme vinduer som RMS
    q1_win = []
    for start in range(0, len(sig1)-win+1, step):
        q1_win.append(float(np.nanmean(q1_full[start:start+win])))
    q1_win = np.array(q1_win)


    # ---- Plot første edge ----
    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1, ax2 = axes

    ax1.plot(t1_centers/60, rms1)
    ax1.set_ylabel("RMS")
    ax1.set_title("Start of recording")

    ax2.plot(t1_centers/60, q1_win)
    ax2.set_ylabel("NK quality")
    ax2.set_xlabel("Time from start (min)")

    if trim_start_s is not None:
        ax1.axvline(trim_start_s/60, color="r", linestyle="--", label="trim_start")
        ax2.axvline(trim_start_s/60, color="r", linestyle="--")
        ax1.legend()

    plt.tight_layout()
    plt.show()

    # ---- Sidste edge (total_s - edge_span_s -> end) ----
    t_edge2_min = max(total_s - edge_span_s, total_s/2)
    start2 = int(t_edge2_min * fs)
    sig2 = sig[start2:]
    t2 = np.arange(len(sig2))/fs + t_edge2_min

    rms2 = []
    t2_centers = []
    for start in range(0, len(sig2)-win+1, step):
        w = sig2[start:start+win]
        rms2.append(float(np.sqrt(np.mean(w**2))))
        t2_centers.append((start+win/2)/fs + t_edge2_min)
    rms2 = np.array(rms2)
    t2_centers = np.array(t2_centers)

    signals2, info2 = nk.ecg_process(sig2, sampling_rate=fs)
    q2_full = np.asarray(signals2["ECG_Quality"])

    q2_win = []
    for start in range(0, len(sig2)-win+1, step):
        q2_win.append(float(np.nanmean(q2_full[start:start+win])))
    q2_win = np.array(q2_win)


    fig, axes = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    ax1, ax2 = axes

    # plot tid relativt til slutningen (fx "min before end")
    t_rel = (total_s - t2_centers) / 60.0  # minutter før slut

    ax1.plot(t_rel, rms2)
    ax1.set_ylabel("RMS")
    ax1.set_title("End of recording")

    ax2.plot(t_rel, q2_win)
    ax2.set_ylabel("NK quality")
    ax2.set_xlabel("Minutes before end")

    if trim_end_s is not None:
        ax1.axvline(trim_end_s/60, color="r", linestyle="--", label="trim_end")
        ax2.axvline(trim_end_s/60, color="r", linestyle="--")
        ax1.legend()

    plt.gca().invert_xaxis()  # så 0 er ved slutningen
    plt.tight_layout()
    plt.show()
