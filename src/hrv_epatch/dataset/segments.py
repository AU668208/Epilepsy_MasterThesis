import numpy as np
import pandas as pd

def build_segment_index(
    df_rec: pd.DataFrame,
    df_evt: pd.DataFrame,
    *,
    window_s: float = 60.0,
    seizure_buffer_s: float = 0.0,
    label_extended: bool = False,
    use_trimmed_axis: bool = True,   # NEW
) -> pd.DataFrame:
    """
    Build segment index combining recordings and seizures.
    Optionally expand seizure intervals by ± buffer seconds.

    Returns
    -------
    DataFrame with:
    - patient_id
    - recording_uid
    - win_start_s, win_end_s
    - context: baseline / seizure_core / seizure_extended
    """

    rows = []

    for _, rec in df_rec.iterrows():
        uid = rec["recording_uid"]

        # duration (prefer trimmed duration if available)
        rec_dur_s = float(rec.get("rec_duration_s", np.nan))
        if not np.isfinite(rec_dur_s):
            rec_start = rec["recording_start"]
            rec_end   = rec["recording_end"]
            rec_dur_s = (rec_end - rec_start).total_seconds()

        # If we have a trimmed duration available, use it
        if use_trimmed_axis and ("rec_duration_trim_s" in df_rec.columns):
            rec_dur_s = float(rec["rec_duration_trim_s"])

        if rec_dur_s <= 0 or not np.isfinite(rec_dur_s):
            continue

        # Find relevant seizures for this recording
        evts = df_evt[df_evt["recording_uid"] == uid].copy()

        # Build windows
        n_windows = int(rec_dur_s // window_s)
        for w in range(n_windows):
            w0 = w * window_s
            w1 = w0 + window_s

            label = "baseline"

            for _, ev in evts.iterrows():
                if use_trimmed_axis and ("t0_trim" in evts.columns) and ("t1_trim" in evts.columns):
                    core0, core1 = float(ev["t0_trim"]), float(ev["t1_trim"])
                else:
                    core0, core1 = float(ev["t0"]), float(ev["t1"])

                # core overlap
                if (w0 < core1) and (w1 > core0):
                    label = "seizure_core"
                    break

                # extended overlap (only if not already core)
                if label_extended and seizure_buffer_s > 0:
                    ext0 = core0 - seizure_buffer_s
                    ext1 = core1 + seizure_buffer_s
                    if (w0 < ext1) and (w1 > ext0):
                        label = "seizure_extended"

            rows.append({
                "recording_uid": uid,
                "patient_id": rec["patient_id"],
                "recording_id": rec["recording_id"],
                "win_start_s": w0,
                "win_end_s": w1,
                "context": label,
            })

    return pd.DataFrame(rows)

