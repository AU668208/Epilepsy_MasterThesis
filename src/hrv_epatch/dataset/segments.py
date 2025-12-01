import numpy as np
import pandas as pd


def build_segment_index(df_rec: pd.DataFrame,
                        df_evt: pd.DataFrame,
                        window_s: int = 60) -> pd.DataFrame:
    """
    Bygger et segment-index for hele datasættet.
    Hver recording opdeles i ikke-overlappende vinduer på window_s sekunder,
    som labeles som 'seizure' hvis de overlapper et anfald, ellers 'baseline'.

    Returnerer:
        DataFrame med kolonner:
            - segment_id (løbenummer)
            - patient_id
            - enrollment_id
            - recording_uid
            - segment_start (datetime)
            - segment_end   (datetime)
            - label ('seizure' / 'baseline')
    """

    rows = []
    seg_id = 0

    # sørg for datetime
    df_rec = df_rec.copy()
    df_rec["recording_start"] = pd.to_datetime(df_rec["recording_start"])
    df_rec["recording_end"] = pd.to_datetime(df_rec["recording_end"])

    df_evt = df_evt.copy()
    df_evt["absolute_start"] = pd.to_datetime(df_evt["absolute_start"])
    df_evt["absolute_end"] = pd.to_datetime(df_evt["absolute_end"])

    win = pd.to_timedelta(window_s, unit="s")

    for _, rec in df_rec.iterrows():
        rid = rec["recording_uid"]
        pid = rec["patient_id"]
        enr = rec.get("enrollment_id", "")

        start = rec["recording_start"]
        end = rec["recording_end"]

        # events tilhørende denne recording
        ev_rec = df_evt[df_evt["recording_uid"] == rid]

        # antal vinduer
        total_s = (end - start).total_seconds()
        n_win = int(np.floor(total_s / window_s))
        if n_win <= 0:
            continue

        for i in range(n_win):
            seg_start = start + i * win
            seg_end = seg_start + win

            # overlapper segmentet et anfald?
            if not ev_rec.empty:
                overlap_mask = (
                    (ev_rec["absolute_start"] < seg_end) &
                    (ev_rec["absolute_end"] > seg_start)
                )
                is_seizure = overlap_mask.any()
            else:
                is_seizure = False

            label = "seizure" if is_seizure else "baseline"

            rows.append({
                "segment_id": seg_id,
                "patient_id": pid,
                "enrollment_id": enr,
                "recording_uid": rid,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "label": label,
            })
            seg_id += 1

    return pd.DataFrame(rows)
