from dataclasses import dataclass
from typing import List

import pandas as pd
import numpy as np


@dataclass
class SeizureEvent:
    """Simple container for a seizure relative to a recording."""
    seizure_id: int
    t0: float  # seconds from recording start
    t1: float  # seconds from recording start


def build_seizure_events_from_df(
    seizure_df: pd.DataFrame,
    rec_start: pd.Timestamp,
    rec_end: pd.Timestamp,
) -> List[SeizureEvent]:
    """
    Build a list of SeizureEvent for a SINGLE recording.

    - Uses EEG times if both start_eeg and end_eeg are available
    - Otherwise falls back to clinical times (start_clinic / end_clinic)
    - Only keeps seizures that overlap the recording interval [rec_start, rec_end]
    - Clips events to the recording boundaries
    """

    if seizure_df is None or seizure_df.empty:
        return []

    events: List[SeizureEvent] = []

    for _, row in seizure_df.iterrows():
        start_eeg = row.get("start_eeg")
        end_eeg   = row.get("end_eeg")
        start_cl  = row.get("start_clinic")
        end_cl    = row.get("end_clinic")

        # Priority: EEG > clinical
        if pd.notna(start_eeg) and pd.notna(end_eeg):
            s_abs, e_abs = start_eeg, end_eeg
        elif pd.notna(start_cl) and pd.notna(end_cl):
            s_abs, e_abs = start_cl, end_cl
        else:
            # no valid full interval for this row
            continue

        if e_abs <= s_abs:
            continue

        # Check overlap with recording interval [rec_start, rec_end]
        if e_abs <= rec_start or s_abs >= rec_end:
            continue  # no overlap

        # Clip to recording interval
        s_clip = max(s_abs, rec_start)
        e_clip = min(e_abs, rec_end)

        t0 = (s_clip - rec_start).total_seconds()
        t1 = (e_clip - rec_start).total_seconds()
        if t1 <= t0:
            continue

        seiz_num = row.get("seizure_number", -1)
        try:
            seiz_id = int(seiz_num)
        except Exception:
            seiz_id = -1

        events.append(
            SeizureEvent(
                seizure_id=seiz_id,
                t0=float(t0),
                t1=float(t1),
            )
        )

    return events
