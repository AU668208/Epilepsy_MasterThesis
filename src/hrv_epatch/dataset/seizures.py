from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np

@dataclass
class SeizureEvent:
    seizure_id: int
    t0: float
    t1: float
    duration_s: float

    # absolute timestamps (clipped)
    absolute_start: Optional[pd.Timestamp] = None
    absolute_end: Optional[pd.Timestamp] = None

    # optional alternative sources (raw, relative to rec_start)
    t0_video: Optional[float] = None
    t1_video: Optional[float] = None
    t0_clinical: Optional[float] = None
    t1_clinical: Optional[float] = None

    # NEW: trim metadata + trim-corrected times
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0

    t0_trim: Optional[float] = None
    t1_trim: Optional[float] = None
    t0_video_trim: Optional[float] = None
    t1_video_trim: Optional[float] = None
    t0_clinical_trim: Optional[float] = None
    t1_clinical_trim: Optional[float] = None


def _sec(ts: pd.Timestamp, rec_start: pd.Timestamp) -> float:
    return float((ts - rec_start).total_seconds())


def _trim_rel(t: Optional[float], trim_start_s: float) -> Optional[float]:
    if t is None or pd.isna(t):
        return None
    return max(0.0, float(t) - float(trim_start_s))


def build_seizure_events_from_df(
    seizure_df: pd.DataFrame,
    rec_start: pd.Timestamp,
    rec_end: pd.Timestamp,
    *,
    trim_start_s: float = 0.0,
    trim_end_s: float = 0.0,
) -> List[SeizureEvent]:
    """
    Build a list of SeizureEvent for a SINGLE recording.

    Raw times (t0/t1, t0_clinical, t0_video, ...) are relative to rec_start.
    Trimmed times (t0_trim, ...) are raw minus trim_start_s, clipped to >= 0.
    """

    if seizure_df is None or seizure_df.empty:
        return []

    trim_start_s = float(trim_start_s or 0.0)
    trim_end_s = float(trim_end_s or 0.0)

    events: List[SeizureEvent] = []

    for _, row in seizure_df.iterrows():
        start_eeg = row.get("start_eeg")
        end_eeg   = row.get("end_eeg")
        start_cl  = row.get("start_clinic")
        end_cl    = row.get("end_clinic")

        # Priority: EEG > clinical (as before)
        if pd.notna(start_eeg) and pd.notna(end_eeg):
            s_abs, e_abs = start_eeg, end_eeg
        elif pd.notna(start_cl) and pd.notna(end_cl):
            s_abs, e_abs = start_cl, end_cl
        else:
            continue

        if e_abs <= s_abs:
            continue

        # Overlap check with recording
        if e_abs <= rec_start or s_abs >= rec_end:
            continue

        # Clip to recording interval
        s_clip = max(s_abs, rec_start)
        e_clip = min(e_abs, rec_end)

        # RAW relative times (to rec_start)
        t0 = _sec(s_clip, rec_start)
        t1 = _sec(e_clip, rec_start)
        if t1 <= t0:
            continue

        # raw times for the underlying sources (not clipped)
        t0_video = _sec(s_abs, rec_start) if pd.notna(s_abs) else None
        t1_video = _sec(e_abs, rec_start) if pd.notna(e_abs) else None
        t0_clin  = _sec(start_cl, rec_start) if pd.notna(start_cl) else None
        t1_clin  = _sec(end_cl, rec_start) if pd.notna(end_cl) else None

        seiz_num = row.get("seizure_number", -1)
        try:
            seiz_id = int(seiz_num)
        except Exception:
            seiz_id = -1

        # TRIMMED versions
        t0_trim = _trim_rel(t0, trim_start_s)
        t1_trim = _trim_rel(t1, trim_start_s)

        ev = SeizureEvent(
            seizure_id=seiz_id,
            t0=float(t0),
            t1=float(t1),
            duration_s=float(t1 - t0),
            absolute_start=s_clip,
            absolute_end=e_clip,

            t0_video=t0_video,
            t1_video=t1_video,
            t0_clinical=t0_clin,
            t1_clinical=t1_clin,

            trim_start_s=trim_start_s,
            trim_end_s=trim_end_s,

            t0_trim=t0_trim,
            t1_trim=t1_trim,
            t0_video_trim=_trim_rel(t0_video, trim_start_s),
            t1_video_trim=_trim_rel(t1_video, trim_start_s),
            t0_clinical_trim=_trim_rel(t0_clin, trim_start_s),
            t1_clinical_trim=_trim_rel(t1_clin, trim_start_s),
        )

        events.append(ev)

    return events


# def build_seizure_events_from_df(
#     seizure_df: pd.DataFrame,
#     rec_start: pd.Timestamp,
#     rec_end: pd.Timestamp,
#     *,
#     trim_start_s: float = 0.0,
#     trim_end_s: float = 0.0,
# ) -> List[SeizureEvent]:

#     """
#     Build a list of SeizureEvent for a SINGLE recording.

#     - Uses EEG times if both start_eeg and end_eeg are available
#     - Otherwise falls back to clinical times (start_clinic / end_clinic)
#     - Only keeps seizures that overlap the recording interval [rec_start, rec_end]
#     - Clips events to the recording boundaries
#     """

#     if seizure_df is None or seizure_df.empty:
#         return []

#     events: List[SeizureEvent] = []

#     for _, row in seizure_df.iterrows():
#         start_eeg = row.get("start_eeg")
#         end_eeg   = row.get("end_eeg")
#         start_cl  = row.get("start_clinic")
#         end_cl    = row.get("end_clinic")

#         # Priority: EEG > clinical
#         if pd.notna(start_eeg) and pd.notna(end_eeg):
#             s_abs, e_abs = start_eeg, end_eeg
#         elif pd.notna(start_cl) and pd.notna(end_cl):
#             s_abs, e_abs = start_cl, end_cl
#         else:
#             # no valid full interval for this row
#             continue

#         if e_abs <= s_abs:
#             continue

#         # Check overlap with recording interval [rec_start, rec_end]
#         if e_abs <= rec_start or s_abs >= rec_end:
#             continue  # no overlap

#         # Clip to recording interval
#         s_clip = max(s_abs, rec_start)
#         e_clip = min(e_abs, rec_end)

#         t0 = (s_clip - rec_start).total_seconds()
#         t1 = (e_clip - rec_start).total_seconds()
#         if t1 <= t0:
#             continue

#         seiz_num = row.get("seizure_number", -1)
#         try:
#             seiz_id = int(seiz_num)
#         except Exception:
#             seiz_id = -1

#         events.append(
#             SeizureEvent(
#                 seizure_id=seiz_id,
#                 t0=float(t0),
#                 t1=float(t1),
#                 t0_video=(s_abs - rec_start).total_seconds() if pd.notna(s_abs) else None,
#                 t1_video=(e_abs - rec_start).total_seconds() if pd.notna(e_abs) else None,
#                 t0_clinical=(start_cl - rec_start).total_seconds() if pd.notna(start_cl) else None,
#                 t1_clinical=(end_cl - rec_start).total_seconds() if pd.notna(end_cl) else None,
#             )
#         )

#     return events
