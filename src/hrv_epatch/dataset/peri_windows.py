from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PeriWindow:
    recording_uid: str
    patient_id: int
    seizure_id: int
    role: str               # baseline_far, baseline_near, ictal, post_near, post_far
    t_start: float          # seconds from rec_start
    t_end: float
    x: np.ndarray           # raw ECG samples
    fs: float


def extract_ecg_window(sig, fs, rec_start, abs_t_start, abs_t_end):
    """
    Convert absolute times → sample indices → extract ECG window.
    abs_t_start, abs_t_end er datetimes.
    """
    if abs_t_start is None or abs_t_end is None:
        return None

    t0 = (abs_t_start - rec_start).total_seconds()
    t1 = (abs_t_end   - rec_start).total_seconds()
    if t1 <= 0:
        return None

    i0 = max(0, int(t0 * fs))
    i1 = min(len(sig), int(t1 * fs))
    if i1 <= i0:
        return None

    return t0, t1, sig[i0:i1]


def build_peri_windows_for_recording(sig, meta, df_evt_rec, rec_uid,
                                     roles=None):
    """
    Bygger peri-ictal vinduer for alle seizures i én recording.
    df_evt_rec: events for kun denne recording med absolute_start/end
    roles: dict med:
        role_name -> (offset_start_s, offset_end_s)
    Returns list[PeriWindow]
    """
    if roles is None:
        roles = {
            "baseline_far":  (-20*60, -20*60 + 60),
            "baseline_near": (-2*60,  -2*60  + 60),
            "ictal":         (0, None),       # None = brug t1_s fra event
            "post_near":     (2*60,   2*60   + 60),
            "post_far":      (20*60,  20*60  + 60),
        }

    windows = []

    for _, ev in df_evt_rec.iterrows():
        seizure_id = ev["seizure_id"]
        abs_start = ev["absolute_start"]
        abs_end = ev["absolute_end"]

        for role, (off0, off1) in roles.items():

            if role == "ictal":
                # råt ictal vindue = hele anfaldet
                abs_t0 = abs_start
                abs_t1 = abs_end

            else:
                # peri-ictal vindue (fast længde)
                abs_t0 = abs_start + pd.to_timedelta(off0, unit="s")
                abs_t1 = abs_start + pd.to_timedelta(off1, unit="s")

            result = extract_ecg_window(
                sig=sig,
                fs=meta.fs,
                rec_start=meta.start_time,
                abs_t_start=abs_t0,
                abs_t_end=abs_t1,
            )

            if result is None:
                continue

            t0_s, t1_s, x = result

            windows.append(
                PeriWindow(
                    recording_uid=rec_uid,
                    patient_id=int(ev["patient_id"]),
                    seizure_id=int(seizure_id),
                    role=role,
                    t_start=t0_s,
                    t_end=t1_s,
                    x=x,
                    fs=meta.fs,
                )
            )

    return windows
