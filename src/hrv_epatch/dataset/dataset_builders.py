"""
Dataset builders for Study 3.

Creates:
- df_rec : recording-level summary table
- df_evt : seizure-event table aligned with df_rec

Relies on:
- hrv_epatch.dataset.index
- hrv_epatch.dataset.seizures
- hrv_epatch.io.tdms
"""

from pathlib import Path
import pandas as pd

from src.hrv_epatch.dataset.index import build_recording_index
from src.hrv_epatch.dataset.seizures import build_seizure_events_from_df
from src.hrv_epatch.io.tdms import load_tdms_from_path
from src.hrv_epatch.dataset.annotations import load_annotations


def build_df_rec_and_df_evt(tdms_root: Path,
                            ann_root: Path,
                            channel_hint: str = "EKG",
                            prefer_tz: str = "Europe/Copenhagen",
                            assume_source_tz: str = "UTC",
                            prefer_naive_local: bool = True):
    """
    Main builder function for Study 3.
    Returns:
        df_rec : one row per recording
        df_evt : one row per seizure event
    """

    # 1) Find all TDMS + annotation pairs
    entries = build_recording_index(tdms_root, ann_root)

    rec_rows = []
    evt_rows = []

    # 2) Loop through all recordings
    for entry in entries:
        key = entry.key  # RecordingKey(patient_id, enrollment_id, recording_id)

        # Create UID e.g. P05a_R02
        enr = key.enrollment_id or ""
        rec_uid = f"P{key.patient_id:02d}{enr}_R{key.recording_id:02d}"

        # 3) Load TDMS to get meta (fs, n_samples, start_time)
        sig, meta = load_tdms_from_path(
            str(entry.tdms_path),
            channel_hint=channel_hint,
            prefer_tz=prefer_tz,
            assume_source_tz=assume_source_tz,
            prefer_naive_local=prefer_naive_local,
        )

        duration_s = meta.n_samples / meta.fs
        rec_start = meta.start_time
        rec_end = rec_start + pd.to_timedelta(duration_s, unit="s") if rec_start is not None else None

        # 4) Append recording row
        rec_rows.append({
            "recording_uid": rec_uid,
            "patient_id": key.patient_id,
            "enrollment_id": key.enrollment_id,
            "recording_id": key.recording_id,
            "tdms_path": str(entry.tdms_path),
            "annotation_path": str(entry.annotation_path) if entry.annotation_path else None,
            "fs": meta.fs,
            "n_samples": meta.n_samples,
            "recording_start": rec_start,
            "recording_end": rec_end,
            "rec_duration_s": duration_s,
        })

        # 5) Handle annotations → seizure events
        if entry.annotation_path is None:
            continue

        # load annotations
        ann_df = None
        try:
            ann_df = load_annotations(entry.annotation_path)
        except Exception as e:
            print(f"[WARN] Could not load annotations for {rec_uid}: {e}")
            continue

        if ann_df is None or ann_df.empty:
            continue

        # Build seizure events using your robust logic
        try:
            events = build_seizure_events_from_df(
                seizure_df=ann_df,
                rec_start=rec_start,
                rec_end=rec_end,
            )
        except Exception as e:
            print(f"[WARN] Could not parse seizure events for {rec_uid}: {e}")
            continue

        print(f"[INFO] {rec_uid}: {len(events)} seizures")

        for ev in events:
            # t0/t1 er sekunder fra rec_start
            abs_start = None
            abs_end = None
            if rec_start is not None:
                abs_start = rec_start + pd.to_timedelta(ev.t0, unit="s")
                abs_end   = rec_start + pd.to_timedelta(ev.t1, unit="s")

            evt_rows.append({
                "recording_uid": rec_uid,
                "patient_id": key.patient_id,
                "enrollment_id": key.enrollment_id,
                "recording_id": key.recording_id,
                "seizure_id": ev.seizure_id,
                "t0_s": ev.t0,
                "t1_s": ev.t1,
                "absolute_start": abs_start,
                "absolute_end": abs_end,
            })

    df_rec = pd.DataFrame(rec_rows).sort_values(
        ["patient_id", "enrollment_id", "recording_id"]
    )
    df_rec.reset_index(drop=True, inplace=True)

    df_evt = pd.DataFrame(evt_rows)

    if not df_evt.empty:
        df_evt = df_evt.sort_values(
            ["patient_id", "enrollment_id", "recording_id", "seizure_id"]
        )
        df_evt.reset_index(drop=True, inplace=True)
    else:
        # Sørg for en tom DataFrame med forventede kolonner,
        # så nedstrøms kode ikke crasher på .columns
        df_evt = pd.DataFrame(
            columns=[
                "recording_uid",
                "patient_id",
                "enrollment_id",
                "recording_id",
                "seizure_id",
                "t0_s",
                "t1_s",
                "absolute_start",
                "absolute_end",
            ]
        )

    return df_rec, df_evt

