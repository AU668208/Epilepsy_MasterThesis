# src/hrv_epatch/dataset/summary.py

import numpy as np
import pandas as pd


def build_patient_summary(df_rec: pd.DataFrame,
                          df_evt: pd.DataFrame) -> pd.DataFrame:
    """
    Bygger patient/enrollment-level summary:
      - antal recordings
      - total varighed (timer)
      - antal seizures
      - hours per seizure
    """

    df_rec_sum = df_rec.copy()
    df_evt_sum = df_evt.copy()

    # ens håndtering af enrollment_id
    df_rec_sum["enrollment_id"] = df_rec_sum["enrollment_id"].fillna("")
    df_evt_sum["enrollment_id"] = df_evt_sum["enrollment_id"].fillna("")

    rec_agg = (
        df_rec_sum
        .groupby(["patient_id", "enrollment_id"], dropna=False)
        .agg(
            Recordings=("recording_uid", "nunique"),
            Total_hours=("rec_duration_s", lambda x: x.sum() / 3600.0),
        )
    )

    evt_agg = (
        df_evt_sum
        .groupby(["patient_id", "enrollment_id"], dropna=False)
        .agg(
            Total_seizures=("seizure_id", "count"),
        )
    )

    df_patient_summary = (
        rec_agg
        .merge(evt_agg, on=["patient_id", "enrollment_id"], how="left")
        .fillna({"Total_seizures": 0})
        .reset_index()
    )

    df_patient_summary["Hours_per_seizure"] = df_patient_summary.apply(
        lambda row: row["Total_hours"] / row["Total_seizures"]
        if row["Total_seizures"] > 0 else np.nan,
        axis=1,
    )

    df_patient_summary = df_patient_summary.rename(
        columns={
            "patient_id": "Patient",
            "enrollment_id": "Enrollment",
        }
    )
    df_patient_summary["Enrollment"] = df_patient_summary["Enrollment"].replace({"": "-"})

    return df_patient_summary


def compute_within_recording_isi(df_evt: pd.DataFrame):
    """
    Inter-seizure intervals (sekunder) KUN indenfor samme recording_uid.
    Returnerer en liste af floats.
    """
    isi = []

    for rid, df_r in df_evt.groupby("recording_uid"):
        df_r_sorted = df_r.sort_values("absolute_start")
        if len(df_r_sorted) < 2:
            continue

        intervals = (
            df_r_sorted["absolute_start"]
            .diff()
            .dt.total_seconds()
            .dropna()
        )
        isi.extend(intervals.values)

    return isi

def compute_isi_per_patient(df_evt: pd.DataFrame) -> pd.DataFrame:
    """
    Beregn inter-seizure intervals (ISI) indenfor samme recording_uid,
    men gem dem per patient og recording.

    Returnerer DataFrame med kolonner:
        - patient_id
        - recording_id
        - recording_uid
        - isi_seconds
        - isi_hours
    """
    rows = []

    # groupby på både patient og recording_uid for overblik
    for (pid, rid, uid), df_r in df_evt.groupby(["patient_id", "recording_id", "recording_uid"]):
        df_r_sorted = df_r.sort_values("absolute_start")
        if len(df_r_sorted) < 2:
            continue

        intervals = (
            df_r_sorted["absolute_start"]
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        for val in intervals.values:
            rows.append(
                {
                    "patient_id": pid,
                    "recording_id": rid,
                    "recording_uid": uid,
                    "isi_seconds": float(val),
                    "isi_hours": float(val) / 3600.0,
                }
            )

    return pd.DataFrame(rows)

def compute_dataset_overview(df_rec: pd.DataFrame,
                             df_evt: pd.DataFrame) -> pd.DataFrame:
    """
    Returnerer et lille overblik over hele datasættet:
      - N patients, enrollments, recordings
      - total varighed
      - total seizures
      - median / IQR recording-længde
    """
    n_patients = df_rec["patient_id"].nunique()

    # enrollment = (patient, enrollment_id) kombinationer
    enr = df_rec[["patient_id", "enrollment_id"]].drop_duplicates()
    n_enrollments = len(enr)

    n_recordings = df_rec["recording_uid"].nunique()

    total_hours = df_rec["rec_duration_s"].sum() / 3600.0
    total_seizures = len(df_evt)

    rec_hours = df_rec["rec_duration_s"] / 3600.0
    median_rec_h = rec_hours.median()
    iqr_rec_h = rec_hours.quantile(0.75) - rec_hours.quantile(0.25)

    overview = pd.DataFrame([
        {"Metric": "Patients", "Value": n_patients},
        {"Metric": "Enrollments", "Value": n_enrollments},
        {"Metric": "Recordings", "Value": n_recordings},
        {"Metric": "Total hours", "Value": total_hours},
        {"Metric": "Total seizures", "Value": total_seizures},
        {"Metric": "Median rec. duration (h)", "Value": median_rec_h},
        {"Metric": "IQR rec. duration (h)", "Value": iqr_rec_h},
    ])

    return overview


def summarise_isi(isi: list[float]) -> pd.DataFrame:
    """
    Lille tabel med statistik på inter-seizure intervals (sekunder).
    """
    arr = np.asarray(isi, dtype=float)
    if arr.size == 0:
        return pd.DataFrame([{"Metric": "n_intervals", "Value": 0}])

    stats = {
        "Metric": [
            "n_intervals",
            "mean (s)",
            "median (s)",
            "IQR (s)",
            "min (s)",
            "max (s)",
        ],
        "Value": [
            int(arr.size),
            float(np.nanmean(arr)),
            float(np.nanmedian(arr)),
            float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25)),
            float(np.nanmin(arr)),
            float(np.nanmax(arr)),
        ],
    }
    return pd.DataFrame(stats)

def summarise_isi_per_patient(df_isi: pd.DataFrame) -> pd.DataFrame:
    """
    Lav en tabel med ISI-statistik per patient.
    """
    if df_isi.empty:
        return pd.DataFrame()

    grp = df_isi.groupby("patient_id")["isi_seconds"]

    df_stats = grp.agg(
        n_intervals="count",
        mean_s="mean",
        median_s="median",
        iqr_s=lambda x: np.nanpercentile(x, 75) - np.nanpercentile(x, 25),
        min_s="min",
        max_s="max",
    ).reset_index()

    # Konverter til timer også
    for col in ["mean_s", "median_s", "iqr_s", "min_s", "max_s"]:
        df_stats[col.replace("_s", "_h")] = df_stats[col] / 3600.0

    return df_stats
