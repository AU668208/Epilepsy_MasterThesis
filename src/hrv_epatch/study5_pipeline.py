
"""
Study 5 pipeline (clean module)

Purpose
-------
Compute Jeppesen-style HRV features (CSI, ModCSI, SlopeHR) from aligned RR files
(LabVIEW RR and Python/NeuroKit RR) and score seizure detection using event-based
threshold exceedances, with optional SQI masking.

This file is designed to be IMPORTABLE from a notebook, without hidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import re


# =========================
# Config / paths
# =========================

@dataclass(frozen=True)
class Study5Paths:
    rr_dir: Path
    recordings_index_csv: Path
    seizure_events_csv: Path
    window_quality_csv: Path
    threshold_xlsx: Optional[Path] = None


@dataclass(frozen=True)
class Study5Cfg:
    win_rr: int = 100
    pad_s: float = 300.0
    gap_s: float = 180.0
    responders: Optional[List[int]] = None

    # RR / algorithm selection
    algo_keep: Optional[set[str]] = None               # e.g. {"neurokit","hamilton2002"}
    recording_uid_keep: Optional[set[int]] = None      # e.g. {7, 9}

    # Feature selection for detection
    value_mode: str = "modcsi"                         # "modcsi" or "csi"
    use_sqi_masking: bool = True                       # evaluate both on/off in main table

    # Thresholding
    use_excel_thresholds: bool = False                 # force Excel thresholds (patient_id map)
    auto_thr_factor: float = 1.05
    auto_thr_prefer_segments: Tuple[str, ...] = ("first_24h", "first_12h", "first_half")
    auto_thr_use_sqi: bool = True                      # use only acceptable windows when estimating thresholds


# =========================
# Filename parsing (RR files)
# =========================

def parse_rr_filename(name_or_path) -> dict:
    """
    Parse RR filename formats like:
      - P01_R01_emrich2023_rr_aligned.csv
      - P08a_R01_neurokit_rr_aligned.csv   (optional enrollment letter a/b/c)
      - P8_R1_hamilton2002_rr_aligned.csv  (also works if not zero-padded)

    Returns dict with: patient_id, enrollment_id (None or 'a'/'b'/'c'),
                       recording_id, algo_id
    """
    name = Path(name_or_path).name

    # P<patient><optional enrollment>_R<recording>_<algo>_rr_aligned.csv
    pat = re.compile(
        r"^P(?P<pid>\d+)(?P<enroll>[abc])?_R(?P<rec>\d+)_"
        r"(?P<algo>[A-Za-z0-9]+)_rr_aligned\.csv$",
        flags=re.IGNORECASE,
    )
    m = pat.match(name)
    if not m:
        raise ValueError(f"Could not parse RR filename: {name}")

    pid = int(m.group("pid"))
    enroll = m.group("enroll")
    enroll = enroll.lower() if enroll else None
    rec = int(m.group("rec"))
    algo = m.group("algo").lower()

    return {
        "patient_id": pid,
        "enrollment_id": enroll,
        "recording_id": rec,
        "algo_id": algo,
    }



def list_rr_files(rr_dir: Path) -> List[Path]:
    rr_dir = Path(rr_dir)
    files = sorted(rr_dir.glob("P*_R*_*_rr_aligned.csv"))
    return [p for p in files if p.is_file()]


# =========================
# Data loading helpers
# =========================

def load_recordings_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"recording_uid", "patient_id", "recording_id", "enrollment_id"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"recordings_index missing columns: {miss}")

    out = df.copy()
    out["recording_uid"] = pd.to_numeric(out["recording_uid"], errors="coerce").astype("Int64")
    out["patient_id"] = pd.to_numeric(out["patient_id"], errors="coerce").astype("Int64")
    out["recording_id"] = pd.to_numeric(out["recording_id"], errors="coerce").astype("Int64")

    # enrollment_id may be NaN -> None
    out["enrollment_id"] = out["enrollment_id"].where(out["enrollment_id"].notna(), None)
    out = out.dropna(subset=["recording_uid", "patient_id", "recording_id"]).copy()
    out["recording_uid"] = out["recording_uid"].astype(int)
    out["patient_id"] = out["patient_id"].astype(int)
    out["recording_id"] = out["recording_id"].astype(int)
    return out


def load_df_seiz(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "recording_uid" not in df.columns:
        raise KeyError("seizure_events missing 'recording_uid'")

    out = df.copy()
    out["recording_uid"] = pd.to_numeric(out["recording_uid"], errors="coerce").astype("Int64")
    out["patient_id"] = pd.to_numeric(out.get("patient_id"), errors="coerce").astype("Int64")
    out = out.dropna(subset=["recording_uid"]).copy()
    out["recording_uid"] = out["recording_uid"].astype(int)

    # Choose preferred time columns later; keep everything.
    return out


def load_window_quality(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # required for merging with features
    need = {"recording_uid", "window_idx", "win_start_s", "win_end_s"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"window_quality missing columns: {miss}")

    out = df.copy()
    out["recording_uid"] = pd.to_numeric(out["recording_uid"], errors="coerce").astype("Int64")
    out["window_idx"] = pd.to_numeric(out["window_idx"], errors="coerce").astype("Int64")
    out["win_start_s"] = pd.to_numeric(out["win_start_s"], errors="coerce")
    out["win_end_s"] = pd.to_numeric(out["win_end_s"], errors="coerce")
    out = out.dropna(subset=["recording_uid", "window_idx", "win_start_s", "win_end_s"]).copy()
    out["recording_uid"] = out["recording_uid"].astype(int)
    out["window_idx"] = out["window_idx"].astype(int)

    # Derive is_acceptable if missing
    if "is_acceptable" not in out.columns:
        # common boolean flags from your Study 2 pipeline
        flags = [c for c in ["is_flatline", "is_noiseburst", "is_clipping"] if c in out.columns]
        if flags:
            bad = np.zeros(len(out), dtype=bool)
            for c in flags:
                bad |= out[c].astype(bool).to_numpy()
            out["is_acceptable"] = ~bad
        else:
            out["is_acceptable"] = True

    return out


def _pick_seiz_cols(df_seiz: pd.DataFrame) -> Tuple[str, str]:
    """
    Pick best available seizure time columns.
    Priority:
      1) t0_trim / t1_trim
      2) t0_clinical_trim / t1_clinical_trim
      3) t0_clinical / t1_clinical
      4) t0 / t1
    """
    candidates = [
        ("t0_trim", "t1_trim"),
        ("t0_clinical_trim", "t1_clinical_trim"),
        ("t0_clinical", "t1_clinical"),
        ("t0", "t1"),
    ]
    for a, b in candidates:
        if a in df_seiz.columns and b in df_seiz.columns:
            return a, b
    raise KeyError(f"Could not find seizure time columns in df_seiz. Have: {list(df_seiz.columns)}")


def attach_window_overlaps_seizure(windows_df: pd.DataFrame, df_seiz: pd.DataFrame) -> pd.DataFrame:
    """
    Add window_overlaps_seizure using available seizure times (trim-aware if present).
    """
    t0_col, t1_col = _pick_seiz_cols(df_seiz)

    w = windows_df.copy()
    if "window_overlaps_seizure" in w.columns:
        return w

    # prepare seizures per recording
    seiz = df_seiz[["recording_uid", t0_col, t1_col]].copy()
    seiz["recording_uid"] = pd.to_numeric(seiz["recording_uid"], errors="coerce")
    seiz[t0_col] = pd.to_numeric(seiz[t0_col], errors="coerce")
    seiz[t1_col] = pd.to_numeric(seiz[t1_col], errors="coerce")
    seiz = seiz.dropna(subset=["recording_uid", t0_col, t1_col]).copy()
    seiz["recording_uid"] = seiz["recording_uid"].astype(int)

    # initialise
    w["window_overlaps_seizure"] = False

    for rid, gwin in w.groupby("recording_uid", sort=False):
        gseiz = seiz[seiz["recording_uid"] == rid]
        if gseiz.empty:
            continue
        ws = gwin["win_start_s"].to_numpy(dtype=float)
        we = gwin["win_end_s"].to_numpy(dtype=float)
        ov = np.zeros(len(gwin), dtype=bool)

        t0s = gseiz[t0_col].to_numpy(dtype=float)
        t1s = gseiz[t1_col].to_numpy(dtype=float)

        for a, b in zip(t0s, t1s):
            ov |= (ws < b) & (we > a)

        w.loc[gwin.index, "window_overlaps_seizure"] = ov

    return w


# =========================
# Jeppesen-style HRV features (validated)
# =========================

def median_prefilter_7(rr_s: np.ndarray) -> np.ndarray:
    """7-beat median filter on RR (seconds)."""
    rr = np.asarray(rr_s, dtype=float)
    if rr.size < 7:
        return rr.copy()
    pad = 3
    x = np.pad(rr, (pad, pad), mode="edge")
    out = np.empty_like(rr)
    for i in range(rr.size):
        out[i] = np.median(x[i:i+7])
    return out


# Backwards-compatible aliases (if you referenced these names elsewhere)
def rolling_median(x: np.ndarray, k: int = 7) -> np.ndarray:
    if k != 7:
        raise ValueError("rolling_median only supports k=7 in this pipeline (Jeppesen MA7).")
    return median_prefilter_7(x)


def poincare_sd1_sd2_vectorized(rr_ms_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    rr_ms_windows: shape (nwin, win_rr)
    Returns SD1, SD2 arrays (ms).
    NeuroKit-compatible:
      SD1 = std(diff)/sqrt(2)
      SD2 = std(sum)/sqrt(2)
    """
    if rr_ms_windows.size == 0:
        return np.array([]), np.array([])
    x = rr_ms_windows[:, :-1]
    y = rr_ms_windows[:, 1:]
    d = y - x
    s = y + x
    sd1 = np.std(d, axis=1, ddof=1) / np.sqrt(2.0)
    sd2 = np.std(s, axis=1, ddof=1) / np.sqrt(2.0)
    return sd1, sd2


def csi_from_sd(sd1: np.ndarray, sd2: np.ndarray) -> np.ndarray:
    return sd2 / np.maximum(sd1, 1e-12)


def modcsi_modified_from_sd(sd1: np.ndarray, sd2: np.ndarray) -> np.ndarray:
    """
    NeuroKit's "CSI_Modified" (as in your validation script):
      (4*SD2)^2 / (4*SD1) = 4*SD2^2 / SD1
    """
    return (4.0 * (sd2 ** 2)) / np.maximum(sd1, 1e-12)


def slope_hr_ls_abs(rr_s_windows: np.ndarray) -> np.ndarray:
    """
    Least-squares slope of HR (BPM) vs time (s) inside each window.
    Returns abs(slope) in BPM/s (Jeppesen uses abs).
    """
    if rr_s_windows.size == 0:
        return np.array([])

    rr = np.asarray(rr_s_windows, dtype=float)
    if rr.ndim != 2:
        raise ValueError("rr_s_windows must be 2D (nwin, win_rr)")

    nwin, win_rr = rr.shape

    # time axis = cumulative RR from window start
    t = np.cumsum(rr, axis=1)  # seconds
    hr = 60.0 / np.maximum(rr, 1e-12)  # BPM

    # center time per window to improve numeric stability
    t_mean = t.mean(axis=1, keepdims=True)
    h_mean = hr.mean(axis=1, keepdims=True)

    dt = t - t_mean
    dh = hr - h_mean

    num = np.sum(dt * dh, axis=1)
    den = np.sum(dt * dt, axis=1) + 1e-12
    slope = num / den
    return np.abs(slope)


# Alias for convenience if your notebook referenced slope_ls()
def slope_ls(x: np.ndarray, y: np.ndarray) -> float:
    """Generic least-squares slope (scalar)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return np.nan
    xm = x.mean()
    ym = y.mean()
    den = np.sum((x - xm) ** 2)
    if den <= 0:
        return np.nan
    return float(np.sum((x - xm) * (y - ym)) / den)


def slope_ls_abs(x: np.ndarray, y: np.ndarray) -> float:
    s = slope_ls(x, y)
    return float(np.abs(s)) if np.isfinite(s) else s


def compute_jeppesen_features(rr_s: np.ndarray, *, win_rr: int = 100) -> pd.DataFrame:
    """
    Compute validated Jeppesen-style features on a single RR series.

    Output columns:
      t_end_s
      CSI100
      ModCSI100_filt
      SlopeHR100_abs_bpm_per_s
      CSI100_x_SlopeHR
      ModCSI100filt_x_SlopeHR
    """
    rr_s = np.asarray(rr_s, dtype=float)
    rr_s = rr_s[np.isfinite(rr_s)]
    if rr_s.size < win_rr:
        return pd.DataFrame(columns=[
            "t_end_s","CSI100","ModCSI100_filt","SlopeHR100_abs_bpm_per_s",
            "CSI100_x_SlopeHR","ModCSI100filt_x_SlopeHR"
        ])

    rr_ms = rr_s * 1000.0

    # rolling windows (step=1)
    nwin = rr_s.size - win_rr + 1
    # shape (nwin, win_rr) using striding
    rr_s_w = np.lib.stride_tricks.sliding_window_view(rr_s, window_shape=win_rr)
    rr_ms_w = np.lib.stride_tricks.sliding_window_view(rr_ms, window_shape=win_rr)

    # CSI uses raw RR (per Jeppesen note 3)
    sd1_raw, sd2_raw = poincare_sd1_sd2_vectorized(rr_ms_w)
    csi = csi_from_sd(sd1_raw, sd2_raw)

    # ModCSI and Slope use MA7-filtered RR
    rr_s_f = median_prefilter_7(rr_s)
    rr_ms_f = rr_s_f * 1000.0
    rr_s_f_w = np.lib.stride_tricks.sliding_window_view(rr_s_f, window_shape=win_rr)
    rr_ms_f_w = np.lib.stride_tricks.sliding_window_view(rr_ms_f, window_shape=win_rr)

    sd1_f, sd2_f = poincare_sd1_sd2_vectorized(rr_ms_f_w)
    modcsi = modcsi_modified_from_sd(sd1_f, sd2_f)

    slope_hr = slope_hr_ls_abs(rr_s_f_w)

    # time for each window = end-time (seconds from start)
    t_end = np.cumsum(rr_s)[win_rr-1:]

    df = pd.DataFrame({
        "t_end_s": t_end.astype(float),
        "CSI100": csi.astype(float),
        "ModCSI100_filt": modcsi.astype(float),
        "SlopeHR100_abs_bpm_per_s": slope_hr.astype(float),
    })
    df["CSI100_x_SlopeHR"] = df["CSI100"] * df["SlopeHR100_abs_bpm_per_s"]
    df["ModCSI100filt_x_SlopeHR"] = df["ModCSI100_filt"] * df["SlopeHR100_abs_bpm_per_s"]
    return df


# =========================
# Build df_feat5 from RR folder
# =========================

def _lookup_recording_uid(df_rec: pd.DataFrame, *, patient_id: int, enrollment_id: Optional[str], recording_id: int) -> Optional[int]:
    m = (
        (df_rec["patient_id"] == int(patient_id)) &
        (df_rec["recording_id"] == int(recording_id))
    )
    # enrollment_id can be None/NaN
    if enrollment_id is None:
        m &= df_rec["enrollment_id"].isna() | (df_rec["enrollment_id"].astype(object).isna()) | (df_rec["enrollment_id"] == None)
    else:
        m &= (df_rec["enrollment_id"].astype(str) == str(enrollment_id))
    hit = df_rec.loc[m, "recording_uid"]
    if hit.empty:
        return None
    return int(hit.iloc[0])


def build_feat5_from_rr_dir(
    rr_dir: Path,
    df_rec: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    win_rr: int = 100,
    algo_keep: Optional[set[str]] = None,
    recording_uid_keep: Optional[set[int]] = None,
) -> pd.DataFrame:
    """
    Build long feature table with one row per RR-window.

    Output includes:
      rr_source in {"labview","python"}
      algo_id
      recording_uid, patient_id
      t_s (end time)
      value columns (CSI/ModCSI/Slope products)
      window_idx, win_start_s, win_end_s, is_acceptable, window_overlaps_seizure
    """
    rr_dir = Path(rr_dir)
    files = list_rr_files(rr_dir)

    rows = []
    for f in files:
        meta = parse_rr_filename(f)
        algo = meta["algo_id"]
        if algo_keep is not None and algo not in set(a.lower() for a in algo_keep):
            continue

        rid = _lookup_recording_uid(
            df_rec,
            patient_id=meta["patient_id"],
            enrollment_id=meta["enrollment_id"],
            recording_id=meta["recording_id"],
        )
        if rid is None:
            continue
        if recording_uid_keep is not None and rid not in recording_uid_keep:
            continue

        df_rr = pd.read_csv(f)
        if "RR_labview_s" not in df_rr.columns or "RR_python_s" not in df_rr.columns:
            raise KeyError(f"{f.name} must contain RR_labview_s and RR_python_s")

        rr_lab = pd.to_numeric(df_rr["RR_labview_s"], errors="coerce").to_numpy(dtype=float)
        rr_py  = pd.to_numeric(df_rr["RR_python_s"], errors="coerce").to_numpy(dtype=float)

        for rr_source, rr in [("labview", rr_lab), ("python", rr_py)]:
            feat = compute_jeppesen_features(rr, win_rr=win_rr)
            if feat.empty:
                continue

            feat = feat.rename(columns={"t_end_s": "t_s"}).copy()
            feat["recording_uid"] = int(rid)
            feat["patient_id"] = int(meta["patient_id"])
            feat["algo_id"] = ("LabVIEW" if rr_source == "labview" else algo)
            feat["rr_source"] = rr_source

            # attach window_idx by 10s windows (from windows_df) using window_idx mapping:
            # assume windows_df uses 10 s windows starting at 0
            # best: compute idx from t_s and merge on (recording_uid, window_idx)
            # if windows_df uses variable window sizes, still works if window_idx aligns.
            # infer window_s from first row
            wsub = windows_df[windows_df["recording_uid"] == int(rid)]
            if wsub.empty:
                continue
            # compute idx using win_start/end (more robust): find nearest window_idx by floor(t_s / window_s)
            # assume contiguous windows
            window_s = float(wsub["win_end_s"].iloc[0] - wsub["win_start_s"].iloc[0])
            feat["window_idx"] = np.floor(feat["t_s"].to_numpy(dtype=float) / window_s).astype(int)

            feat = feat.merge(
                wsub[["recording_uid","window_idx","win_start_s","win_end_s","is_acceptable","window_overlaps_seizure"]],
                on=["recording_uid","window_idx"],
                how="left",
                validate="many_to_one"
            )

            rows.append(feat)

    if not rows:
        return pd.DataFrame()

    df_feat5 = pd.concat(rows, ignore_index=True)

    # clean up
    df_feat5["recording_uid"] = pd.to_numeric(df_feat5["recording_uid"], errors="coerce").astype("Int64")
    df_feat5["patient_id"] = pd.to_numeric(df_feat5["patient_id"], errors="coerce").astype("Int64")
    df_feat5["t_s"] = pd.to_numeric(df_feat5["t_s"], errors="coerce")
    df_feat5 = df_feat5.dropna(subset=["recording_uid","patient_id","t_s"]).copy()
    df_feat5["recording_uid"] = df_feat5["recording_uid"].astype(int)
    df_feat5["patient_id"] = df_feat5["patient_id"].astype(int)

    # ensure boolean columns exist
    if "is_acceptable" not in df_feat5.columns:
        df_feat5["is_acceptable"] = True
    if "window_overlaps_seizure" not in df_feat5.columns:
        df_feat5["window_overlaps_seizure"] = False

    return df_feat5


# =========================
# Threshold maps (Excel + auto)
# =========================

def load_thr_maps_from_excel(path: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Expect an Excel with at least columns:
      patient_id
      thr_modcsi (ModCSI100_filt x SlopeHR)   OR a close name
      thr_csi    (CSI100 x SlopeHR)           OR a close name

    Returns (thr_map_modcsi, thr_map_csi) as patient_id->float.
    """
    df = pd.read_excel(path)

    # flexible column matching
    cols = {c.lower(): c for c in df.columns}
    if "patient_id" not in cols:
        raise KeyError(f"threshold_xlsx missing patient_id. Columns: {list(df.columns)}")

    pid_col = cols["patient_id"]

    def _find(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    mod_col = _find("thr_modcsi", "thr_modcsi100", "thr_modcsi100_filt_x_slope", "modcsi", "modcsi_threshold")
    csi_col = _find("thr_csi", "thr_csi100", "thr_csi100_x_slope", "csi", "csi_threshold")

    if mod_col is None or csi_col is None:
        raise KeyError(f"Could not find thr columns in Excel. Columns: {list(df.columns)}")

    out = df[[pid_col, mod_col, csi_col]].copy()
    out[pid_col] = pd.to_numeric(out[pid_col], errors="coerce")
    out[mod_col] = pd.to_numeric(out[mod_col], errors="coerce")
    out[csi_col] = pd.to_numeric(out[csi_col], errors="coerce")
    out = out.dropna(subset=[pid_col]).copy()
    out[pid_col] = out[pid_col].astype(int)

    thr_mod = {int(pid): float(v) for pid, v in zip(out[pid_col], out[mod_col]) if np.isfinite(v)}
    thr_csi = {int(pid): float(v) for pid, v in zip(out[pid_col], out[csi_col]) if np.isfinite(v)}
    return thr_mod, thr_csi


def build_auto_thresholds_105pct(
    df_feat5: pd.DataFrame,
    df_seiz: pd.DataFrame,
    *,
    value_col: str,
    factor: float = 1.05,
    prefer_segments: Tuple[str, ...] = ("first_24h", "first_12h", "first_half"),
    use_sqi: bool = True,
    excel_fallback: Optional[Dict[int, float]] = None,
) -> Tuple[Dict[int, float], pd.DataFrame]:
    """
    Compute patient-specific threshold = factor * max(value_col) in chosen segment.
    Segment selection is per patient, with seizure-aware fallback:
      - If ANY seizure overlaps the segment -> try next segment.
    Segments:
      first_12h: [0, 43200]
      first_24h: [0, 86400]
      first_half: [0, 0.5*rec_duration] (approximated from max t_s within patient+recording)
    Returns:
      thr_map (patient_id -> threshold)
      meta_df (one row per patient describing the choice)
    """
    if value_col not in df_feat5.columns:
        raise KeyError(f"value_col {value_col} missing from df_feat5")

    t0_col, t1_col = _pick_seiz_cols(df_seiz)

    seiz = df_seiz[["recording_uid","patient_id", t0_col, t1_col]].copy()
    seiz["recording_uid"] = pd.to_numeric(seiz["recording_uid"], errors="coerce")
    seiz["patient_id"] = pd.to_numeric(seiz["patient_id"], errors="coerce")
    seiz[t0_col] = pd.to_numeric(seiz[t0_col], errors="coerce")
    seiz[t1_col] = pd.to_numeric(seiz[t1_col], errors="coerce")
    seiz = seiz.dropna(subset=["recording_uid","patient_id", t0_col, t1_col]).copy()
    seiz["recording_uid"] = seiz["recording_uid"].astype(int)
    seiz["patient_id"] = seiz["patient_id"].astype(int)

    base = df_feat5.copy()
    if use_sqi and "is_acceptable" in base.columns:
        base = base[base["is_acceptable"].astype(bool)].copy()

    thr_map: Dict[int, float] = {}
    meta_rows = []

    for pid, g in base.groupby("patient_id", sort=False):
        # choose a "representative" recording_uid for segment definitions:
        # use the first one (as in your prints), but record it for transparency.
        rid_used = int(g["recording_uid"].iloc[0])
        g_r = g[g["recording_uid"] == rid_used].copy()
        if g_r.empty:
            g_r = g.copy()
            rid_used = int(g_r["recording_uid"].iloc[0])

        # approximate recording duration from max t_s in this recording
        rec_dur = float(np.nanmax(g_r["t_s"].to_numpy(dtype=float)))
        half_end = rec_dur * 0.5

        seg_defs = {
            "first_12h": 43200.0,
            "first_24h": 86400.0,
            "first_half": half_end,
        }

        # seizures in that recording (trim-aware)
        seiz_r = seiz[seiz["recording_uid"] == rid_used]
        intervals = []
        if not seiz_r.empty:
            intervals = list(zip(seiz_r[t0_col].to_numpy(dtype=float), seiz_r[t1_col].to_numpy(dtype=float)))

        chosen = None
        seg_start = 0.0
        seg_end = None
        max_val = np.nan

        for mode in prefer_segments:
            end = float(seg_defs.get(mode, half_end))
            if end <= 0:
                continue

            # if any seizure overlaps [0, end] then reject
            has_seiz = False
            for a, b in intervals:
                if (0.0 < b) and (end > a):
                    has_seiz = True
                    break
            if has_seiz:
                continue

            cand = g_r[(g_r["t_s"] >= 0.0) & (g_r["t_s"] <= end)]
            if cand.empty:
                continue
            mv = float(np.nanmax(pd.to_numeric(cand[value_col], errors="coerce")))
            if not np.isfinite(mv):
                continue

            chosen = mode
            seg_end = end
            max_val = mv
            break

        fallback = False
        if chosen is None:
            # no seizure-free segment found
            fallback = True
            if excel_fallback is not None and int(pid) in excel_fallback:
                thr = float(excel_fallback[int(pid)])
                thr_map[int(pid)] = thr
                meta_rows.append({
                    "patient_id": int(pid),
                    "mode_used": "excel_fallback",
                    "recording_uid_used": rid_used,
                    "segment_start_s": np.nan,
                    "segment_end_s": np.nan,
                    "max_value_in_segment": np.nan,
                    "factor": factor,
                    "threshold": thr,
                    "fallback_used": True,
                })
                continue
            else:
                # final fallback: global max in this recording
                cand = g_r.copy()
                mv = float(np.nanmax(pd.to_numeric(cand[value_col], errors="coerce")))
                chosen = "global_max"
                seg_end = float(np.nanmax(g_r["t_s"]))
                max_val = mv

        thr = float(factor * max_val)
        thr_map[int(pid)] = thr
        meta_rows.append({
            "patient_id": int(pid),
            "mode_used": chosen,
            "recording_uid_used": rid_used,
            "segment_start_s": float(seg_start),
            "segment_end_s": float(seg_end) if seg_end is not None else np.nan,
            "max_value_in_segment": float(max_val) if np.isfinite(max_val) else np.nan,
            "factor": float(factor),
            "threshold": thr,
            "fallback_used": bool(fallback),
        })

    meta_df = pd.DataFrame(meta_rows).sort_values("patient_id").reset_index(drop=True)
    return thr_map, meta_df


# =========================
# Event scoring (same as your earlier function)
# =========================

def _extract_events_from_series(t: np.ndarray, x: np.ndarray, *, gap_s: float = 60.0) -> list[dict]:
    if t.size == 0:
        return []
    dt = np.diff(t)
    splits = np.where(dt > gap_s)[0] + 1
    groups = np.split(np.arange(t.size), splits)
    events = []
    for idx in groups:
        tt = t[idx]
        xx = x[idx]
        k = int(np.argmax(xx))
        events.append({
            "t_start": float(tt[0]),
            "t_end": float(tt[-1]),
            "duration_s": float(tt[-1] - tt[0]),
            "t_peak": float(tt[k]),
            "peak_value": float(xx[k]),
            "n_points": int(len(idx)),
        })
    return events


def build_event_list(
    df_feat: pd.DataFrame,
    *,
    value_col: str,
    thr_col: str,
    time_col: str = "t_s",
    gap_s: float = 60.0,
) -> pd.DataFrame:
    need = {"recording_uid", time_col, value_col, thr_col}
    miss = need - set(df_feat.columns)
    if miss:
        raise KeyError(f"df_feat missing columns: {miss}")

    out = []
    df = df_feat.copy()
    df["recording_uid"] = pd.to_numeric(df["recording_uid"], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[thr_col] = pd.to_numeric(df[thr_col], errors="coerce")
    df = df.dropna(subset=["recording_uid", time_col, value_col, thr_col]).copy()
    df["recording_uid"] = df["recording_uid"].astype(int)

    for rid, g in df.groupby("recording_uid", sort=False):
        thr = float(g[thr_col].iloc[0])
        gg = g[g[value_col] > thr].copy()
        if gg.empty:
            continue
        gg = gg.sort_values(time_col, kind="mergesort")

        t = gg[time_col].to_numpy(dtype=float)
        x = gg[value_col].to_numpy(dtype=float)

        events = _extract_events_from_series(t, x, gap_s=gap_s)
        for e in events:
            out.append({"recording_uid": int(rid), **e})

    if not out:
        return pd.DataFrame(columns=["recording_uid","t_start","t_end","duration_s","t_peak","peak_value","n_points"])
    return pd.DataFrame(out)


def _analyzable_hours_from_windows(windows_df: pd.DataFrame, *, use_sqi: bool) -> float:
    need = {"recording_uid","win_start_s","win_end_s","is_acceptable"}
    miss = need - set(windows_df.columns)
    if miss:
        raise KeyError(f"windows_df missing columns for time accounting: {miss}")

    w = windows_df[["recording_uid","win_start_s","win_end_s","is_acceptable"]].drop_duplicates().copy()
    if use_sqi:
        w = w[w["is_acceptable"].astype(bool)]
    dur_s = (pd.to_numeric(w["win_end_s"], errors="coerce") - pd.to_numeric(w["win_start_s"], errors="coerce")).clip(lower=0).sum()
    return float(dur_s / 3600.0)


def score_pipeline_events(
    df_feat: pd.DataFrame,
    df_seiz: pd.DataFrame,
    *,
    value_col: str,
    thr_col: str,
    pad_s: float = 300.0,
    gap_s: float = 60.0,
    use_sqi: bool = False,
    windows_df_for_time: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Event-based scoring (per your earlier approach).
    Uses seizure times chosen by _pick_seiz_cols().
    """
    t0_col, t1_col = _pick_seiz_cols(df_seiz)

    feat = df_feat.copy()
    if use_sqi and "is_acceptable" in feat.columns:
        feat = feat[feat["is_acceptable"].astype(bool)].copy()

    used_rids = (
        pd.to_numeric(feat["recording_uid"], errors="coerce").dropna().astype(int).unique()
        if "recording_uid" in feat.columns else np.array([], dtype=int)
    )

    # analyzable hours from windows (not from feature rows)
    if windows_df_for_time is None:
        if {"win_start_s","win_end_s"}.issubset(feat.columns):
            windows_df_for_time = feat[["recording_uid","win_start_s","win_end_s","is_acceptable"]].drop_duplicates()
        else:
            raise KeyError("Need windows_df_for_time or window columns in df_feat to compute FAR.")
    total_h = _analyzable_hours_from_windows(windows_df_for_time, use_sqi=use_sqi)

    seiz = df_seiz.copy()
    seiz["recording_uid"] = pd.to_numeric(seiz["recording_uid"], errors="coerce")
    seiz[t0_col] = pd.to_numeric(seiz[t0_col], errors="coerce")
    seiz[t1_col] = pd.to_numeric(seiz[t1_col], errors="coerce")
    seiz = seiz.dropna(subset=["recording_uid", t0_col, t1_col]).copy()
    seiz["recording_uid"] = seiz["recording_uid"].astype(int)

    n_seiz_total = int(len(seiz))
    n_seiz_total_used = int(seiz[seiz["recording_uid"].isin(used_rids)].shape[0])

    df_events = build_event_list(
        feat, value_col=value_col, thr_col=thr_col, time_col="t_s", gap_s=gap_s
    )

    if df_events.empty:
        return dict(
            n_seiz_total=n_seiz_total,
            n_seiz_total_used=n_seiz_total_used,
            n_seiz_detected=0,
            recall_total=(0.0 if n_seiz_total else np.nan),
            recall_used=(0.0 if n_seiz_total_used else np.nan),
            FP_events=0,
            FAR_per_h=(0.0 if total_h > 0 else np.nan),
            total_h=total_h,
            n_events=0,
            n_recordings_used=int(len(used_rids)),
        )

    detected = 0
    fp_events = 0

    for rid, ev in df_events.groupby("recording_uid", sort=False):
        seiz_r = seiz[seiz["recording_uid"] == rid]
        intervals = []
        if not seiz_r.empty:
            t0s = seiz_r[t0_col].to_numpy(dtype=float) - pad_s
            t1s = seiz_r[t1_col].to_numpy(dtype=float) + pad_s
            intervals = list(zip(t0s, t1s))

        if intervals:
            for _, s in seiz_r.iterrows():
                t0p = float(s[t0_col]) - pad_s
                t1p = float(s[t1_col]) + pad_s
                hit = ((ev["t_end"] >= t0p) & (ev["t_start"] <= t1p)).any()
                detected += int(hit)

        if intervals:
            inside_any = np.zeros(len(ev), dtype=bool)
            e_start = ev["t_start"].to_numpy(dtype=float)
            e_end = ev["t_end"].to_numpy(dtype=float)
            for (a, b) in intervals:
                inside_any |= (e_end >= a) & (e_start <= b)
            fp_events += int((~inside_any).sum())
        else:
            fp_events += int(len(ev))

    recall_total = detected / n_seiz_total if n_seiz_total else np.nan
    recall_used = detected / n_seiz_total_used if n_seiz_total_used else np.nan
    far = fp_events / total_h if total_h > 0 else np.nan

    return dict(
        n_seiz_total=n_seiz_total,
        n_seiz_total_used=n_seiz_total_used,
        n_seiz_detected=int(detected),
        recall_total=float(recall_total) if np.isfinite(recall_total) else recall_total,
        recall_used=float(recall_used) if np.isfinite(recall_used) else recall_used,
        FP_events=int(fp_events),
        FAR_per_h=float(far) if np.isfinite(far) else far,
        total_h=float(total_h),
        n_events=int(len(df_events)),
        n_recordings_used=int(len(used_rids)),
    )


# =========================
# Main tables (+ responder split)
# =========================

def add_responder_label(df: pd.DataFrame, responders: List[int], *, pid_col: str = "patient_id") -> pd.DataFrame:
    out = df.copy()
    out[pid_col] = pd.to_numeric(out[pid_col], errors="coerce")
    out["is_responder"] = out[pid_col].astype("Int64").isin(list(map(int, responders)))
    return out


def build_event_main_table(
    df_feat5: pd.DataFrame,
    df_seiz: pd.DataFrame,
    windows_df: pd.DataFrame,
    *,
    pad_s: float,
    gap_s: float,
    value_mode: str,
    collapse_labview_algo: bool = True,
    use_sqi_rows: Tuple[bool, ...] = (False, True),
) -> pd.DataFrame:
    """
    One row per (RR_source, Algorithm, SQI).
    """
    if value_mode not in {"modcsi", "csi"}:
        raise ValueError("value_mode must be 'modcsi' or 'csi'")

    value_col = "ModCSI100filt_x_SlopeHR" if value_mode == "modcsi" else "CSI100_x_SlopeHR"
    thr_col = "thr_modcsi" if value_mode == "modcsi" else "thr_csi"

    need = {"rr_source","algo_id","recording_uid","win_start_s","win_end_s","is_acceptable", value_col, thr_col, "t_s"}
    miss = need - set(df_feat5.columns)
    if miss:
        raise KeyError(f"df_feat5 missing columns: {miss}")

    rows = []
    for rr_src, sub in df_feat5.groupby("rr_source", sort=False):
        sub = sub.copy()
        if rr_src == "labview" and collapse_labview_algo:
            sub["algo_id"] = "LabVIEW"

        for algo, g in sub.groupby("algo_id", sort=False):
            for use_sqi in use_sqi_rows:
                res = score_pipeline_events(
                    g,
                    df_seiz,
                    value_col=value_col,
                    thr_col=thr_col,
                    pad_s=pad_s,
                    gap_s=gap_s,
                    use_sqi=use_sqi,
                    windows_df_for_time=windows_df,
                )
                rows.append({
                    "RR_source": ("LabVIEW" if rr_src=="labview" else "Python"),
                    "Algorithm": ("—" if (rr_src=="labview" and collapse_labview_algo) else str(algo)),
                    "SQI": ("on" if use_sqi else "off"),
                    **res
                })

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(["RR_source","Algorithm","SQI"]).reset_index(drop=True)
    return df_out


def build_event_tables_with_responder_split(
    df_feat5: pd.DataFrame,
    df_seiz: pd.DataFrame,
    windows_df: pd.DataFrame,
    responders: List[int],
    *,
    pad_s: float,
    gap_s: float,
    value_mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_main = build_event_main_table(
        df_feat5, df_seiz, windows_df,
        pad_s=pad_s, gap_s=gap_s, value_mode=value_mode,
        collapse_labview_algo=True,
    )

    feat2 = add_responder_label(df_feat5, responders)
    seiz2 = add_responder_label(df_seiz, responders)

    parts = []
    for grp, gfeat in feat2.groupby("is_responder", sort=False):
        gseiz = seiz2[seiz2["is_responder"] == grp].copy()
        tab = build_event_main_table(
            gfeat, gseiz, windows_df,
            pad_s=pad_s, gap_s=gap_s, value_mode=value_mode,
            collapse_labview_algo=True,
        )
        tab.insert(0, "is_responder", bool(grp))
        parts.append(tab)

    df_split = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df_main, df_split


# =========================
# Top-level runner
# =========================

@dataclass
class Study5Outputs:
    df_feat5: pd.DataFrame
    df_main: pd.DataFrame
    df_split: pd.DataFrame
    thr_meta_modcsi: pd.DataFrame
    thr_meta_csi: pd.DataFrame
    thr_map_modcsi: Dict[int, float]
    thr_map_csi: Dict[int, float]


def run_study5(paths: Study5Paths, cfg: Study5Cfg) -> Study5Outputs:
    """
    Main entry point.
    Returns df_feat5 (features), df_main (overall table), df_split (responder split),
    and threshold meta/maps.

    Notes
    -----
    - Uses window_quality_csv as the *time base* for SQI masking and FAR denominator.
    - Uses seizure_events_csv time columns automatically (prefers trim-aware if present).
    """
    df_rec = load_recordings_index(paths.recordings_index_csv)
    df_seiz = load_df_seiz(paths.seizure_events_csv)

    windows_df = load_window_quality(paths.window_quality_csv)
    windows_df = attach_window_overlaps_seizure(windows_df, df_seiz)

    # Excel thresholds (optional)
    thr_excel_mod, thr_excel_csi = ({}, {})
    if cfg.use_excel_thresholds and paths.threshold_xlsx is not None and Path(paths.threshold_xlsx).exists():
        thr_excel_mod, thr_excel_csi = load_thr_maps_from_excel(Path(paths.threshold_xlsx))


    # Build df_feat5
    df_feat5 = build_feat5_from_rr_dir(
        paths.rr_dir,
        df_rec=df_rec,
        windows_df=windows_df,
        win_rr=cfg.win_rr,
        algo_keep=cfg.algo_keep,
        recording_uid_keep=cfg.recording_uid_keep,
    )
    if df_feat5.empty:
        raise RuntimeError("df_feat5 ended up empty. Check rr_dir parsing / recordings_index mapping.")

    # Threshold maps
    if cfg.use_excel_thresholds:
        thr_map_modcsi = thr_excel_mod
        thr_map_csi = thr_excel_csi
        thr_meta_modcsi = pd.DataFrame([{"note": "Excel thresholds used"}])
        thr_meta_csi = pd.DataFrame([{"note": "Excel thresholds used"}])
    else:
        thr_map_modcsi, thr_meta_modcsi = build_auto_thresholds_105pct(
            df_feat5[df_feat5["rr_source"] == "labview"],  # estimate thresholds on LabVIEW RR by default
            df_seiz,
            value_col="ModCSI100filt_x_SlopeHR",
            factor=cfg.auto_thr_factor,
            prefer_segments=cfg.auto_thr_prefer_segments,
            use_sqi=cfg.auto_thr_use_sqi,
            excel_fallback=thr_excel_mod if thr_excel_mod else None,
        )
        thr_map_csi, thr_meta_csi = build_auto_thresholds_105pct(
            df_feat5[df_feat5["rr_source"] == "labview"],
            df_seiz,
            value_col="CSI100_x_SlopeHR",
            factor=cfg.auto_thr_factor,
            prefer_segments=cfg.auto_thr_prefer_segments,
            use_sqi=cfg.auto_thr_use_sqi,
            excel_fallback=thr_excel_csi if thr_excel_csi else None,
        )

    # Attach thresholds
    df_feat5 = df_feat5.copy()
    df_feat5["thr_modcsi"] = df_feat5["patient_id"].map(thr_map_modcsi)
    df_feat5["thr_csi"] = df_feat5["patient_id"].map(thr_map_csi)

    # Tables
    if cfg.responders:
        df_main, df_split = build_event_tables_with_responder_split(
            df_feat5, df_seiz, windows_df,
            responders=cfg.responders,
            pad_s=cfg.pad_s,
            gap_s=cfg.gap_s,
            value_mode=cfg.value_mode,
        )
    else:
        df_main = build_event_main_table(
            df_feat5, df_seiz, windows_df,
            pad_s=cfg.pad_s,
            gap_s=cfg.gap_s,
            value_mode=cfg.value_mode,
            collapse_labview_algo=True,
        )
        df_split = pd.DataFrame()

    return Study5Outputs(
        df_feat5=df_feat5,
        df_main=df_main,
        df_split=df_split,
        thr_meta_modcsi=thr_meta_modcsi,
        thr_meta_csi=thr_meta_csi,
        thr_map_modcsi=thr_map_modcsi,
        thr_map_csi=thr_map_csi,
    )
