# src/hrv_epatch/io/labview.py
from __future__ import annotations
from typing import Iterable, Optional
import re
from datetime import datetime
import numpy as np
import pandas as pd

def read_labview_rr(path: str, skiprows: int = 22) -> np.ndarray:
    """
    Read RR intervals from a LabVIEW .lvm (or similar text) file and return seconds as float array.

    - Accepts tab-separated with decimal comma or auto-detected separators.
    - Chooses the most plausible RR column by median range (~0.1–5 s).
    - Auto-converts from ms/µs if needed (based on the median value).
    """
    try:
        df = pd.read_csv(path, sep="\t", engine="python", skiprows=skiprows, header=0, decimal=",")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", skiprows=skiprows, header=0)
        df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce"))

    cols = [c for c in df.columns if str(c).lower() not in ("x_value", "xvalue", "comment")]
    rr = None
    if "Untitled" in df.columns:
        rr = pd.to_numeric(df["Untitled"], errors="coerce").dropna().to_numpy(float)

    if rr is None:
        for c in cols:
            v = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float)
            if v.size:
                med = float(np.nanmedian(v))
                if 0.1 < med < 5.0:
                    rr = v
                    break
        if rr is None and cols:
            rr = pd.to_numeric(df[cols[0]], errors="coerce").dropna().to_numpy(float)

    if rr is None or rr.size == 0:
        raise ValueError("Could not find an RR column in the provided LabVIEW file.")

    med = float(np.nanmedian(rr))
    if med > 10000:       # microseconds
        rr = rr / 1_000_000.0
    elif med > 5:         # milliseconds
        rr = rr / 1_000.0
    return rr

def read_header_datetime_lvm(path: str, default_date_fmt: str = "%Y/%m/%d") -> Optional[datetime]:
    """
    Parse 'Date' and 'Time' from the .lvm header (before ***End_of_Header***).
    Returns naive datetime if found; else None.
    """
    date_val, time_val = None, None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ls = line.strip()
            if ls.startswith("***End_of_Header***"):
                break
            low = ls.lower()
            if low.startswith("date\t") and date_val is None:
                date_val = ls.split("\t", 1)[1].strip()
            elif "time" in low and "time_pref" not in low and low.startswith("time\t") and time_val is None:
                time_val = ls.split("\t", 1)[1].strip()

    if not time_val:
        return None

    m = re.match(r"^(\d{2}:\d{2}:\d{2})[,.](\d+)$", time_val)
    if m:
        hhmmss, frac = m.group(1), m.group(2)
        if len(frac) > 6 and int(frac[6]) >= 5:
            frac6 = str(int(frac[:6]) + 1).zfill(6)
        else:
            frac6 = frac[:6].ljust(6, "0")
        t_dt = datetime.strptime(f"{hhmmss}.{frac6}", "%H:%M:%S.%f")
    else:
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", time_val):
            return None
        t_dt = datetime.strptime(time_val, "%H:%M:%S")

    if date_val:
        for fmt in (default_date_fmt, "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                d = datetime.strptime(date_val, fmt).date()
                return datetime.combine(d, t_dt.time())
            except ValueError:
                continue
        return t_dt
    return t_dt

def rr_to_peak_samples(rr_seconds: Iterable[float], fs: float, t0_s: float = 0.0) -> np.ndarray:
    """
    Convert RR sequence (seconds) to absolute peak sample indices at sampling rate fs.
    The first peak is placed at t0_s.
    """
    rr = np.asarray(rr_seconds, dtype=float).ravel()
    t_peaks = t0_s + np.cumsum(np.insert(rr, 0, 0.0))
    return np.rint(t_peaks * fs).astype(np.int64)
