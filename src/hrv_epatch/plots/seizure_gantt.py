"""
Unified Gantt plot for recordings and seizures,
based on structured index DataFrames (df_rec, df_evt).

Compatible with 01_Build_dataset_index CSV output.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_gantt_from_index(
    df_rec: pd.DataFrame,
    df_evt: pd.DataFrame,
    max_hours: float = 96.0,
    day_grid: bool = True,
    figsize=(18, 10),
    min_box_width_min: float = 5.0,
):
    """
    Plot a Gantt-style timeline of all recordings and seizures.

    Parameters
    ----------
    df_rec : pd.DataFrame
        Must contain at least:
            - recording_uid (int)
            - patient_id (int)
            - enrollment_id (str or None)
            - recording_id (int)
            - recording_start (Timestamp)
            - recording_end (Timestamp)
    
    df_evt : pd.DataFrame
        Must contain at least:
            - recording_uid (int)
            - t0, t1  (seconds relative to recording_start)

    max_hours : float
        Outer time-window for plotting (hours from recording_start.normalize()).
        The final x-limits are shrunk to the range covering all recordings,
        rounded to the nearest 6-hour grid inside [0, max_hours].

    day_grid : bool
        Draw vertical dashed lines every 24 hours.

    min_box_width_min : float
        Minimum visual width of a seizure bar.

    Returns
    -------
    fig, ax : matplotlib Figure and Axis
    """

    recs_sorted = df_rec.sort_values(
        ["patient_id", "enrollment_id", "recording_id"]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(recs_sorted))
    labels = []

    # Tracking all recording-start/-end times in hours
    all_rec_start_h = []
    all_rec_end_h = []

    for i, rec in recs_sorted.iterrows():
        pid = rec["patient_id"]
        enr = rec["enrollment_id"]
        rid = rec["recording_id"]
        rec_start = rec["recording_start"]
        rec_end = rec["recording_end"]

        date_str = rec_start.strftime("%y-%m-%d")

        label = f"Patient{pid}"
        if isinstance(enr, str) and enr:
            label += f"-{enr.upper()}"
        label += f"-R{rid} ({date_str})"
        labels.append(label)

        # Reference midnight for this recording
        day0 = rec_start.normalize()
        rec_duration_h = (rec_end - rec_start).total_seconds() / 3600.0
        rec_start_h = (rec_start - day0).total_seconds() / 3600.0
        rec_end_h = rec_start_h + rec_duration_h

        # Track global start/stop for recordings
        all_rec_start_h.append(rec_start_h)
        all_rec_end_h.append(rec_end_h)

        # Clip recording to [0, max_hours] for plotting
        rec_plot_start = max(rec_start_h, 0.0)
        rec_plot_end = min(rec_end_h, max_hours)

        if rec_plot_end > rec_plot_start:
            ax.hlines(
                y=i,
                xmin=rec_plot_start,
                xmax=rec_plot_end,
                color="lightgray",
                linewidth=6,
                zorder=1,
            )

        # Seizures in this recording
        evts = df_evt[df_evt["recording_uid"] == rec["recording_uid"]]

        for _, ev in evts.iterrows():
            ev_start = rec_start + pd.to_timedelta(ev["t0"], unit="s")
            ev_end   = rec_start + pd.to_timedelta(ev["t1"], unit="s")

            s_h = (ev_start - day0).total_seconds() / 3600.0
            e_h = (ev_end   - day0).total_seconds() / 3600.0

            # Clip seizures to [0, max_hours]
            if e_h <= 0 or s_h >= max_hours:
                continue

            s_h = max(s_h, 0.0)
            e_h = min(e_h, max_hours)

            width_h = e_h - s_h
            if width_h <= 0:
                continue

            width_h = max(width_h, min_box_width_min / 60.0)

            ax.broken_barh(
                [(s_h, width_h)],
                (i - 0.3, 0.6),
                facecolors="darkred",
                edgecolors="black",
                linewidth=0.4,
                zorder=3,
            )

    # If for some reason there are no recordings, fallback
    if len(all_rec_start_h) == 0:
        xmin_raw = 0.0
        xmax_raw = max_hours
    else:
        xmin_raw = min(all_rec_start_h)
        xmax_raw = max(all_rec_end_h)

    # Start: always DOWN to nearest 6th hour
    xmin = 6.0 * np.floor(xmin_raw / 6.0)
    # End: always UP to nearest 6th hour
    xmax = 6.0 * np.ceil(xmax_raw / 6.0)

    # Limit to [0, max_hours]
    xmin = max(xmin, 0.0)
    xmax = min(xmax, max_hours)

    # Y-labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    # X-limits based on recordings
    ax.set_xlim(xmin, xmax)

    # Day grid (every 24 hours) within [xmin, xmax]
    if day_grid:
        start_day = int(np.floor(xmin / 24.0) * 24)
        end_day = int(np.ceil(xmax / 24.0) * 24)
        for d in range(start_day, end_day + 1, 24):
            if xmin <= d <= xmax:
                ax.axvline(
                    d,
                    color="lightgray",
                    linestyle="--",
                    linewidth=0.7,
                    zorder=0,
                )

    # X-ticks every 6th hour within [xmin, xmax]
    ticks = np.arange(xmin, xmax + 1e-6, 6.0)
    tick_labels = []
    for t in ticks:
        tod = t % 24
        hh = int(tod)
        mm = int((tod - hh) * 60)
        tick_labels.append(f"{hh:02d}:{mm:02d}")

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Time (hh:mm, unfolded over days)")
    ax.set_ylabel("Patient / enrollment / recording")
    ax.set_title("Recording periods and seizures per recording")

    legend_handles = [
        Patch(facecolor="lightgray", edgecolor="lightgray", label="Recording"),
        Patch(facecolor="darkred", edgecolor="black", label="Seizures (included)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    return fig, ax
