import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.hrv_epatch.plots.plotstyle import palette

def plot_std_timecourse_by_label(
    df_join: pd.DataFrame,
    recording_uid: int,
    outpath: Path | None = None,
    title_suffix: str = "",
    show: bool = True,
):
    """
    Plot window-level STD over time for a single recording,
    with visually distinct colour coding for:
        - baseline
        - seizure_extended
        - seizure_core
    """

    df_r = df_join[df_join["recording_uid"] == recording_uid].copy()
    if df_r.empty:
        print(f"No windows found for recording_uid={recording_uid}")
        return

    # Tid i timer fra recording start
    df_r["t_hours"] = df_r["win_start_s"] / 3600.0

    # Tydelige farver
    label_colors = {
        "baseline":           palette.get("primary", "#8FB996"),   # muted green
        "seizure_extended":   "#F2C14E",   # warm yellow
        "seizure_core":       "#D94E41",   # dark red
    }

    # Tegn i rækkefølge: baseline (bagest), extended, core (øverst)
    plot_order = ["baseline", "seizure_extended", "seizure_core"]

    plt.figure(figsize=(14, 5))

    for label in plot_order:
        df_l = df_r[df_r["label"] == label]
        if df_l.empty:
            continue

        alpha = 0.35 if label == "baseline" else (0.65 if label == "seizure_extended" else 0.95)
        size  = 10   if label == "baseline" else (18   if label == "seizure_extended" else 25)

        plt.scatter(
            df_l["t_hours"],
            df_l["std"],
            s=size,
            alpha=alpha,
            label=label.replace("_", " "),
            color=label_colors[label],
        )

    plt.xlabel("Time from recording start (hours)")
    plt.ylabel("Window STD (a.u.)")

    title = f"Window-level STD over time – recording_uid={recording_uid}"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)

    plt.legend(frameon=True)
    plt.tight_layout()

    if outpath is not None:
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print("Saved figure to:", outpath)

    if show:
        plt.show()
    else:
        plt.close()

def plot_all_std_timecourses(
    df_join: pd.DataFrame,
    df_rec: pd.DataFrame,
    outdir: Path | str,
    title_suffix: str = "extended ±120 s",
):
    """
    Lav STD-tidsserier for alle recordings i df_join og gem én PNG pr. recording.

    Parameters
    ----------
    df_join : pd.DataFrame
        Window-metrics + 'label' + 'win_start_s' + 'recording_uid'.
        (fx df_join_ext120)

    df_rec : pd.DataFrame
        Recording-indeks med mindst:
            - recording_uid
            - patient_id
            - recording_id

    outdir : Path or str
        Output-mappe, hvor figurer gemmes.

    title_suffix : str
        Tekst der sættes i parentes i figurens titel
        (fx "extended ±120 s").
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Sørg for stabil rækkefølge
    recording_uids = sorted(df_join["recording_uid"].unique())

    for uid in recording_uids:
        # Forsøg at finde patient/recording info til filnavn
        rec_row = df_rec[df_rec["recording_uid"] == uid]
        if not rec_row.empty:
            rec_row = rec_row.iloc[0]
            pid = int(rec_row["patient_id"])
            rid = int(rec_row["recording_id"])
            fname = f"P{pid:02d}_R{rid:02d}_uid{uid:02d}_std_timecourse.png"
        else:
            fname = f"uid{uid:02d}_std_timecourse.png"

        outpath = outdir / fname

        plot_std_timecourse_by_label(
            df_join=df_join,
            recording_uid=uid,
            outpath=outpath,
            title_suffix=title_suffix,
            show=False,  # vigtigt, så vi ikke åbner alle figurer interaktivt
        )
