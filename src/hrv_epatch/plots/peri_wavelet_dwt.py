import numpy as np
import matplotlib.pyplot as plt
import pywt


def compute_dwt_energy(ecg, wavelet='db4', level=5):
    """
    Compute wavelet decomposition band energies for an ECG segment.
    Returns:
        coeffs_dict: { "A5": energy, "D5": energy, ..., "D1": energy }
    """
    coeffs = pywt.wavedec(ecg, wavelet, level=level)

    # coeffs structure: [A5, D5, D4, ..., D1]
    band_names = ["A5", "D5", "D4", "D3", "D2", "D1"]

    energies = {}
    for name, c in zip(band_names, coeffs):
        energies[name] = np.sum(c**2)

    return energies


def plot_dwt_bands(energies, ax=None, title=""):
    """
    Plot bar chart of DWT band energies on a given Axes.
    If ax is None, uses current axes.
    """
    if ax is None:
        ax = plt.gca()

    bands = list(energies.keys())
    vals = [energies[b] for b in bands]

    bars = ax.bar(bands, vals, color="tab:blue", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Energy")
    ax.grid(axis="y", alpha=0.3)

    # write values above bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )



def plot_dwt_triplet(df_peri, fs, patient_id, seizure_id,
                     figsize=(12, 12), wavelet='db4', level=5,
                     save_path=None, show_plot=True):
    """
    3x1 DWT-figur: baseline_near, ictal, post_near.
    """
    # Extract peri segments
    win_baseline = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "baseline_near")
    ]
    win_ictal = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "ictal")
    ]
    win_post = df_peri[
        (df_peri.patient_id == patient_id) &
        (df_peri.seizure_id == seizure_id) &
        (df_peri.role == "post_near")
    ]

    if len(win_baseline) == 0 or len(win_ictal) == 0 or len(win_post) == 0:
        raise ValueError("Missing one or more peri-ictal windows.")

    ecg_base = win_baseline.iloc[0]["signal"]
    ecg_ict = win_ictal.iloc[0]["signal"]
    ecg_post = win_post.iloc[0]["signal"]

    # Compute energies
    E_base = compute_dwt_energy(ecg_base, wavelet, level)
    E_ict = compute_dwt_energy(ecg_ict, wavelet, level)
    E_post = compute_dwt_energy(ecg_post, wavelet, level)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)

    plot_dwt_bands(E_base, ax=axes[0],
                   title=f"Patient {patient_id}, Seizure {seizure_id} — Baseline_Near")
    plot_dwt_bands(E_ict, ax=axes[1], title="Ictal")
    plot_dwt_bands(E_post, ax=axes[2], title="Post_Near")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # Luk figuren for at undgå visning i batch-mode
    if not show_plot:
        plt.close(fig)

    return fig

