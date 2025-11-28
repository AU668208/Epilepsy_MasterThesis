# %%
from src.hrv_epatch.io.labview import read_labview_rr, read_header_datetime_lvm
from src.hrv_epatch.io.data_loader import  Load_full_ecg_data
import pandas as pd
from datetime import datetime
import neurokit2 as nk
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class SeizureEvent:
    seizure_id: int
    t0: float       # sekunder fra recording start
    t1: float       # sekunder fra recording start


# %%
path_lvm = r"E:\ML algoritme tl anfaldsdetektion vha HRV\LabView-Results\Patient5_remove_s5_e5.lvm"
rr_intervals_lv = read_labview_rr(path_lvm)
starttime_lv = read_header_datetime_lvm(path_lvm)
print(f"Read {len(rr_intervals_lv)} LabVIEW RR intervals from LVM.")
print(f"LabVIEW start time from LVM header: {starttime_lv}")

# %%
res = Load_full_ecg_data("Patient 5")
seizure_df = res["Seizures"]
print(res.keys())

# %%
time_to_drop_end = 300  # seconds in both start and end of signal
time_to_drop_start = starttime_lv.timestamp() - res['StartTime'].timestamp()
print(f"Time to drop at start based on LabVIEW start time: {time_to_drop_start} seconds")
fs = res['SampleRate']
starttime_tdms = res['StartTime']
print(f"ECG start time from TDMS: {starttime_tdms}")
samples_to_drop_start = int(time_to_drop_start * fs)  # Convert to integer
samples_to_drop_end = int(time_to_drop_end * fs)
print(f"Dropping {samples_to_drop_start} samples from start and {samples_to_drop_end} samples from end of ECG signal")
ecg = res['ECG'].iloc[samples_to_drop_start:-samples_to_drop_end]
print(f"ECG length before dropping: {len(res['ECG'])} samples")
print(f"ECG length after dropping: {len(ecg)} samples")
corrected_starttime_tdms = starttime_tdms + pd.Timedelta(seconds=time_to_drop_start)
print(f"Corrected ECG start time: {corrected_starttime_tdms}")

# %%


# Extract the ECG signal values
ecg_signal = ecg["Value"].to_numpy()

# Use NeuroKit2 to process the ECG signal and find R-peaks
cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=fs)
r_peaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=fs)[0]

# Display the indices of the detected R-peaks
detected_r_peaks = np.flatnonzero(r_peaks["ECG_R_Peaks"].to_numpy())


# %%
print(f"Detected R-peaks: {detected_r_peaks}")
print(f"Number of detected R-peaks: {len(detected_r_peaks)}")

n_dif_rpeaks = len(rr_intervals_lv) - len(detected_r_peaks)
print(f"Difference in number of R-peaks: {n_dif_rpeaks}")

# %%
# Convert LabView RR intervals to R-peaks
labview_r_peaks = np.cumsum(np.insert(rr_intervals_lv, 0, 0)) * fs
labview_r_peaks = np.round(labview_r_peaks).astype(int)

# Convert NeuroKit2 detected R-peaks to RR intervals
rr_intervals_detected = np.diff(detected_r_peaks) / fs

# Display the first 10 LabView R-peaks and NeuroKit2 RR intervals
print("First 10 LabView R-peaks (converted from RR intervals):", labview_r_peaks[:10])
print("First 10 NeuroKit2 RR intervals (converted from R-peaks):", rr_intervals_detected[:10])

# Compare the lengths of the two sets of R-peaks
print(f"Number of LabView R-peaks: {len(labview_r_peaks)}")
print(f"Number of NeuroKit2 R-peaks: {len(detected_r_peaks)}")

# Display the first 10 elements of LabView R-peaks and NeuroKit2 detected R-peaks
print("First 10 LabView R-peaks:", labview_r_peaks[:10])
print("First 10 NeuroKit2 detected R-peaks:", detected_r_peaks[:10])

# Display the first 10 elements of LabView RR intervals and NeuroKit2 RR intervals
print("First 10 LabView RR intervals:", rr_intervals_lv[:10])
print("First 10 NeuroKit2 RR intervals:", rr_intervals_detected[:10])

# %%
print(fs)

1.13867188*fs

# %%
# Ensure both arrays have the same length for comparison
min_length = min(len(detected_r_peaks), len(labview_r_peaks))
aligned_detected_r_peaks = detected_r_peaks[:min_length]
aligned_labview_r_peaks = labview_r_peaks[:min_length]

# Calculate the absolute difference in R-peak placement
r_peak_differences = np.abs(aligned_detected_r_peaks - aligned_labview_r_peaks)
average_r_peak_mistake = np.mean(r_peak_differences)

# Ensure both RR interval arrays have the same length for comparison
min_length_rr = min(len(rr_intervals_detected), len(rr_intervals_lv))
aligned_rr_intervals_detected = rr_intervals_detected[:min_length_rr]
aligned_rr_intervals_lv = rr_intervals_lv[:min_length_rr]

# Calculate the absolute difference in RR intervals
rr_interval_differences = np.abs(aligned_rr_intervals_detected - aligned_rr_intervals_lv)
average_rr_interval_mistake = np.mean(rr_interval_differences)

# Print the results
print(f"Average mistake in R-peak placement: {average_r_peak_mistake:.2f} samples")
print(f"Average mistake in RR interval length: {average_rr_interval_mistake:.4f} seconds")

# %%
import matplotlib.pyplot as plt

def visualize_ecg_with_r_peaks(
    raw_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    fs: float,
    detected_r_peaks: np.ndarray,
    labview_r_peaks: np.ndarray,
    start_time: float,
    duration: float
):
    """
    Visualize raw ECG, cleaned ECG, and R-peaks from NeuroKit2 and LabView.

    Parameters:
    - raw_signal: Raw ECG signal (numpy array).
    - cleaned_signal: Cleaned ECG signal (numpy array).
    - fs: Sampling frequency (float).
    - detected_r_peaks: R-peaks detected by NeuroKit2 (numpy array).
    - labview_r_peaks: R-peaks from LabView (numpy array).
    - start_time: Start time in seconds for the visualization window (float).
    - duration: Duration in seconds for the visualization window (float).
    """
    start_sample = int(start_time * fs)
    end_sample = int((start_time + duration) * fs)

    # Extract the window of interest
    raw_window = raw_signal[start_sample:end_sample]
    cleaned_window = cleaned_signal[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs

    # Get R-peaks within the window
    detected_r_peaks_window = detected_r_peaks[
        (detected_r_peaks >= start_sample) & (detected_r_peaks < end_sample)
    ] - start_sample
    labview_r_peaks_window = labview_r_peaks[
        (labview_r_peaks >= start_sample) & (labview_r_peaks < end_sample)
    ] - start_sample

    # Plot the signals and R-peaks
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, raw_window, label="Raw ECG", color="gray", alpha=0.7)
    plt.plot(time_axis, cleaned_window, label="Cleaned ECG", color="blue", alpha=0.9)

    # Plot NeuroKit2 R-peaks
    plt.scatter(
        time_axis[detected_r_peaks_window],
        cleaned_window[detected_r_peaks_window],
        color="orange",
        label="NeuroKit2 R-peaks",
        zorder=3,
    )

    # Plot LabView R-peaks
    plt.scatter(
        time_axis[labview_r_peaks_window],
        cleaned_window[labview_r_peaks_window],
        color="green",
        label="LabView R-peaks",
        zorder=3,
    )

    plt.title(f"ECG Visualization (Start: {start_time}s, Duration: {duration}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
visualize_ecg_with_r_peaks(
    raw_signal=ecg_signal,
    cleaned_signal=cleaned_ecg,
    fs=fs,
    detected_r_peaks=detected_r_peaks,
    labview_r_peaks=labview_r_peaks,
    start_time=0,  # Adjust the start time as needed
    duration=10      # Adjust the duration as needed
)

# %%
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def plot_with_offset(
    raw_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    fs: float,
    detected_r_peaks: np.ndarray,
    labview_r_peaks: np.ndarray,
    start_time: float,
    duration: float,
    coarse_range_s: float = 10.0,
    fine_range_s: float = 0.05
):
    """
    Interaktiv visualisering af ECG med to R-peak-serier og justerbart offset
    mellem LabVIEW og ECG/NeuroKit.

    - raw_signal, cleaned_signal: hele signalet (numpy arrays)
    - fs: samplingfrekvens [Hz]
    - detected_r_peaks: NeuroKit R-peak-indeks (samples, samme akse som ECG)
    - labview_r_peaks: LabVIEW R-peak-indeks (i LabVIEW-akse, altså uden offset)
    - start_time: vinduets start i sekunder (ECG-akse)
    - duration: vinduesbredde i sekunder
    """
    # ---- ECG-vindue ----
    start_sample = int(start_time * fs)
    end_sample = int((start_time + duration) * fs)

    raw_window = raw_signal[start_sample:end_sample]
    cleaned_window = cleaned_signal[start_sample:end_sample]
    time_axis = np.arange(start_sample, end_sample) / fs

    # NeuroKit peaks i vindue (flyttes ikke af offset)
    detected_window_idx = detected_r_peaks[
        (detected_r_peaks >= start_sample) & (detected_r_peaks < end_sample)
    ] - start_sample

    # Helper: beregn LabVIEW-peaks i vindue for et givent offset (sekunder)
    def compute_labview_indices(offset_s: float) -> np.ndarray:
        shift_samples = int(round(offset_s * fs))
        labview_shifted = labview_r_peaks + shift_samples  # flyt til ECG-akse
        mask = (labview_shifted >= start_sample) & (labview_shifted < end_sample)
        local_idx = labview_shifted[mask] - start_sample    # lokale indekser i vinduet
        return local_idx

    # Initialt offset = 0
    init_offset = 0.0
    labview_idx = compute_labview_indices(init_offset)

    # ---- Figur og plots ----
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)  # plads til 2 sliders

    # Signaler
    ax.plot(time_axis, raw_window, label="Raw ECG", color="gray", alpha=0.7)
    ax.plot(time_axis, cleaned_window, label="Cleaned ECG", color="blue", alpha=0.9)

    # NeuroKit peaks
    ax.scatter(
        time_axis[detected_window_idx],
        cleaned_window[detected_window_idx],
        color="orange",
        label="NeuroKit2 R-peaks",
        zorder=3,
    )

    # LabVIEW peaks (initialt)
    labview_scatter = ax.scatter(
        time_axis[labview_idx],
        cleaned_window[labview_idx],
        color="green",
        label="LabView R-peaks",
        zorder=3,
    )

    ax.set_title(f"ECG Visualization (Start: {start_time}s, Duration: {duration}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Sliders: grov + fin offset ----
    # grov offset (f.eks. ±2 sek)
    ax_offset_coarse = plt.axes([0.25, 0.12, 0.65, 0.03])
    slider_coarse = Slider(
        ax=ax_offset_coarse,
        label="Offset coarse (s)",
        valmin=-coarse_range_s,
        valmax=coarse_range_s,
        valinit=0.0,
        valstep=0.025,
    )

    # fin offset (f.eks. ±0.05 sek)
    ax_offset_fine = plt.axes([0.25, 0.06, 0.65, 0.03])
    slider_fine = Slider(
        ax=ax_offset_fine,
        label="Offset fine (s)",
        valmin=-fine_range_s,
        valmax=fine_range_s,
        valinit=0.0,
        valstep=0.001,
    )

    # tekst nederst der viser total offset
    offset_text = fig.text(0.5, 0.01, f"Total offset: {init_offset:.4f} s",
                           ha="center", va="bottom")

    def update(val):
        total_offset = slider_coarse.val + slider_fine.val
        new_idx = compute_labview_indices(total_offset)

        if len(new_idx) == 0:
            # Ingen peaks i vindue -> tom scatter
            labview_scatter.set_offsets(np.empty((0, 2)))
        else:
            labview_scatter.set_offsets(
                np.c_[time_axis[new_idx], cleaned_window[new_idx]]
            )

        offset_text.set_text(f"Total offset: {total_offset:.4f} s")
        fig.canvas.draw_idle()

    slider_coarse.on_changed(update)
    slider_fine.on_changed(update)

    plt.show()



# Example usage
plot_with_offset(
    raw_signal=ecg_signal,
    cleaned_signal=cleaned_ecg,
    fs=fs,
    detected_r_peaks=detected_r_peaks,
    labview_r_peaks=labview_r_peaks,
    start_time=10000,
    duration=10,
    coarse_range_s=10.0,
    fine_range_s=0.05,
)


