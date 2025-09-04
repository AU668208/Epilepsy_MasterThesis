# DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import os
import random
from nptdms import TdmsFile 

# Convert fixed Python datetime lists to np.datetime64
summertime_np = np.array([
    "2010-03-28T02:00", "2011-03-27T02:00", "2012-03-25T02:00", "2013-03-31T02:00", "2014-03-30T02:00",
    "2015-03-29T02:00", "2016-03-27T02:00", "2017-03-26T02:00", "2018-03-25T02:00", "2019-03-31T02:00",
    "2020-03-29T02:00", "2021-03-28T02:00", "2022-03-27T02:00", "2023-03-26T02:00", "2024-03-31T02:00"
], dtype='datetime64[m]')

wintertime_np = np.array([
    "2010-10-31T03:00", "2011-10-30T03:00", "2012-10-28T03:00", "2013-10-27T03:00", "2014-10-26T03:00",
    "2015-10-25T03:00", "2016-10-30T03:00", "2017-10-29T03:00", "2018-10-28T03:00", "2019-10-27T03:00",
    "2020-10-25T03:00", "2021-10-31T03:00", "2022-10-30T03:00", "2023-10-29T03:00", "2024-10-27T03:00"
], dtype='datetime64[m]')

def correct_annotation_timestamp_np(anfalds_tidspunkt: np.datetime64) -> np.datetime64:
    """Return corrected timestamp for daylight saving time in np.datetime64 format."""
    year = int(str(anfalds_tidspunkt)[:4])  # Extract year as int

    # Find matching year index
    idx = None
    for i in range(len(summertime_np)):
        if str(summertime_np[i])[:4] == str(year):
            idx = i
            break
    if idx is None:
        raise ValueError(f"Year {year} not found in summertime/wintertime lists.")

    start_summer = summertime_np[idx]
    start_winter = wintertime_np[idx]

    if start_summer <= anfalds_tidspunkt < start_winter:
        return anfalds_tidspunkt - np.timedelta64(2, 'h')  # Summer time: UTC+2
    else:
        return anfalds_tidspunkt - np.timedelta64(1, 'h')  # Winter time: UTC+1
    
def Load_full_ecg_data(patient_id: str):
    """
    Load full ECG signal + all seizure annotations for a patient.
    """
    import os
    import pandas as pd
    import numpy as np
    from nptdms import TdmsFile
    base_dir = r"E:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne"
    #base_dir = r"D:\ML algoritme tl anfaldsdetektion vha HRV\ePatch data from Aarhus to Lausanne"

    # Find mapper
    folder_top = os.listdir(base_dir)
    seizure_log_dir = os.path.join(base_dir, folder_top[8])
    data_dir = os.path.join(base_dir, folder_top[7])

    # Find patientens mappe
    patient_path = os.path.join(data_dir, patient_id)
    subfolders = [
        f for f in os.listdir(patient_path)
        if any(k in f.lower() for k in ["enrollment", "recording"])
    ]
    session_path = os.path.join(patient_path, subfolders[0])
    files = os.listdir(session_path)

    tdms_file = next((os.path.join(session_path, f) for f in files if f.endswith(".tdms")), None)
    index_file = next((os.path.join(session_path, f) for f in files if "tdms_index" in f), None)

    # Læs meta-info
    with TdmsFile.open(index_file) as index_tdms:
        info = pd.DataFrame()
        for group in index_tdms.groups():
            for channel in group.channels():
                for prop, val in channel.properties.items():
                    info.loc[prop, "Value"] = val

    tdms_data = TdmsFile.read(tdms_file)
    signal = tdms_data.groups()[0].channels()[0].data

    wf_start = info.loc["wf_start_time", "Value"]
    wf_increment = info.loc["wf_increment", "Value"]
    sample_rate = int(1 / wf_increment)

    # Hele signalet → DataFrame
    step_ns = int(round(wf_increment * 1e9))
    start_np = np.datetime64(wf_start, "ns")
    timestamps = start_np + np.arange(len(signal)) * np.timedelta64(step_ns, "ns")
    ecg_df = pd.DataFrame({"Timestamp": timestamps, "Value": signal})

    # Indlæs alle anfaldsannoteringer
    seizure_log_file = next((os.path.join(seizure_log_dir, f) for f in os.listdir(seizure_log_dir) if patient_id in f), None)
    seizure_df = pd.read_excel(seizure_log_file, skiprows=5)
    seizure_df.columns = seizure_df.iloc[0]
    seizure_df = seizure_df[1:].reset_index(drop=True)

    return {
        "PatientID": patient_id,
        "ECG": ecg_df,
        "Seizures": seizure_df,
        "SampleRate": sample_rate
    }

