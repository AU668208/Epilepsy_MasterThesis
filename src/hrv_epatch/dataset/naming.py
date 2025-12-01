import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class RecordingKey:
    patient_id: int
    enrollment_id: Optional[str]  # 'a','b','c' eller None
    recording_id: int             # 1,2,...

def parse_recording_key(tdms_path: Path) -> RecordingKey:
    """
    Examples:
      'Patient 5_1.tdms'      -> patient_id=5, enrollment_id=None, recording_id=1
      'Patient 8a_1.tdms'     -> patient_id=8, enrollment_id='a', recording_id=1
      'Patient 38b_2.tdms'    -> patient_id=38, enrollment_id='b', recording_id=2
    """
    m = re.match(r"Patient\s+(\d+)([abc]?)_(\d+)\.tdms$", tdms_path.name)
    if not m:
        raise ValueError(f"Could not parse filename: {tdms_path}")
    pid = int(m.group(1))
    enroll = m.group(2) or None
    rec = int(m.group(3))
    return RecordingKey(pid, enroll, rec)
