from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from .naming import parse_recording_key, RecordingKey
from .annotations import find_annotation_file

@dataclass
class RecordingEntry:
    key: RecordingKey
    tdms_path: Path
    annotation_path: Optional[Path]

def build_recording_index(
    tdms_root: Path,
    annotations_root: Path,
) -> List[RecordingEntry]:

    entries: List[RecordingEntry] = []

    # Find alle TDMS-filer
    for tdms_file in tdms_root.rglob("*.tdms"):
        key = parse_recording_key(tdms_file)

        # find annotation (Patient 5a.xls etc.)
        ann = find_annotation_file(key, annotations_root)

        entries.append(RecordingEntry(
            key=key,
            tdms_path=tdms_file,
            annotation_path=ann,
        ))

    return sorted(entries, key=lambda e: (e.key.patient_id, e.key.enrollment_id or "", e.key.recording_id))
