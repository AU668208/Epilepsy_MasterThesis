from typing import Iterator, Tuple, Optional
import numpy as np
import pandas as pd

from pathlib import Path

from .index import RecordingEntry, build_recording_index
from ..io.tdms import load_tdms_from_path, TdmsMeta
from .annotations import load_annotations

def iter_recordings(tdms_root: Path, ann_root: Path):
    index = build_recording_index(tdms_root, ann_root)

    for entry in index:
        sig, meta = load_tdms_from_path(entry.tdms_path)
        ann = load_annotations(entry.annotation_path) if entry.annotation_path else None
        yield sig, meta, ann
