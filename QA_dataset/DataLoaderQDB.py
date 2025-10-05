import numpy as np
import pandas as pd

def ann_reader(annotation, anntr=None, from_idx=None, to_idx=None):
    """
    Annotation Reader
    -------------------------------------------------------------------------
    This function reads quality annotations from a CSV file for the BUT QDB dataset.

    Args:
        annotation (str): Path to the annotation CSV file (without '.csv' extension).
        anntr (list or np.ndarray, optional): Indices of annotators to use.
        from_idx (int, optional): Start sample index.
        to_idx (int, optional): End sample index.

    Returns:
        np.ndarray: Array of annotations, sample-by-sample.

    Example:
        ann = ann_reader('100001_ANN', [1, 4], 1, 1000000)
    """

    # Read CSV file into numpy array
    TA = pd.read_csv(f"{annotation}.csv").to_numpy()

    sl = np.array([0, 3, 6, 9])  # MATLAB is 1-based, Python is 0-based
    # Find valid columns for annsize
    valid_sl = sl[~np.isnan(TA[-1, sl + 1])]
    annsize = [len(anntr) if anntr is not None else 4, int(TA[-1, valid_sl[-1] + 1]) if valid_sl.size > 0 else TA.shape[1]]

    ann = np.zeros(annsize)

    # Handle default arguments
    if anntr is None:
        anntr = np.arange(4)
    if from_idx is None:
        from_idx = 0
    if to_idx is None:
        to_idx = annsize[1]

    poc = 0
    sl = sl[anntr]
    for i in sl:
        Lan = np.sum(~np.isnan(TA[:, i]))
        poc += 1

        # Find M
        M = None
        for j in range(Lan):
            if TA[j, i] >= from_idx:
                M = j
                break

        # Find N
        N = None
        for j in range(M + 1, Lan):
            if TA[j, i + 1] >= to_idx:
                N = j
                break

        # Fill ann
        for j in range(M, N + 1):
            start = int(TA[j, i])
            end = int(TA[j, i + 1])
            value = TA[j, i + 2]
            ann[poc - 1, start:end + 1] = value

    ann = ann[:, from_idx:to_idx]
    return ann
