import pandas as pd

def describe_by_label(df_join: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Compute median segment-level statistics grouped by label/context.
    Supports multiple label column names (label, context_x/y, etc.).
    """
    # Detect label/context column
    candidates = [
        c for c in df_join.columns 
        if c in ("label", "context")
        or "context" in c.lower()
        or "label" in c.lower()
    ]
    if not candidates:
        raise ValueError("No context/label column found in dataframe.")

    label_col = candidates[0]

    # Metrics expected to be present
    metric_cols = ["std", "range", "is_noiseburst"]
    missing = [m for m in metric_cols if m not in df_join.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    out = (
        df_join
        .groupby(label_col)[metric_cols]
        .median()
        .rename(columns={
            "std": f"std_{name}",
            "range": f"range_{name}",
            "is_noiseburst": f"noise_{name}",
        })
    )

    return out
