import re
import pandas as pd

# Canonical schema we will use everywhere in code & on disk
CANONICAL_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

# Map known legacy names -> canonical names
LEGACY_TO_CANONICAL = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    # also handle common variants just in case
    "sepal length": "sepal_length",
    "sepal width": "sepal_width",
    "petal length": "petal_length",
    "petal width": "petal_width",
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any incoming/old Iris column names to canonical snake_case,
    ensure correct order & types, and drop unexpected columns.
    """
    # lowercase & strip whitespace
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # strip units like ' (cm)' generically (before mapping), keep 'target' as is
    def strip_units(name: str) -> str:
        if name == "target":
            return name
        return re.sub(r"\s*\(.*?\)\s*$", "", name)  # remove trailing " (cm)" etc.

    stripped = {c: strip_units(c) for c in df.columns}
    df.rename(columns=stripped, inplace=True)

    # apply explicit legacy -> canonical mapping
    df.rename(columns=LEGACY_TO_CANONICAL, inplace=True)

    # if columns already snake_case (API payload), they pass through

    # keep only known columns & coerce types
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            # Create missing column as NaN to avoid KeyError; will be handled by imputer or dropna later
            df[col] = pd.NA

    # reorder
    df = df[CANONICAL_COLUMNS]

    # type coercion (best effort)
    num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")

    return df