from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def extract_region_from_filename(filename: str) -> Optional[int]:
    name = filename.lower()
    if "reg10" in name:
        return 10
    if "reg5a" in name or "reg5b" in name:
        return 5
    for region in range(1, 10):
        if f"reg{region}" in name:
            return region
    return None


def iter_excel_files(input_dir: Path) -> Iterable[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    return sorted([p for p in input_dir.glob("*.xlsx") if p.is_file()])


def load_all_regions_from_excels(input_dir: Path) -> pd.DataFrame:
    """Loads all `*.xlsx` in a folder and attaches `cms_region` and `source_file`."""
    frames: list[pd.DataFrame] = []
    for file_path in iter_excel_files(input_dir):
        df = pd.read_excel(file_path)
        df["cms_region"] = extract_region_from_filename(file_path.name)
        df["source_file"] = file_path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_processed_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Keep these consistent for the dashboard.
    if "inspection_date" in df.columns:
        df["inspection_date"] = pd.to_datetime(df["inspection_date"], errors="coerce")
    if "deficiency_tag" in df.columns:
        df["deficiency_tag"] = pd.to_numeric(df["deficiency_tag"], errors="coerce").astype("Int64")
    if "cms_region" in df.columns:
        df["cms_region"] = pd.to_numeric(df["cms_region"], errors="coerce").astype("Int64")
    return df


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

