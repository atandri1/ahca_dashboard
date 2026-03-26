from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ahca_dashboard.aggregations import (  # noqa: E402
    regional_summary_statistics,
    tag_summary_statistics,
)
from ahca_dashboard.config import DEFAULT_MIN_YEAR  # noqa: E402
from ahca_dashboard.io import ensure_dir, load_all_regions_from_excels  # noqa: E402
from ahca_dashboard.transform import (  # noqa: E402
    add_derived_columns,
    filter_to_jkl,
    filter_to_min_year,
    filter_to_target_tags,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the processed analysis dataset used by the Streamlit app.")
    p.add_argument("--input", type=Path, required=True, help="Folder containing regional Excel files (*.xlsx).")
    p.add_argument("--output", type=Path, required=True, help="Output folder for processed CSVs.")
    p.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR, help="Minimum inspection year to keep.")
    p.add_argument("--no-min-year-filter", action="store_true", help="Do not apply a minimum year filter.")
    p.add_argument("--no-tag-filter", action="store_true", help="Keep all tags (still filters to J/K/L).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output))

    df_all = load_all_regions_from_excels(input_dir)
    if df_all.empty:
        print(f"No Excel files found in: {input_dir}")
        print("Expected files like: text2567_20251001_cms_reg1.xlsx")
        return 2

    # Minimal cleanup aligned with notebook.
    if "inspection_date" in df_all.columns:
        df_all["inspection_date"] = pd.to_datetime(df_all["inspection_date"], errors="coerce")

    df = filter_to_jkl(df_all)
    if not args.no_tag_filter:
        df = filter_to_target_tags(df)
    df = add_derived_columns(df)
    if not args.no_min_year_filter:
        df = filter_to_min_year(df, min_year=args.min_year)

    analysis_path = output_dir / "analysis_dataset_jkl_top10tags.csv"
    df.to_csv(analysis_path, index=False)
    print(f"Wrote: {analysis_path} ({len(df):,} rows)")

    regional_summary_path = output_dir / "regional_summary_statistics.csv"
    regional_summary_statistics(df).to_csv(regional_summary_path, index=False)
    print(f"Wrote: {regional_summary_path}")

    tag_summary_path = output_dir / "tag_summary_statistics.csv"
    tag_summary_statistics(df).to_csv(tag_summary_path, index=False)
    print(f"Wrote: {tag_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

