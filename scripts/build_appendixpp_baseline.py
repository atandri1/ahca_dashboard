from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ahca_dashboard.appendixpp import build_baseline_from_pdf  # noqa: E402
from ahca_dashboard.io import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Appendix PP F-tag baseline text from a PDF into a CSV.")
    p.add_argument("--pdf", type=Path, required=True, help="Path to the Appendix PP PDF.")
    p.add_argument("--output", type=Path, default=Path("data/processed"), help="Output folder for CSV.")
    p.add_argument("--filename", type=str, default="appendix_pp_ftags.csv", help="Output CSV filename.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.output)
    df = build_baseline_from_pdf(args.pdf)
    if df.empty:
        print("No F-tags parsed. The PDF may be image-scanned or the parsing regex may need adjustment.")
        return 2

    out_path = out_dir / args.filename
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} ({len(df):,} F-tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

