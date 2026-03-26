from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script without packaging.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ahca_dashboard.alignment import compute_tfidf_similarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add TF-IDF regulatory alignment similarity to a citations CSV.")
    p.add_argument("--citations", type=Path, required=True, help="Citations CSV (must include deficiency_tag, inspection_text).")
    p.add_argument("--baseline", type=Path, required=True, help="Appendix PP baseline CSV from build_appendixpp_baseline.py.")
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    citations = pd.read_csv(args.citations)
    baseline = pd.read_csv(args.baseline)

    sim = compute_tfidf_similarity(citations, baseline)
    citations = citations.copy()
    citations["tfidf_sim"] = sim

    args.output.parent.mkdir(parents=True, exist_ok=True)
    citations.to_csv(args.output, index=False)
    print(f"Wrote: {args.output} ({len(citations):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

