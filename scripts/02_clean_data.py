"""Step 02 - clean the merged listings and persist the cleaning log.

Usage: uv run python scripts/02_clean_data.py
"""

from __future__ import annotations

import json
import logging

from revaluation.config import METRICS_DIR
from revaluation.data.clean import build_clean_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    _, payload = build_clean_dataset(save=True)
    print(json.dumps(payload["rows"], indent=2))
    print(json.dumps(payload["report"]["outlier_flags"], indent=2))
    print(f"Cleaning report written to {METRICS_DIR / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
