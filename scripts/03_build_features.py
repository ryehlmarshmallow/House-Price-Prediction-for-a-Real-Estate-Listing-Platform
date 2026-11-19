"""Step 03 - engineer features and persist the analytical table.

Usage: uv run python scripts/03_build_features.py
"""

from __future__ import annotations

import logging

from revaluation.data.features import build_modeling_table, feature_group_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    table = build_modeling_table(save=True)
    logger.info("Modeling table: %s rows x %s columns", *table.shape)
    groups = feature_group_table()
    print(groups.groupby("group").size().to_string())


if __name__ == "__main__":
    main()
