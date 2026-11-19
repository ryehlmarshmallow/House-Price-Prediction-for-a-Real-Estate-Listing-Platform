"""Step 01 - generate the synthetic contextual data set.

Usage: uv run python scripts/01_generate_context.py
"""

from __future__ import annotations

import logging

from revaluation.data.load import load_combined
from revaluation.data.synthetic import build_context_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    combined = load_combined()
    context = build_context_dataset(combined)
    logger.info("Synthetic context generated: %s rows x %s columns", *context.shape)
    logger.info(
        "Amenity score range: %.1f - %.1f",
        context["amenity_score"].min(),
        context["amenity_score"].max(),
    )


if __name__ == "__main__":
    main()
