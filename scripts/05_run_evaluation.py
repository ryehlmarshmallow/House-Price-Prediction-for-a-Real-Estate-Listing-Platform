"""Step 05 - evaluate the selected model and produce diagnostic figures.

Usage: uv run python scripts/05_run_evaluation.py
"""

from __future__ import annotations

import json
import logging

from revaluation.models.evaluate import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    diagnostics = run_evaluation()
    print(
        json.dumps(
            {
                "selected_model": diagnostics["selected_model"],
                "rmse": round(diagnostics["business_metrics"]["rmse"], 2),
                "mae": round(diagnostics["business_metrics"]["mae"], 2),
                "mape": round(diagnostics["business_metrics"]["mape"], 2),
                "r2": round(diagnostics["business_metrics"]["r2"], 4),
                "within_tolerance_usd": round(
                    diagnostics["business_metrics"]["within_tolerance_usd"], 2
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
