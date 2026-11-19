"""Step 04 - train, cross-validate and persist the final model.

Usage: uv run python scripts/04_train_models.py
"""

from __future__ import annotations

import json
import logging

from revaluation.models.train import run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    summary = run(save=True)
    print(
        json.dumps(
            {
                "selected_model": summary["selected_model_label"],
                "rmse": round(summary["metrics"]["rmse"], 2),
                "mae": round(summary["metrics"]["mae"], 2),
                "mape": round(summary["metrics"]["mape"], 2),
                "r2": round(summary["metrics"]["r2"], 4),
                "conformal": summary["conformal"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
