"""Step 08 - export the data dictionary to Markdown and Excel.

Usage: uv run python scripts/07_export_data_dictionary.py

The dictionary is generated from the same specification objects that drive the
pipeline (``SYNTHETIC_SPECS`` and ``FEATURE_DEFINITIONS``), so documentation and
implementation cannot diverge.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from revaluation.config import DATA_DESCRIPTION, DOCS_DIR, ID_COLUMN
from revaluation.data.clean import (
    ABSENCE_TO_NONE,
    ABSENCE_TO_ZERO,
    MODE_IMPUTED,
    OUTLIER_RULES,
    snake_case,
)
from revaluation.data.features import FEATURE_DEFINITIONS
from revaluation.data.load import load_test, load_train
from revaluation.data.synthetic import SYNTHETIC_SPECS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dictionary")

HEADER_FILL = PatternFill("solid", fgColor="1F2D3D")
HEADER_FONT = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
BODY_FONT = Font(name="Calibri", size=9.5)
THIN = Side(style="thin", color="D7DDE6")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def parse_description() -> dict[str, str]:
    """Extract the short definition of each raw variable from the Kaggle file."""
    if not DATA_DESCRIPTION.exists():
        return {}
    text = DATA_DESCRIPTION.read_text(encoding="utf-8", errors="ignore")
    definitions: dict[str, str] = {}
    pattern = re.compile(r"^([A-Za-z0-9]+):\s*(.*)$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            key, description = match.group(1), match.group(2).strip()
            if description and key not in definitions:
                definitions[key] = re.sub(r"\s+", " ", description)
    return definitions


def raw_table() -> pd.DataFrame:
    train = load_train()
    test = load_test()
    descriptions = parse_description()
    rows = []
    for column in train.columns:
        series = train[column]
        if pd.api.types.is_numeric_dtype(series):
            dtype = "numeric"
            example = f"{series.min():.4g} to {series.max():.4g}"
        else:
            dtype = "categorical"
            example = ", ".join(str(value) for value in series.dropna().unique()[:4])
        role = (
            "identifier"
            if column == ID_COLUMN
            else "target"
            if column == "SalePrice"
            else "feature"
        )
        rows.append(
            {
                "Field": column,
                "Type": dtype,
                "Role": role,
                "Missing in train": int(series.isna().sum()),
                "Missing in test": int(test[column].isna().sum()) if column in test else 0,
                "Valid range or levels": example,
                "Definition": descriptions.get(column, "See Kaggle data_description.txt."),
            }
        )
    return pd.DataFrame(rows)


def synthetic_table() -> pd.DataFrame:
    frame = pd.DataFrame(list(SYNTHETIC_SPECS))
    return frame.rename(
        columns={
            "column": "Field",
            "dtype": "Type",
            "unit": "Unit",
            "range": "Valid range",
            "logic": "Generation logic",
            "assumption": "Business assumption",
        }
    )


def engineered_table() -> pd.DataFrame:
    return pd.DataFrame(list(FEATURE_DEFINITIONS)).rename(
        columns={
            "feature": "Field",
            "formula": "Formula",
            "unit": "Unit",
            "rationale": "Rationale",
        }
    )


def cleaning_table() -> pd.DataFrame:
    rows = []
    for column in ABSENCE_TO_NONE:
        rows.append(
            {
                "Field": snake_case(column),
                "Rule": "Missing means the feature is absent",
                "Treatment": "Impute the literal 'None'",
            }
        )
    for column in ABSENCE_TO_ZERO:
        rows.append(
            {
                "Field": snake_case(column),
                "Rule": "Missing means a zero quantity",
                "Treatment": "Impute 0",
            }
        )
    for column in MODE_IMPUTED:
        rows.append(
            {
                "Field": snake_case(column),
                "Rule": "Unstructured missingness",
                "Treatment": "Impute the training mode",
            }
        )
    rows.extend(
        [
            {
                "Field": "lot_frontage",
                "Rule": "Unstructured missingness (259 training records)",
                "Treatment": "Impute the neighbourhood median, then the global median",
            },
            {
                "Field": "garage_yr_blt",
                "Rule": "No garage or missing permit year",
                "Treatment": "Construction year, adjusted by the median garage lag",
            },
            {
                "Field": "gr_liv_area / SalePrice",
                "Rule": OUTLIER_RULES["implausible_area_price"],
                "Treatment": "Remove from the training set (2 records)",
            },
            {
                "Field": "lot_area",
                "Rule": OUTLIER_RULES["extreme_lot"],
                "Treatment": "Flag only; no removal",
            },
        ]
    )
    return pd.DataFrame(rows)


def write_markdown(raw: pd.DataFrame, synthetic: pd.DataFrame, engineered: pd.DataFrame,
                   cleaning: pd.DataFrame, path: Path) -> None:
    intro = f"""# Data dictionary

## 1. Purpose and sources

This document describes every field used by the residential valuation platform.
It covers three layers:

1. **Base data** - the {len(raw)} columns of the Kaggle *House Prices - Advanced
   Regression Techniques* competition, which contains {len(load_train())} historical
   sales in Ames, Iowa with 79 explanatory attributes.
2. **Synthetic contextual data** - {len(synthetic)} fields generated in Python with
   documented business logic. These fields are produced for every property, including
   the competition test records, and are never derived from the sale price.
3. **Engineered features** - {len(engineered)} derived variables constructed during
   feature engineering, with their formula, unit and business rationale.

The target variable is `SalePrice`, expressed in US dollars and modelled on the
natural-log scale as `log_price = ln(1 + SalePrice)`.

## 2. Data quality conventions

| Convention | Meaning | Example |
|---|---|---|
| Structural absence | The property does not have the feature | A house without a basement has no basement quality |
| Genuine missingness | The value is unknown and must be imputed | Lot frontage, recorded for 1,201 of 1,460 training properties |
| Zero quantity | The feature exists conceptually but has zero extent | Garage area of a house without a garage |

Every cleaning rule and its before/after impact is recorded in
`artifacts/metrics/cleaning_report.json`.

## 3. Base data fields

The table below lists every raw column, its type, role, missingness and the
definition taken from the competition documentation.

{raw.to_markdown(index=False)}

## 4. Synthetic contextual fields

Each synthetic field is documented with its data type, unit, valid range, the exact
generation logic and the business assumption behind it. Generation is deterministic
for a given seed and input order, and the generator is seeded with 42.

{synthetic.to_markdown(index=False)}

## 5. Engineered features

Derived features are computed by `revaluation.data.features.engineer_features`.
The function is row-wise, which guarantees that the same transformation can be
applied to a single listing at inference time.

{engineered.to_markdown(index=False)}

## 6. Cleaning rules

{cleaning.to_markdown(index=False)}

## 7. Leakage controls

The following fields are available in the analytical table but are deliberately
excluded from the model inputs:

| Field | Reason |
|---|---|
| `SalePrice`, `log_price`, `price_per_sqm` | Direct functions of the target |
| `sale_condition`, `sale_type` | Observed only after the transaction closes |
| `days_on_market`, `listing_date` | Marketing outcome, unknown at listing time |
| `yr_sold`, `mo_sold` | Replaced by the macro-financial context of the sale month |
| `id`, `source_file`, `sale_date` | Identifiers and bookkeeping |

## 8. Versioning

The dictionary is generated by `scripts/07_export_data_dictionary.py`. Any change
to the synthetic specification or to the feature engineering code must be followed
by a regeneration of this document and of `data_dictionary.xlsx`.
"""
    path.write_text(intro, encoding="utf-8")
    logger.info("Markdown dictionary written to %s", path)


def autosize(worksheet) -> None:
    for column_cells in worksheet.columns:
        length = max(
            (len(str(cell.value)) if cell.value is not None else 0) for cell in column_cells
        )
        width = min(max(12, length + 2), 78)
        worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width


def style_sheet(worksheet) -> None:
    for cell in worksheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY_FONT
            cell.border = BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    autosize(worksheet)


def write_workbook(raw: pd.DataFrame, synthetic: pd.DataFrame, engineered: pd.DataFrame,
                   cleaning: pd.DataFrame, path: Path) -> None:
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Overview"
    overview.append(["Document", "Data dictionary for the residential valuation platform"])
    overview.append(["Base data", "Kaggle House Prices - Advanced Regression Techniques (Ames, Iowa)"])
    overview.append(["Training records", len(load_train())])
    overview.append(["Evaluation records", len(load_test())])
    overview.append(["Target", "SalePrice (USD), modelled as ln(1 + SalePrice)"])
    overview.append(["Synthetic fields", len(synthetic)])
    overview.append(["Engineered features", len(engineered)])
    overview.append(["Random seed", 42])
    overview.append(["Cleaning log", "artifacts/metrics/cleaning_report.json"])
    overview.append(["Generation code", "src/revaluation/data/synthetic.py"])
    overview.append(["Feature code", "src/revaluation/data/features.py"])
    for row in overview.iter_rows():
        row[0].font = Font(bold=True, name="Calibri", size=10)
    overview.column_dimensions["A"].width = 24
    overview.column_dimensions["B"].width = 90

    for title, frame in (
        ("Synthetic fields", synthetic),
        ("Engineered features", engineered),
        ("Base fields", raw),
        ("Cleaning rules", cleaning),
    ):
        worksheet = workbook.create_sheet(title=title)
        worksheet.append(list(frame.columns))
        for record in frame.itertuples(index=False):
            worksheet.append([None if pd.isna(value) else value for value in record])
        style_sheet(worksheet)

    workbook.save(path)
    logger.info("Excel dictionary written to %s", path)


def main() -> None:
    raw = raw_table()
    synthetic = synthetic_table()
    engineered = engineered_table()
    cleaning = cleaning_table()

    write_markdown(raw, synthetic, engineered, cleaning, DOCS_DIR / "data_dictionary.md")
    write_workbook(raw, synthetic, engineered, cleaning, DOCS_DIR / "data_dictionary.xlsx")
    print(f"raw fields: {len(raw)}; synthetic: {len(synthetic)}; engineered: {len(engineered)}")


if __name__ == "__main__":
    main()
