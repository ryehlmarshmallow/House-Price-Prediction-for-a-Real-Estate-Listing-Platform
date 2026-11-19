"""Capture screenshots of the Streamlit application for layout verification.

Usage:
    uv run python scripts/capture_ui.py --url http://127.0.0.1:8501 --output artifacts/reports/ui_home.png

Requires the development dependency ``playwright`` (``uv run playwright install chromium``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def capture(url: str, output: Path, click_text: str | None = None, wait_ms: int = 3500) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1400})
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(wait_ms)
        if click_text:
            page.get_by_text(click_text, exact=True).first.click()
            page.wait_for_timeout(2500)
        page.screenshot(path=str(output), full_page=True)
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/ui_home.png"))
    parser.add_argument("--click", default=None)
    arguments = parser.parse_args()
    capture(arguments.url, arguments.output, arguments.click)
    print(f"screenshot written to {arguments.output}")
