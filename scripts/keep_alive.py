"""Visit the live dashboard in a headless browser so it does not fall asleep.

Streamlit Community Cloud puts an app to sleep after a stretch with no visitors.
A plain HTTP request does not count as a visit, so this opens the page in a
real browser, presses the wake-up button if the app is asleep, and waits until
the dashboard has loaded. Run by the keep-alive GitHub Actions workflow.
"""

import re
import sys

from playwright.sync_api import Page, sync_playwright

APP_URL = "https://azizdaim-turbofan.streamlit.app"
TIMEOUT_MS = 180_000


def dashboard_loaded(page: Page) -> bool:
    """Return True once the app's navigation has rendered in any frame."""
    return any(
        frame.get_by_text("Fleet overview").count() > 0 for frame in page.frames
    )


def main() -> int:
    """Open the app, wake it if needed, and report whether it loaded."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(APP_URL, timeout=TIMEOUT_MS)

        wake = page.get_by_role("button", name=re.compile("get this app back up"))
        try:
            wake.wait_for(timeout=10_000)
            print("App was asleep, waking it up")
            wake.click()
        except Exception:
            print("App was awake")

        for _ in range(TIMEOUT_MS // 2_000):
            if dashboard_loaded(page):
                print("Dashboard loaded")
                browser.close()
                return 0
            page.wait_for_timeout(2_000)

        print("Dashboard did not load in time", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
