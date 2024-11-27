#from playwright.sync_api import sync_playwright
#from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright, Playwright


def run(playwright):
    # launch the browser
    browser = playwright.chromium.launch(headless=False)
    # opens a new browser page
    page = browser.new_page()
    # navigate to the website
    page.goto('https://example.com')
    # take a full-page screenshot
    page.screenshot(path='/home/saul/Desktop/RE/screenshots/example.png', full_page=True)
    # always close the browser
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
