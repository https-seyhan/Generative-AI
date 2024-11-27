#from playwright.sync_api import sync_playwright
#from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright, Playwright


def run(playwright):
    # launch the browser
    browser = playwright.chromium.launch(headless=False)
    # opens a new browser page
    page = browser.new_page()
    # navigate to the website
    page_addr = 'https://www.uts.edu.au/'
    page.goto(page_addr)
    title = page.title()
    print("Page Title:", title)
    main_heading = page.locator('h1').first.inner_text()
    print("Main Heading:", main_heading)
    # Extract text from a specific section
    #section_text = page.locator('section.featured-news').inner_text()
    #print(section_text)
    # take a full-page screenshot
    page.screenshot(path='/home/saul/Desktop/RE/screenshots/' + page_addr.sep['.'][0] +'.png', full_page=True)
    # always close the browser
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
