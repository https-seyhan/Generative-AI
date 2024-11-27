#from playwright.sync_api import sync_playwright
#from playwright.sync_api import sync_playwright
from playwright.sync_api import sync_playwright, Playwright
import os



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

def capture_screenshots(playwright):
    # Launch the browser in headless mode
    browser = playwright.chromium.launch()
    page = browser.new_page()
    
    # Create a folder for screenshots
    os.makedirs("uts_screenshots", exist_ok=True)

    # Starting URL
    base_url = 'https://www.uts.edu.au/study'
    page.goto(base_url)

    # Get all links on the main page under "/study"
    links = page.locator('a[href^="/study"]').evaluate_all("elements => elements.map(e => e.href)")
    
    print(f"Found {len(links)} links.")

    # Visit each link and take a screenshot
    for idx, link in enumerate(set(links)):  # Use set() to remove duplicates
        print(f"Visiting: {link}")
        page.goto(link)
        
        # Format filename safely
        filename = f"uts_screenshots/page_{idx+1}.png"
        
        # Take screenshot
        page.screenshot(path=filename, full_page=True)
    
    # Close the browser
    browser.close()
    print("Screenshots saved in 'uts_screenshots/' folder.")


with sync_playwright() as playwright:
    #run(playwright)
    capture_screenshots(playwright)
