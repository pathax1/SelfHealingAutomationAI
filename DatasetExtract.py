# ********************************************************************************************************
# File Name     : modular_extractor.py
# Description   : Modular self-healing AI extractor with smart DOM snippet for training data.
# Author        : Aniket Pathare
# Date Modified : 2025-07-13
# ********************************************************************************************************

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from lxml import html
import time
import csv
import os

def launch_browser(url):
    chrome_options = Options()
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    print(f"[INFO] Browser launched at: {url}")
    return driver

def click_element(driver, by, value, timeout=10):
    try:
        print(f"[ACTION] Trying click on: ({by}, {value})")
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((by, value))).click()
        print("[ CLICKED]")
        time.sleep(2)
    except Exception as e:
        print(f"[ FAILED] Could not click: {e}")

def wait_for_element(driver, by, value, timeout=10):
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by, value)))
        print(f"[🟢 READY] Element found: ({by}, {value})")
    except Exception as e:
        print(f"[ TIMEOUT] Could not find element: {e}")

def generate_element_key(tag_name, tag_id, tag_name_attr, tag_class, tag_data_qa, tag_text):
    """Generate a robust, readable element key for training."""
    if tag_data_qa:
        return f"qaid_{tag_name}_{tag_data_qa}"
    elif tag_id:
        return f"id_{tag_name}_{tag_id}"
    elif tag_name_attr:
        return f"name_{tag_name}_{tag_name_attr}"
    elif tag_class:
        return f"cls_{tag_name}_{tag_class.split()[0]}"
    elif tag_text:
        return f"txt_{tag_name}_{tag_text[:10].replace(' ', '_')}"
    else:
        return f"unknown_{tag_name}"

def get_best_dom_match_by_key_from_soup(soup, element_key, class_val=None):
    """Return best-matching element for the key among candidates with an exact class string if provided."""
    import difflib
    # Only elements with the exact class string, if class_val is specified
    if class_val:
        candidates = []
        for tag in soup.find_all(['button', 'a', 'input']):
            tag_class = tag.get("class")
            if tag_class and " ".join(tag_class) == class_val:
                candidates.append(tag)
        if not candidates:
            return None
        print(f"[EXTRACTOR] Found {len(candidates)} candidates with exact class '{class_val}'")
    else:
        candidates = soup.find_all(['button', 'a', 'input'])
    best_score = 0
    best_tag = None
    search_key = element_key.lower().replace("_", "").replace("-", "").replace(" ", "")
    for tag in candidates:
        attr_string = ""
        for attr in ['id', 'data-qa-id', 'name']:
            attr_string += str(tag.get(attr, ""))
        attr_string += tag.text or ""
        attr_string = attr_string.lower().replace("_", "").replace("-", "").replace(" ", "")
        score = difflib.SequenceMatcher(None, search_key, attr_string).ratio()
        if score > best_score:
            best_score = score
            best_tag = tag
    if best_tag:
        print(f"[EXTRACTOR] Selected best candidate for '{element_key}' with score {best_score:.2f}: {str(best_tag)[:100]}")
    return str(best_tag) if best_tag else None

def extract_dom_snippet_for_healing(driver, locator, max_length=512, key=None):
    """
    For training: Always match the runtime healing logic!
    - If class exists, compare all DOM elements with the exact full class string, pick best match by key.
    - Else, try direct locator.
    - Else, fallback to fuzzy key match.
    """
    class_val = locator.get("class", "").strip() if "class" in locator else ""
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    if class_val:
        # 1. All elements with exact class string (space-safe, all classes in same order)
        result = get_best_dom_match_by_key_from_soup(soup, key, class_val)
        if result:
            print(f"[EXTRACTOR] DOM extracted for key '{key}' by exact class '{class_val}'")
            return result[:max_length]
    # 2. Direct locator as fallback
    by_map = {
        "XPATH": By.XPATH,
        "CSS_SELECTOR": By.CSS_SELECTOR,
        "ID": By.ID,
        "NAME": By.NAME,
        "CLASS_NAME": By.CLASS_NAME,
        "TAG_NAME": By.TAG_NAME,
        "LINK_TEXT": By.LINK_TEXT,
        "PARTIAL_LINK_TEXT": By.PARTIAL_LINK_TEXT,
    }
    try:
        by = by_map.get(locator["by"])
        elem = driver.find_element(by, locator["value"])
        snippet = elem.get_attribute("outerHTML")
        if snippet:
            print(f"[EXTRACTOR] DOM extracted for key '{key}' by direct locator ({locator['by']}, {locator['value']})")
            return snippet[:max_length]
        parent = elem.find_element(By.XPATH, "..")
        snippet = parent.get_attribute("outerHTML")
        if snippet:
            print(f"[EXTRACTOR] DOM extracted for key '{key}' by parent of direct locator")
            return snippet[:max_length]
    except Exception as e:
        print(f"[DEBUG] Could not extract element or parent: {e}")
    # 3. Fallback: fuzzy key match in all actionable tags
    result = get_best_dom_match_by_key_from_soup(soup, key)
    if result:
        print(f"[EXTRACTOR] DOM extracted for key '{key}' by fuzzy key match")
        return result[:max_length]
    print("[DEBUG] All DOM extraction methods failed, returning error HTML.")
    return "<html>ERROR: could not extract dom</html>"

def extract_elements(driver, output_csv="ai_selector_training_data.csv"):
    print("[INFO] Extracting elements...")

    try:
        body = driver.find_element(By.TAG_NAME, "body")
        for _ in range(5):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.8)
    except Exception:
        pass

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    target_tags = ["a", "button", "input", "span", "div", "label", "h1", "h2", "h3"]

    rows = []
    for tag in soup.find_all(target_tags):
        tag_name = tag.name
        tag_id = tag.get("id", "")
        tag_class = " ".join(tag.get("class", [])) if tag.get("class") else ""
        tag_name_attr = tag.get("name", "")
        tag_text = tag.get_text(strip=True)
        tag_data_qa = tag.get("data-qa-id", "")

        element_key = generate_element_key(tag_name, tag_id, tag_name_attr, tag_class, tag_data_qa, tag_text)

        # Build locator for healing function
        locator = {}
        if tag_data_qa:
            locator = {"by": "CSS_SELECTOR", "value": f"{tag_name}[data-qa-id='{tag_data_qa}']"}
        elif tag_id:
            locator = {"by": "ID", "value": tag_id}
        elif tag_name_attr:
            locator = {"by": "NAME", "value": tag_name_attr}
        elif tag_class:
            locator = {"by": "CSS_SELECTOR", "value": f"{tag_name}.{tag_class.split()[0]}", "class": tag_class}
        else:
            locator = {"by": "TAG_NAME", "value": tag_name}

        dom_snippet = extract_dom_snippet_for_healing(driver, locator, max_length=512, key=element_key)
        if dom_snippet.startswith("<html>ERROR"):
            print(f"[SKIP] Could not extract DOM for {element_key}")
            continue

        print(f"[SNIPPET] Element key: {element_key}")
        print(f"[SNIPPET] DOM Snippet: {dom_snippet[:80].replace(chr(10),' ')} ...")

        # Build XPath/CSS selectors for ground truth
        if tag_data_qa:
            xpath = f"//{tag_name}[@data-qa-id='{tag_data_qa}']"
            css = f"{tag_name}[data-qa-id='{tag_data_qa}']"
        elif tag_id:
            xpath = f"//{tag_name}[@id='{tag_id}']"
            css = f"{tag_name}#{tag_id}"
        elif tag_name_attr:
            xpath = f"//{tag_name}[@name='{tag_name_attr}']"
            css = f"{tag_name}[name='{tag_name_attr}']"
        elif tag_class:
            first_class = tag_class.split()[0]
            xpath = f"//{tag_name}[contains(@class, '{first_class}')]"
            css = f"{tag_name}.{first_class}"
        elif tag_text:
            safe_text = tag_text[:10].replace("'", "")
            xpath = f"//{tag_name}[contains(text(),'{safe_text}')]"
            css = tag_name
        else:
            continue

        xpath = xpath.replace('"', "'").strip()
        if not xpath.startswith("//"):
            continue

        rows.append([element_key, dom_snippet, xpath, css, tag_class])

    write_header = not os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["element_key", "input_text", "target_text_xpath", "target_text_css", "class"])
        writer.writerows(rows)

    print(f"[ DONE] Extracted {len(rows)} elements. Appended to {output_csv}")

if __name__ == "__main__":
    driver = launch_browser("https://www.zara.com/ie/")
    extract_elements(driver)  # Step 1

    click_element(driver, By.XPATH, "//button[@id='onetrust-reject-all-handler']")
    extract_elements(driver)  # Step 2

    click_element(driver, By.XPATH, "//a[@data-qa-id='layout-header-user-logon']")
    wait_for_element(driver, By.XPATH, "//button[@data-qa-id='oauth-logon-button']")
    extract_elements(driver)  # Step 3

    #  Smart JS click to avoid obstruction
    driver.execute_script("document.querySelector('button[data-qa-id=\"oauth-logon-button\"]').click()")

    wait_for_element(driver, By.XPATH, "//input[@id='zds-:r5:']")
    time.sleep(2)
    extract_elements(driver)  # Step 4

    click_element(driver, By.XPATH, "//a[@data-qa-id='logon-view-alternate-button']")
    wait_for_element(driver, By.XPATH, "//input[@data-qa-input-qualifier='email']")
    extract_elements(driver)  # Step 5
    time.sleep(2)

    click_element(driver, By.XPATH, "//button[@data-qa-id='layout-header-toggle-menu']")
    extract_elements(driver)  # Step 5
    time.sleep(2)

    click_element(driver, By.XPATH,"//div[@data-qa-qualifier='category-level-1']//a[.//span[text()='MAN']]")
    time.sleep(2)

    click_element(driver, By.XPATH,"//li[@data-qa-qualifier='category-level-3']//a[@class='layout-categories-category__link link' and .//span[text()='VIEW ALL']]")
    extract_elements(driver)  # Step 5

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    extract_elements(driver)  # Step 5
    time.sleep(2)

    driver.quit()
