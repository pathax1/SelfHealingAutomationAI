# ***************************************************************************************************************************************************************************************
# File: CommonFunctions.py
# Description: Core utilities for the Self-Healing Test Automation Framework.
#              Includes element highlighting, generic actions with built-in waits,
#              locator repository lookup, and browser launch logic.
# Author: Adapted with AI-driven enhancements
# Date Updated: 2025-05-07
# ***************************************************************************************************************************************************************************************
from docx import Document
from docx.shared import Inches
from selenium.webdriver.remote.webdriver import WebDriver
import os
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from datetime import datetime
from pathlib import Path
# -------------------------------------------------------------------------------------------------------------------------------------
# Element Highlighting
# -------------------------------------------------------------------------------------------------------------------------------------

def highlight_element(driver, element, color="red", border_width="3px"):
    """
    Temporarily highlights a WebElement by drawing a colored border around it.
    """
    original_style = element.get_attribute("style") or ""
    highlight_style = f"border: {border_width} solid {color};"

    driver.execute_script(
        "arguments[0].setAttribute('style', arguments[1]);",
        element, highlight_style
    )
    import time; time.sleep(0.3)
    driver.execute_script(
        "arguments[0].setAttribute('style', arguments[1]);",
        element, original_style
    )

# -------------------------------------------------------------------------------------------------------------------------------------
# Generic Action Handler with Built-In Waits
# -------------------------------------------------------------------------------------------------------------------------------------

def iaction(driver, element, identifywith, iProperty, ivalue=None, timeout=15):
    """
    Perform a Selenium action on the given element type using the specified locator,
    with built-in waits appropriate to each element category.

    :param driver: Selenium WebDriver instance
    :param element: Logical type (Textbox, Button, Checkbox, etc.)
    :param identifywith: Locator strategy name ("XPATH", "CSS_SELECTOR", etc.)
    :param iProperty: Locator string
    :param ivalue: Optional value for input operations
    :param timeout: Maximum seconds to wait for the element
    """
    locate_by = {
        "XPATH": By.XPATH,
        "CSS_SELECTOR": By.CSS_SELECTOR,
        "ID": By.ID,
        "NAME": By.NAME,
        "CLASS_NAME": By.CLASS_NAME,
        "TAG_NAME": By.TAG_NAME,
        "LINK_TEXT": By.LINK_TEXT,
        "PARTIAL_LINK_TEXT": By.PARTIAL_LINK_TEXT,
    }
    if identifywith not in locate_by:
        raise ValueError(f"Invalid identification method: {identifywith}")
    by = locate_by[identifywith]

    # Map element types to appropriate wait conditions
    wait_conditions = {
        "Textbox":      EC.element_to_be_clickable((by, iProperty)),
        "Button":       EC.element_to_be_clickable((by, iProperty)),
        "Radio Button": EC.element_to_be_clickable((by, iProperty)),
        "Checkbox":     EC.element_to_be_clickable((by, iProperty)),
        "Hyperlink":    EC.element_to_be_clickable((by, iProperty)),
        "Image":        EC.visibility_of_element_located((by, iProperty)),
        "Element":      EC.visibility_of_element_located((by, iProperty)),
    }
    if element not in wait_conditions:
        raise ValueError(f"No wait condition defined for element type '{element}'")

    try:
        # Wait for the element
        web_elem = WebDriverWait(driver, timeout).until(wait_conditions[element])
        highlight_element(driver, web_elem)

        # Perform the action based on element type
        match element:
            case "Textbox":
                web_elem.send_keys(ivalue)
            case "Checkbox":
                if not web_elem.is_selected():
                    web_elem.click()
            case "Button" | "Hyperlink" | "Radio Button":
                web_elem.click()
            case "Image" | "Element":
                pass  # visibility check only
        return f"{element} handled using {identifywith}='{iProperty}'"

    except Exception as e:
        raise RuntimeError(
            f"Error performing action [{element}] with locator [{identifywith}:'{iProperty}']: {e}"
        )

# -------------------------------------------------------------------------------------------------------------------------------------
# Locator Repository Lookup
# -------------------------------------------------------------------------------------------------------------------------------------

_LOCATOR_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, 'Object_Repository', 'locators.json'
)
try:
    with open(_LOCATOR_PATH, 'r') as f:
        _LOCATORS = json.load(f)
except FileNotFoundError:
    _LOCATORS = {}


def get_locator(key, fallback_index=0):
    """
    Retrieve the locator tuple (strategy, value) for a logical key from locators.json.
    """
    if key not in _LOCATORS:
        raise KeyError(f"Locator key '{key}' not found in repository.")
    entries = _LOCATORS[key]
    if not (0 <= fallback_index < len(entries)):
        raise IndexError(f"Fallback index {fallback_index} out of range for '{key}'")
    loc = entries[fallback_index]
    return loc['by'], loc['value']

# -------------------------------------------------------------------------------------------------------------------------------------
# Simple Action Wrappers Using Repository
# -------------------------------------------------------------------------------------------------------------------------------------

def click(driver, key):
    """Click a Button identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Button", identifywith=by, iProperty=prop)


def click_link(driver, key):
    """Click a Hyperlink identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Hyperlink", identifywith=by, iProperty=prop)


def type_text(driver, key, text):
    """Enter text into a Textbox identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Textbox", identifywith=by, iProperty=prop, ivalue=text)


def select_radio(driver, key):
    """Select a Radio Button identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Radio Button", identifywith=by, iProperty=prop)


def toggle_checkbox(driver, key):
    """Toggle a Checkbox identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Checkbox", identifywith=by, iProperty=prop)


def verify_element(driver, key):
    """Verify a generic element (visible) identified by logical key."""
    by, prop = get_locator(key)
    return iaction(driver, element="Element", identifywith=by, iProperty=prop)

# -------------------------------------------------------------------------------------------------------------------------------------
# Browser Launch & Navigation
# -------------------------------------------------------------------------------------------------------------------------------------

def LaunchBrowser(url):
    """
    Launch Chrome browser and navigate to the specified URL.

    :param url: Fully qualified URL string
    """
    # Setup ChromeDriver service
    driver_path = os.getenv(
        'CHROMEDRIVER_PATH', os.path.join(os.getcwd(), 'chromedriver.exe')
    )
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service)

    # Navigate
    driver.maximize_window()
    driver.get(url)
    return driver



# … existing highlight_element(), iaction(), get_locator(), etc. …

# -------------------------------------------------------------------------------------------------------------------------------------
# Screenshot Helper
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# Word Report & Screenshot Helpers
# -------------------------------------------------------------------------------------------------------------------------------------


def start_word_report(test_case_id: str) -> tuple[Document, str]:
    """
    Create a new Word document under Test_Report/docs named
      {TestCase_ID}_{DD_MM_YYYY_HH_MM_SS}.docx
    and write the test description as the first paragraph.
    Returns (Document, full_path).
    """
    # load the description from the control file
    ctrl_path = os.path.join("Execution_Control_File", "ExecutionControl.xlsx")
    df_ctrl = pd.read_excel(ctrl_path)
    df_ctrl.columns = df_ctrl.columns.str.strip()
    desc_row = df_ctrl.loc[df_ctrl["TestCase_ID"].astype(str).str.strip() == test_case_id]
    if desc_row.empty:
        raise KeyError(f"No description found for {test_case_id} in ExecutionControl.xlsx")
    description = desc_row.iloc[0]["Description"]

    # prepare file path
    ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    docs_dir = os.path.join("Test_Report", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    filename = f"{test_case_id}_{ts}.docx"
    full_path = os.path.join(docs_dir, filename)

    # create and initialize document
    doc = Document()
    doc.add_heading(f"Test Report: {test_case_id}", level=1)
    doc.add_paragraph(f"Description: {description}")
    doc.add_paragraph(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.save(full_path)

    return doc, full_path

def add_screenshot_to_report(driver: WebDriver, doc: Document, doc_path: str, test_case_id: str):
    """
    Take a screenshot and append it to the given Document, then save to doc_path.
    """
    # 1) Capture the PNG
    ts = datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
    shots_dir = os.path.join("Test_Report", "screenshots")
    os.makedirs(shots_dir, exist_ok=True)
    png_name = f"{test_case_id}_{ts}.png"
    png_path = os.path.join(shots_dir, png_name)
    driver.save_screenshot(png_path)

    # 2) Build caption from ExecutionControl.xlsx
    ctrl_df = pd.read_excel(os.path.join("Execution_Control_File","ExecutionControl.xlsx"))
    ctrl_df.columns = ctrl_df.columns.str.strip()
    desc = (
        ctrl_df.loc[
            ctrl_df["TestCase_ID"].astype(str).str.strip()==test_case_id,
            "Description"
        ]
        .iat[0]
    )

    caption = f"{test_case_id} — {desc}"
    doc.add_paragraph(caption)
    doc.add_picture(png_path, width=Inches(6))

    # 3) Save back to the known path
    doc.save(doc_path)

def finalize_word_report(doc: Document, doc_path: str):
    """
    No-op (you could add footers/page-numbers here if desired).
    """
    pass
