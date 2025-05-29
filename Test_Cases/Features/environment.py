# Test_Cases/Features/environment.py

# ***********************************************************************************************************************
# File         : environment.py
# Description  : Behave hooks for launching browser, starting reports, and tracking
#                self-healing metrics per scenario.
# Author       : Aniket Pathare (AI-enhanced)
# Date Updated : 2025-05-11
# ***********************************************************************************************************************

import os
import pandas as pd

# import the healing counter from run_controller
from run_controller import count_healing_events

from Common_Functions.CommonFunctions import (
    LaunchBrowser,
    start_word_report,
    add_screenshot_to_report,
    finalize_word_report
)

def before_scenario(context, scenario):
    """
    Behave hook: runs before every scenario.

    - Pulls TestCase_ID and URL from -D args
    - Records initial self-healing event count
    - Launches browser to given URL
    - Loads test data row into context.testdata
    - Starts Word report for this TC
    """
    # Retrieve TestCase ID and URL
    tc_id = context.config.userdata.get("testcase")
    url   = context.config.userdata.get("url")

    # Record healing count before scenario execution
    context.healed_before = count_healing_events(tc_id)

    # Launch browser session
    context.driver = LaunchBrowser(url)

    # Load test data for this TestCase
    td_file = os.path.join("Test_Data", "TestData.xlsx")
    df = pd.read_excel(td_file, sheet_name="Data")
    df.columns = df.columns.str.strip()
    row = df[df["TestCase_ID"].astype(str).str.strip() == tc_id]
    if row.empty:
        raise KeyError(f"No test data found for TestCase_ID: {tc_id}")
    context.testdata = row.iloc[0].to_dict()

    # Initialize Word report
    context.report_doc, context.report_path = start_word_report(tc_id)


def after_scenario(context, scenario):
    """
    Behave hook: runs after every scenario.

    - Counts self-healing events after scenario
    - Logs healed count into Word report
    - Finalizes and saves the Word report
    - Quits the browser session
    """
    # Calculate healed locator count
    tc_id = context.config.userdata.get("testcase")
    context.healed_after = count_healing_events(tc_id)
    healed_count = context.healed_after - context.healed_before

    # Append healing summary to report
    context.report_doc.add_paragraph(f"Healed Locators: {healed_count}")

    # Finalize and save report
    finalize_word_report(context.report_doc, context.report_path)

    # Teardown browser
    context.driver.quit()
