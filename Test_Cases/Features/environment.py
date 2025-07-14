# ***********************************************************************************************************************
# File         : environment.py
# Description  : Behave hooks for launching browser, starting reports, and tracking
#                self-healing metrics per scenario.
# Author       : Aniket Pathare (AI-enhanced)
# Date Updated : 2025-06-12
# ***********************************************************************************************************************

import os
import pandas as pd

# --- Healing event count from run_controller for stats ---
from run_controller import count_healing_events

from Common_Functions.CommonFunctions import (
    LaunchBrowser,
    start_word_report,
    add_screenshot_to_report,
    finalize_word_report
)

def before_scenario(context, scenario):
    # Retrieve TestCase ID and URL from Behave -D options
    tc_id = context.config.userdata.get("testcase")
    url   = context.config.userdata.get("url")

    # Track healing count before scenario
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
    try:
        # Calculate healed locator count
        tc_id = context.config.userdata.get("testcase")
        context.healed_after = count_healing_events(tc_id)
        healed_count = context.healed_after - context.healed_before

        # Append healing summary to report
        if hasattr(context, "report_doc") and hasattr(context, "report_path"):
            context.report_doc.add_paragraph(f"Healed Locators: {healed_count}")
            # Finalize and save report
            finalize_word_report(context.report_doc, context.report_path)

        # Teardown browser
        if hasattr(context, "driver") and context.driver:
            context.driver.quit()

    except Exception as e:
        print(f"[HOOK ERROR][after_scenario] {e}")
        # Do not re-raise: log but do not cause Behave to FAIL due to hook reporting errors
