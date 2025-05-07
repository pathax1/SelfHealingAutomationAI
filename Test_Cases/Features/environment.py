# Test_Cases/Features/environment.py

import os
import pandas as pd
from Common_Functions.CommonFunctions import LaunchBrowser

def before_scenario(context, scenario):
    # Pull URL and TestCase_ID from behave -D args
    tc_id = context.config.userdata.get("testcase")
    url   = context.config.userdata.get("url")
    if not url:
        raise KeyError("Missing URL. Make sure run_controller.py passes -D url=<site>.")
    # Launch exactly one browser here
    context.driver = LaunchBrowser(url)

    # Load the row for this TC into context.testdata
    td_file = os.path.join("Test_Data", "TestData.xlsx")
    df     = pd.read_excel(td_file, sheet_name="Data")
    df.columns = df.columns.str.strip()
    row    = df[df["TestCase_ID"].astype(str).str.strip() == tc_id]
    if row.empty:
        raise KeyError(f"No test data for {tc_id}")
    context.testdata = row.iloc[0].to_dict()

def after_scenario(context, scenario):
    # Quit that one browser
    try:
        context.driver.quit()
    except Exception:
        pass
