#***********************************************************************************************************************
# File         : run_controller.py
# Description  : Read data from Execution control file,
#                trigger execution of those scripts marked 'Y',
#                pass TestCase_ID + URL into Behave, collect PASS/FAIL,
#                and generate Excel & Allure reports.
# Author       : ChatGPT (adapted)
# Date Updated : 2025-05-07
#***********************************************************************************************************************

import os
import glob
import pandas as pd
import subprocess

# --- Paths ---
CONTROL_FILE   = os.path.join("Execution_Control_File", "ExecutionControl.xlsx")
FEATURE_DIR    = os.path.join("Test_Cases", "Features")
TESTDATA_FILE  = os.path.join("Test_Data", "TestData.xlsx")
REPORT_DIR     = os.path.join("Test_Report")
ALLURE_RESULTS = os.path.join(REPORT_DIR, "allure-results")

def main():
    # 1) Load control file
    if not os.path.exists(CONTROL_FILE):
        print(f"[ERROR] COULD NOT FIND: {CONTROL_FILE}")
        return
    ctrl_df = pd.read_excel(CONTROL_FILE)

    # 2) Load environment→URL map from Config sheet
    if not os.path.exists(TESTDATA_FILE):
        print(f"[ERROR] COULD NOT FIND: {TESTDATA_FILE}")
        return
    config_df = pd.read_excel(TESTDATA_FILE, sheet_name="Config")
    config_df.columns = config_df.columns.str.strip()

    # 3) Ensure report directories exist
    os.makedirs(ALLURE_RESULTS, exist_ok=True)

    # 4) Iterate and run
    for idx, row in ctrl_df.iterrows():
        flag  = str(row.get("Execution", "")).strip().upper()
        tc_id = str(row.get("TestCase_ID", "")).strip()
        env   = str(row.get("Environment", "")).strip()

        if flag != "Y":
            print(f"[SKIPPED] Execution flag not 'Y' for {tc_id}")
            ctrl_df.at[idx, "Status"] = "SKIPPED"
            continue

        # 4a) Lookup URL for this environment
        env_row = config_df.loc[config_df["Env"] == env]
        if env_row.empty:
            print(f"[ERROR] No URL found for Environment '{env}'")
            ctrl_df.at[idx, "Status"] = "ERROR"
            continue
        url = env_row.iloc[0]["URL"]

        # 4b) Locate the feature file by pattern *{tc_id}*.feature
        pattern = os.path.join(FEATURE_DIR, f"*{tc_id}*.feature")
        matches = glob.glob(pattern)
        if not matches:
            print(f"[SKIPPED] FEATURE NOT FOUND for pattern: {pattern}")
            ctrl_df.at[idx, "Status"] = "SKIPPED"
            continue
        if len(matches) > 1:
            print(f"[ERROR] Multiple feature files match {tc_id}: {matches}")
            ctrl_df.at[idx, "Status"] = "ERROR"
            continue
        feat_file = matches[0]

        # 4c) Build Behave command
        print(f"[RUNNING] {tc_id} on {env} → {url}")
        cmd = [
            "behave",
            feat_file,
            "-D", f"testcase={tc_id}",
            "-D", f"url={url}",
            # Uncomment the next two lines if you have allure-behave installed
            # "-f", "allure_behave.formatter:AllureFormatter",
            # "-o", ALLURE_RESULTS,
        ]

        # 4d) Execute and capture result
        ret = subprocess.call(cmd)
        status = "PASS" if ret == 0 else "FAIL"
        ctrl_df.at[idx, "Status"] = status
        print(f"[{status}] {tc_id}")

    # 5) Persist updated statuses
    ctrl_df.to_excel(CONTROL_FILE, index=False)
    print(f"[INFO] Execution finished. Control file updated: {CONTROL_FILE}")

if __name__ == "__main__":
    main()
