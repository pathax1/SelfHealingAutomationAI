#***********************************************************************************************************************
# File         : run_controller.py
# Description  : Read data from Execution control file,
#                trigger execution of those scripts marked 'Y',
#                pass TestCase_ID + URL into Behave, collect PASS/FAIL,
#                generate Excel & Allure reports, export summary CSV,
#                then launch Streamlit dashboard and open it in browser.
# Author       : ChatGPT (adapted)
# Date Updated : 2025-05-09
#***********************************************************************************************************************

import os
import sys
import glob
import pandas as pd
import subprocess
import webbrowser
from datetime import datetime

# --- Paths ---
CONTROL_FILE    = os.path.join("Execution_Control_File", "ExecutionControl.xlsx")
FEATURE_DIR     = os.path.join("Test_Cases", "Features")
TESTDATA_FILE   = os.path.join("Test_Data", "TestData.xlsx")
REPORT_DIR      = os.path.join("Test_Report")
ALLURE_RESULTS  = os.path.join(REPORT_DIR, "allure-results")
SUMMARY_CSV     = os.path.join(REPORT_DIR, "results_summary.csv")
STREAMLIT_APP   = os.path.join("Streamlit_Dashboard", "streamlit_app.py")
STREAMLIT_URL   = "http://localhost:8501"

def main():
    # 1) Load control file
    if not os.path.exists(CONTROL_FILE):
        print(f"[ERROR] COULD NOT FIND: {CONTROL_FILE}")
        return
    ctrl_df = pd.read_excel(CONTROL_FILE)

    # 2) Load Config sheet for environment→URL
    if not os.path.exists(TESTDATA_FILE):
        print(f"[ERROR] COULD NOT FIND: {TESTDATA_FILE}")
        return
    config_df = pd.read_excel(TESTDATA_FILE, sheet_name="Config")
    config_df.columns = config_df.columns.str.strip()

    # 3) Prepare report directories
    os.makedirs(ALLURE_RESULTS, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 4) Run each flagged test and collect summary
    summary_records = []
    for idx, row in ctrl_df.iterrows():
        flag  = str(row.get("Execution", "")).strip().upper()
        tc_id = str(row.get("TestCase_ID", "")).strip()
        env   = str(row.get("Environment", "")).strip()

        if flag != "Y":
            ctrl_df.at[idx, "Status"] = "SKIPPED"
            summary_records.append({
                "TestCase_ID": tc_id, "Environment": env,
                "StartTime": None, "EndTime": None, "Duration": None,
                "Status": "SKIPPED"
            })
            print(f"[SKIPPED] {tc_id} (flag != Y)")
            continue

        # 4a) Lookup URL
        env_row = config_df.loc[config_df["Env"] == env]
        if env_row.empty:
            ctrl_df.at[idx, "Status"] = "ERROR"
            summary_records.append({
                "TestCase_ID": tc_id, "Environment": env,
                "StartTime": None, "EndTime": None, "Duration": None,
                "Status": "ERROR"
            })
            print(f"[ERROR] No URL for env '{env}'")
            continue
        url = env_row.iloc[0]["URL"]

        # 4b) Find feature file
        pattern = os.path.join(FEATURE_DIR, f"*{tc_id}*.feature")
        matches = glob.glob(pattern)
        if not matches or len(matches) > 1:
            status = "SKIPPED" if not matches else "ERROR"
            ctrl_df.at[idx, "Status"] = status
            summary_records.append({
                "TestCase_ID": tc_id, "Environment": env,
                "StartTime": None, "EndTime": None, "Duration": None,
                "Status": status
            })
            print(f"[{status}] Feature lookup for '{tc_id}': {matches}")
            continue
        feat_file = matches[0]

        # 4c) Execute Behave
        print(f"[RUNNING] {tc_id} on {env} → {url}")
        start_time = datetime.now()
        cmd = [
            sys.executable, "-m", "behave",
            feat_file,
            "-D", f"testcase={tc_id}",
            "-D", f"url={url}",
            # "-f", "allure_behave.formatter:AllureFormatter",
            # "-o", ALLURE_RESULTS,
        ]
        ret = subprocess.call(cmd)
        end_time = datetime.now()

        status = "PASS" if ret == 0 else "FAIL"
        ctrl_df.at[idx, "Status"] = status
        duration = (end_time - start_time).total_seconds()
        summary_records.append({
            "TestCase_ID": tc_id, "Environment": env,
            "StartTime": start_time, "EndTime": end_time,
            "Duration": duration, "Status": status
        })
        print(f"[{status}] {tc_id} ({duration:.1f}s)")

    # 5) Persist updated statuses
    ctrl_df.to_excel(CONTROL_FILE, index=False)
    print(f"[INFO] Control file updated: {CONTROL_FILE}")

    # 6) Write summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] Summary CSV written: {SUMMARY_CSV}")

    # 7) Launch Streamlit dashboard and open in browser
    if os.path.exists(STREAMLIT_APP):
        print(f"[INFO] Starting Streamlit: {STREAMLIT_APP}")
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP],
            cwd=os.getcwd()
        )
        # give it a moment to start
        import time; time.sleep(3)
        webbrowser.open(STREAMLIT_URL)
        print(f"Dashboard is running at {STREAMLIT_URL}. Press Ctrl+C to stop.")
        proc.communicate()
    else:
        print(f"[WARN] Streamlit app not found at {STREAMLIT_APP}")

if __name__ == "__main__":
    main()
