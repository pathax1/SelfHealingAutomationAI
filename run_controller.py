#***********************************************************************************************************************
# File         : run_controller.py
# Description  : Read data from Execution control file,
#                trigger execution of those scripts marked 'Y',
#                pass TestCase_ID + URL into Behave, collect PASS/FAIL,
#                generate Excel & Allure reports, export summary CSV,
#                then launch Streamlit dashboard and open it in browser.
# Author       : Aniket Pathare (enhanced with AI-driven Self-Healing)
# Date Updated : 2025-05-11
#***********************************************************************************************************************

import os
import sys
import glob
import pandas as pd
import subprocess
import webbrowser
import time
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
HEALING_LOG     = os.path.join("Logs", "healing.log")

# -------------------------------------------------------------------------------------------------------------------------------------
# Helper: Count self-healing events
# -------------------------------------------------------------------------------------------------------------------------------------
def count_healing_events(test_case_id=None):
    """
    Count healing events in the log. Optionally filter by TestCase_ID if logged.
    """
    count = 0
    if os.path.exists(HEALING_LOG):
        with open(HEALING_LOG, 'r') as log:
            for line in log:
                if not line.strip():
                    continue
                if test_case_id:
                    # expecting log entries contain the key in brackets
                    if f"[{test_case_id}]" in line:
                        count += 1
                else:
                    count += 1
    return count

# -------------------------------------------------------------------------------------------------------------------------------------
# Main Runner
# -------------------------------------------------------------------------------------------------------------------------------------
def main():
    # 1) Validate control file
    if not os.path.exists(CONTROL_FILE):
        print(f"[ERROR] Control file not found: {CONTROL_FILE}")
        return
    ctrl_df = pd.read_excel(CONTROL_FILE)

    # 2) Validate test data config
    if not os.path.exists(TESTDATA_FILE):
        print(f"[ERROR] Test data file not found: {TESTDATA_FILE}")
        return
    config_df = pd.read_excel(TESTDATA_FILE, sheet_name="Config")
    config_df.columns = config_df.columns.str.strip()

    # 3) Prepare report directories
    os.makedirs(ALLURE_RESULTS, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 4) Execute tests and collect summary
    summary_records = []
    for idx, row in ctrl_df.iterrows():
        flag  = str(row.get("Execution", "")).strip().upper()
        tc_id = str(row.get("TestCase_ID", "")).strip()
        env   = str(row.get("Environment", "")).strip()

        # Initialize healing count before execution
        healed_before = count_healing_events(tc_id)

        start_time = None
        end_time   = None
        duration   = None
        status     = 'SKIPPED'
        healed_count = 0

        if flag != 'Y':
            status = 'SKIPPED'
            print(f"[SKIPPED] {tc_id} (flag != Y)")
        else:
            # Lookup URL from config
            env_row = config_df.loc[config_df["Env"] == env]
            if env_row.empty:
                status = 'ERROR'
                print(f"[ERROR] No URL for environment '{env}' for {tc_id}")
            else:
                url = env_row.iloc[0]["URL"]
                # Find feature file
                pattern = os.path.join(FEATURE_DIR, f"*{tc_id}*.feature")
                matches = glob.glob(pattern)
                if not matches:
                    status = 'SKIPPED'
                    print(f"[SKIPPED] Feature file not found for {tc_id}")
                elif len(matches) > 1:
                    status = 'ERROR'
                    print(f"[ERROR] Multiple feature files found for {tc_id}: {matches}")
                else:
                    feat_file = matches[0]
                    # Execute Behave with Allure
                    print(f"[RUNNING] {tc_id} on {env} → {url}")
                    start_time = datetime.now()
                    cmd = [
                        sys.executable, "-m", "behave",
                        feat_file,
                        "-D", f"testcase={tc_id}",
                        "-D", f"url={url}",
                        "-f", "allure_behave.formatter:AllureFormatter",
                        "-o", ALLURE_RESULTS
                    ]
                    ret = subprocess.call(cmd)
                    end_time = datetime.now()
                    status = 'PASS' if ret == 0 else 'FAIL'
                    print(f"[{status}] {tc_id} ({(end_time - start_time).total_seconds():.1f}s)")

        # After execution or skip/error, count healing
        healed_after = count_healing_events(tc_id)
        healed_count = healed_after - healed_before

        # Record times/duration
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()

        summary_records.append({
            "TestCase_ID": tc_id,
            "Environment": env,
            "StartTime": start_time,
            "EndTime": end_time,
            "Duration": duration,
            "Status": status,
            "Healed_Count": healed_count
        })

        # Update status back into control file dataframe
        ctrl_df.at[idx, "Status"] = status

    # 5) Persist updated statuses back to Excel
    ctrl_df.to_excel(CONTROL_FILE, index=False)
    print(f"[INFO] Control file updated: {CONTROL_FILE}")

    # 6) Write summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] Summary CSV written: {SUMMARY_CSV}")

    # 7) Launch Streamlit dashboard
    if os.path.exists(STREAMLIT_APP):
        print(f"[INFO] Starting Streamlit: {STREAMLIT_APP}")
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP],
            cwd=os.getcwd()
        )
        time.sleep(3)
        webbrowser.open(STREAMLIT_URL)
        print(f"Dashboard is running at {STREAMLIT_URL}. Press Ctrl+C to stop.")
        proc.communicate()
    else:
        print(f"[WARN] Streamlit app not found at {STREAMLIT_APP}")

if __name__ == "__main__":
    main()
