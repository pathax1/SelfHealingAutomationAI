#***********************************************************************************************************************
# File         : run_controller.py
# Description  : Orchestrates sequential test execution per ExecutionControl.xlsx,
#                dynamically picks model for AI healing per test run, collects stats,
#                generates Excel & Allure reports, writes summary CSV for dashboard,
#                launches Streamlit dashboard (if not already running).
# Author       : Aniket Pathare (enhanced with AI-driven Self-Healing)
# Date Updated : 2025-06-27
#***********************************************************************************************************************

import os
import sys
import glob
import pandas as pd
import subprocess
import webbrowser
import time
import json
from datetime import datetime
import socket
import threading

# --- System monitoring imports
import psutil

# --- Paths ---
CONTROL_FILE    = os.path.join("Execution_Control_File", "ExecutionControl.xlsx")
FEATURE_DIR     = os.path.join("Test_Cases", "Features")
TESTDATA_FILE   = os.path.join("Test_Data", "TestData.xlsx")
REPORT_DIR      = os.path.join("Test_Report")
ALLURE_RESULTS  = os.path.join(REPORT_DIR, "allure-results")
SUMMARY_CSV     = os.path.join(REPORT_DIR, "results_summary.csv")
STREAMLIT_APP   = os.path.join("Streamlit_Dashboard", "streamlit_app.py")
STREAMLIT_URL   = "http://localhost:8501"
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
HEALING_LOG     = os.path.join(BASE_DIR, "Logs", "healing.log")
SYSTEM_STATS_LOG= os.path.join(BASE_DIR, "Logs", "system_stats.log")

# --- Model Mapping ---
MODEL_MAP = {
    "Llama":   r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "Gemma":   r"C:\Users\Autom\PycharmProjects\Automation AI\Models\gemma-7b-it-Q5_K_M.gguf",
    "Mistral": r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf",
    "Phi":     r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Phi-3-mini-4k-instruct.Q4_0.gguf",
    "BART":    r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model",
    "T5":      r"C:\Users\Autom\PycharmProjects\Automation AI\Self_Healing_AI\trained_xpath_t5",
}

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def count_healing_events(test_case_id=None, model_name=None, start_time=None, end_time=None):
    import re
    count = 0
    times = []
    healing_times_sec = []
    if os.path.exists(HEALING_LOG):
        with open(HEALING_LOG, 'r', encoding="utf-8") as log:
            for line in log:
                if not line.strip():
                    continue
                tc_match = f"[TC:{test_case_id}]" if test_case_id else ""
                model_match = f"[MODEL:{model_name}]" if model_name else ""
                time_str = line.split(']')[0][1:]
                try:
                    evt_time = datetime.fromisoformat(time_str)
                except:
                    evt_time = None
                if tc_match in line and (not model_match or model_match in line):
                    if evt_time and start_time and end_time:
                        if not (start_time <= evt_time <= end_time):
                            continue
                    count += 1
                    times.append(evt_time)
                    # Extract HealingTime if present
                    m = re.search(r"\[HealingTime:([0-9.]+)\]", line)
                    if m:
                        try:
                            healing_times_sec.append(float(m.group(1)))
                        except Exception:
                            pass
    return count, times, healing_times_sec

def get_json_status(tc_id):
    pattern = os.path.join(REPORT_DIR, f"*{tc_id}*-result.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    file = matches[0]
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        status = str(data.get("status", "")).strip().lower()
        if status == "passed":
            return "PASS"
        elif status in ("failed", "broken"):
            return "FAIL"
        elif status == "skipped":
            return "SKIPPED"
        else:
            return status.upper() if status else None
    except Exception as ex:
        print(f"[WARN] Could not parse {file}: {ex}")
        return None

# --- SYSTEM MONITORING THREAD ---
def monitor_system_stats(stop_event, log_path, tc_id, model_name, stats_collector):
    min_cpu, max_cpu, min_ram, max_ram = 100.0, 0.0, 100.0, 0.0
    sum_cpu, sum_ram, count = 0.0, 0.0, 0
    spike_threshold = 80  # %
    with open(log_path, "a") as logf:
        while not stop_event.is_set():
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            # For dashboard stats
            min_cpu = min(min_cpu, cpu)
            max_cpu = max(max_cpu, cpu)
            min_ram = min(min_ram, ram)
            max_ram = max(max_ram, ram)
            sum_cpu += cpu
            sum_ram += ram
            count += 1
            # If spike, log
            if cpu >= spike_threshold or ram >= spike_threshold:
                logf.write(f"[{datetime.now()}][TC:{tc_id}][MODEL:{model_name}] [CPU:{cpu}%][RAM:{ram}%]\n")
            time.sleep(1)
    # Collect stats for summary
    if count > 0:
        stats_collector["min_cpu"] = min_cpu
        stats_collector["max_cpu"] = max_cpu
        stats_collector["avg_cpu"] = sum_cpu / count
        stats_collector["min_ram"] = min_ram
        stats_collector["max_ram"] = max_ram
        stats_collector["avg_ram"] = sum_ram / count
        stats_collector["cpu_samples"] = count

def main():
    if not os.path.exists(CONTROL_FILE):
        print(f"[ERROR] Control file not found: {CONTROL_FILE}")
        return
    ctrl_df = pd.read_excel(CONTROL_FILE)
    if not os.path.exists(TESTDATA_FILE):
        print(f"[ERROR] Test data file not found: {TESTDATA_FILE}")
        return
    config_df = pd.read_excel(TESTDATA_FILE, sheet_name="Config")
    config_df.columns = config_df.columns.str.strip()
    os.makedirs(ALLURE_RESULTS, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "Logs"), exist_ok=True)
    summary_records = []
    for idx, row in ctrl_df.iterrows():
        flag  = str(row.get("Execution", "")).strip().upper()
        tc_id = str(row.get("TestCase_ID", "")).strip()
        env   = str(row.get("Environment", "")).strip()
        model_name = str(row.get("Model", "Llama")).strip()
        if model_name not in MODEL_MAP:
            print(f"[ERROR] Invalid model: {model_name}. Skipping row.")
            continue
        os.environ["MODEL_PATH"] = MODEL_MAP[model_name]
        os.environ["MODEL_NAME"] = model_name
        healed_before, _, _ = count_healing_events(tc_id, model_name)
        start_time = None
        end_time   = None
        duration   = None
        status     = 'SKIPPED'
        healed_count = 0

        # SYSTEM STATS MONITORING
        sys_stats = {}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_system_stats, args=(stop_event, SYSTEM_STATS_LOG, tc_id, model_name, sys_stats))
        if flag == 'Y':
            monitor_thread.start()

        if flag != 'Y':
            status = 'SKIPPED'
            print(f"[SKIPPED] {tc_id} (flag != Y)")
        else:
            env_row = config_df.loc[config_df["Env"] == env]
            if env_row.empty:
                status = 'ERROR'
                print(f"[ERROR] No URL for environment '{env}' for {tc_id}")
            else:
                url = env_row.iloc[0]["URL"]
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
                    print(f"[RUNNING] {tc_id} on {env} → {url} with {model_name}")
                    start_time = datetime.now()
                    cmd = [
                        sys.executable, "-m", "behave",
                        "--no-capture",
                        feat_file,
                        "-D", f"testcase={tc_id}",
                        "-D", f"url={url}",
                        "-D", f"model={model_name}",
                        "-f", "allure_behave.formatter:AllureFormatter",
                        "-o", ALLURE_RESULTS
                    ]
                    ret = subprocess.call(cmd)
                    end_time = datetime.now()
                    # --------- CRITICAL CHANGE: Parse Allure/Behave JSON status, NOT just ret code ---------
                    json_status = get_json_status(tc_id)
                    if json_status:
                        status = json_status
                    else:
                        status = 'PASS' if ret == 0 else 'FAIL'
                    print(f"[{status}] {tc_id} ({(end_time - start_time).total_seconds():.1f}s) [{model_name}]")
        if flag == 'Y':
            stop_event.set()
            monitor_thread.join()

        healed_after, healing_times, healing_times_sec = count_healing_events(tc_id, model_name, start_time, end_time)
        healed_count = healed_after - healed_before
        healing_latencies = []
        if healing_times:
            healing_times_sorted = sorted([t for t in healing_times if t])
            for i, t in enumerate(healing_times_sorted):
                if i == 0:
                    healing_latencies.append(0)
                else:
                    healing_latencies.append((t - healing_times_sorted[i-1]).total_seconds())
        avg_healing_time = round(sum(healing_latencies)/len(healing_latencies), 2) if healing_latencies else 0
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()
        # Extend summary with system stats
        summary_records.append({
            "TestCase_ID": tc_id,
            "Environment": env,
            "Model": model_name,
            "StartTime": start_time,
            "EndTime": end_time,
            "Duration": duration,
            "Status": status,
            "Healed_Count": healed_count,
            "Avg_Healing_Time": avg_healing_time,
            "Total_Healing_Events": healed_count,
            "Min_CPU%": sys_stats.get("min_cpu"),
            "Max_CPU%": sys_stats.get("max_cpu"),
            "Avg_CPU%": sys_stats.get("avg_cpu"),
            "Min_RAM%": sys_stats.get("min_ram"),
            "Max_RAM%": sys_stats.get("max_ram"),
            "Avg_RAM%": sys_stats.get("avg_ram"),
            "CPU_Samples": sys_stats.get("cpu_samples"),
        })
        ctrl_df.at[idx, "Status"] = status

    ctrl_df.to_excel(CONTROL_FILE, index=False)
    print(f"[INFO] Control file updated: {CONTROL_FILE}")
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"[INFO] Summary CSV written: {SUMMARY_CSV}")

    port = 8501
    if os.path.exists(STREAMLIT_APP):
        if not is_port_in_use(port):
            print(f"[INFO] Starting Streamlit: {STREAMLIT_APP}")
            proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", STREAMLIT_APP],
                cwd=os.getcwd()
            )
            time.sleep(3)
            print(f"[INFO] Streamlit server launched.")
            print(f"Dashboard is running at {STREAMLIT_URL}. Press Ctrl+C to stop.")
        else:
            print(f"[INFO] Streamlit is already running on port {port}. Opening in browser...")
            webbrowser.open(STREAMLIT_URL)
    else:
        print(f"[WARN] Streamlit app not found at {STREAMLIT_APP}")

if __name__ == "__main__":
    main()
