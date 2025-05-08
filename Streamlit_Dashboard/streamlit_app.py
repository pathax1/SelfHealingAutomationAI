import streamlit as st
import pandas as pd
import os, glob, json
from datetime import datetime

# ─── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Automation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 1) Load summary CSV ────────────────────────────────────────────
CSV_PATH = os.path.join("Test_Report", "results_summary.csv")
if not os.path.exists(CSV_PATH):
    st.error(f"Summary CSV not found at {CSV_PATH}. Run run_controller.py with export.")
    st.stop()

df = pd.read_csv(CSV_PATH)
df["StartTime"] = pd.to_datetime(df.get("StartTime"), errors="coerce")
df["EndTime"]   = pd.to_datetime(df.get("EndTime"), errors="coerce")
df["Duration"]  = (df["EndTime"] - df["StartTime"]).dt.total_seconds()

# ─── 2) Sidebar filters ─────────────────────────────────────────────
st.sidebar.header("Filters")
envs     = sorted(df["Environment"].dropna().unique())
statuses = sorted(df["Status"].dropna().unique())
date_min = df["StartTime"].min().date()
date_max = df["StartTime"].max().date()
dates    = st.sidebar.date_input("Date range", [date_min, date_max])
chosen_env    = st.sidebar.multiselect("Environment", envs, default=envs)
chosen_status = st.sidebar.multiselect("Status", statuses, default=statuses)

mask = (
    df["Environment"].isin(chosen_env) &
    df["Status"].isin(chosen_status) &
    (df["StartTime"].dt.date >= dates[0]) &
    (df["StartTime"].dt.date <= dates[1])
)
filtered = df[mask]

# ─── 3) KPI metrics ─────────────────────────────────────────────────
st.title("🔬 Automation Test Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tests", len(filtered))
c2.metric("Passed", (filtered["Status"]=="PASS").sum())
c3.metric("Failed", (filtered["Status"]=="FAIL").sum())
c4.metric("Avg Duration (s)", f"{filtered['Duration'].mean():.1f}")

# ─── 4) Tabs ─────────────────────────────────────────────────────────
tabs = st.tabs(["Details", "Charts", "Screenshots & Report", "Raw Logs & Steps"])

# Details Tab with hyperlinks in Status
with tabs[0]:
    st.markdown("### Detailed Results")
    # Build HTML table with clickable Status links
    display_df = filtered.copy()
    def make_status_link(row):
        pattern = os.path.join("Test_Report","docs", f"{row.TestCase_ID}_*.docx")
        docs = glob.glob(pattern)
        if docs:
            path = os.path.abspath(docs[-1])
            # file:// link to local Word doc
            return f'<a href="file:///{path}" target="_blank">{row.Status}</a>'
        return row.Status
    display_df["Status"] = display_df.apply(make_status_link, axis=1)
    # Render as HTML (escape=False so links render)
    st.markdown(
        display_df.to_html(
            index=False,
            escape=False,
            classes="table table-striped"
        ),
        unsafe_allow_html=True
    )
    # Download filtered CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered CSV", csv, "filtered_results.csv")

# Charts Tab including Pass/Fail distribution
with tabs[1]:
    st.markdown("### Execution Status Over Time")
    trend = (
        filtered.groupby(filtered["StartTime"].dt.date)["Status"]
        .value_counts().unstack(fill_value=0)
    )
    st.area_chart(trend)

    st.markdown("### Duration by Test Case")
    st.bar_chart(filtered.groupby("TestCase_ID")["Duration"].mean())

    st.markdown("### Pass/Fail Distribution")
    st.bar_chart(filtered["Status"].value_counts())

# Screenshots & Report Tab
with tabs[2]:
    st.markdown("### Screenshots & Test Report")
    sel_tc = st.selectbox("Select TestCase_ID", filtered["TestCase_ID"].unique())
    row = filtered[filtered["TestCase_ID"]==sel_tc].iloc[0]

    # 1) Downloadable Word report
    docs = glob.glob(os.path.join("Test_Report","docs", f"{sel_tc}_*.docx"))
    if docs:
        doc_path = docs[-1]
        with open(doc_path, "rb") as f:
            st.download_button(
                "Download Test Report (Word)",
                data=f,
                file_name=os.path.basename(doc_path),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    else:
        st.info("No Word report found for this test run.")

    # 2) Show screenshots
    st.markdown("#### Screenshots")
    shots = glob.glob(os.path.join("Test_Report","screenshots", f"{sel_tc}_*","*.png"))
    if shots:
        cols = st.columns(3)
        for i, p in enumerate(shots):
            cols[i%3].image(p, caption=os.path.basename(p), use_column_width=True)
    else:
        st.info("No screenshots available.")

# Raw Logs & Steps Tab
with tabs[3]:
    st.markdown("### Raw Allure JSON")
    jsons = glob.glob("Test_Report/allure-results/*.json")
    sel_json = st.selectbox("Choose JSON", jsons)
    if sel_json:
        raw = json.load(open(sel_json))
        st.json(raw)

        # Normalize to list of dicts
        items = raw if isinstance(raw, list) else [raw]

        st.markdown("### Step-Level Details")
        steps = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("steps", "children"):
                if key in item and isinstance(item[key], list):
                    for s in item[key]:
                        status = s.get("status") or s.get("stepStatus")
                        start  = s.get("start") or s.get("startMillis")
                        stop   = s.get("stop")  or s.get("stopMillis")
                        def to_time(ts):
                            try:
                                return datetime.fromtimestamp(ts/1000).strftime("%H:%M:%S")
                            except:
                                return ""
                        steps.append({
                            "Name":   s.get("name", ""),
                            "Status": status,
                            "Start":  to_time(start) if start else "",
                            "Stop":   to_time(stop)  if stop  else "",
                        })
        if steps:
            df_steps = pd.DataFrame(steps)
            st.dataframe(df_steps, use_container_width=True)
        else:
            st.info("No step details found in this JSON.")

# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<sub>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Streamlit Dashboard</sub>",
    unsafe_allow_html=True
)
