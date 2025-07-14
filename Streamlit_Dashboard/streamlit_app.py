# streamlit_app.py
import streamlit as st
import pandas as pd
import os, glob, json, re, platform, psutil
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from wordcloud import WordCloud
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Selector Automation: End-to-End Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- FILE PATHS ---
CSV_PATH = os.path.join("Test_Report", "results_summary.csv")
HEAL_LOG = "healing.log"
HEAL_DOM_LOG = "healing_dom_snippets.log"
TRAINED_STATS_DIR = "Model_Training_Stats"
DATA_RAW_CSV = "ai_selector_training_data.csv"
DATA_CLEAN_CSV = "ai_selector_training_data_cleaned.csv"

# --- DATA LOADERS ---
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def parse_healing_log(path=HEAL_LOG):
    # [2024-06-27 23:56:17][TC:TC_Login_02][MODEL:BART][Key:login_username][Selector://input[@id='login_username']][HealingTime:1.234]
    pattern = (
        r'\[(?P<timestamp>[^\]]+)\]'
        r'\[TC:(?P<test_case_id>[^\]]+)\]'
        r'\[MODEL:(?P<model>[^\]]+)\]'
        r'\[Key:(?P<locator_key>[^\]]+)\]'
        r'(?:\[Selector:(?P<selector>[^\]]*)\])?'
        r'(?:\[HealingTime:(?P<healing_time>[\d\.]+)\])?'
    )
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(pattern, line)
                if m:
                    row = m.groupdict()
                    row['healing_time'] = float(row.get('healing_time') or 0)
                    row['timestamp'] = pd.to_datetime(row.get('timestamp'), errors="coerce")
                    rows.append(row)
    return pd.DataFrame(rows)

def parse_dom_log(path=HEAL_DOM_LOG):
    # [timestamp][TC:...][MODEL:...][Key:...][DOM:...]
    pattern = (
        r'\[(?P<timestamp>[^\]]+)\]'
        r'\[TC:(?P<test_case_id>[^\]]+)\]'
        r'\[MODEL:(?P<model>[^\]]+)\]'
        r'\[Key:(?P<locator_key>[^\]]+)\]'
        r'\[DOM:(?P<dom>.*?)\]$'
    )
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(pattern, line)
                if m:
                    row = m.groupdict()
                    row['timestamp'] = pd.to_datetime(row.get('timestamp'), errors="coerce")
                    rows.append(row)
    return pd.DataFrame(rows)

def get_system_stats():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)
    uname = platform.uname()
    return {
        "CPU": uname.processor or platform.processor(),
        "CPU Usage (%)": cpu,
        "RAM (GB)": round(mem.total/1e9,2),
        "RAM Usage (%)": mem.percent,
        "OS": uname.system + " " + uname.release,
        "Machine": uname.machine,
        "GPU": get_gpu_name()
    }

def get_gpu_name():
    # Try NVIDIA-SMI for NVIDIA, fallback to None
    try:
        import subprocess
        output = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True)
        gpu = output.decode().split("\n")[0].strip()
        return gpu or "N/A"
    except Exception:
        return "N/A"

def get_model_specs():
    # Publicly available info (examples, feel free to update)
    # Add or edit as needed
    return [
        {
            "Model": "BART",
            "Size": "406M",
            "Weights": "~1.5 GB",
            "Architecture": "Encoder-Decoder",
            "OpenSource": "Yes",
        },
        {
            "Model": "Llama",
            "Size": "7B / 13B / 70B",
            "Weights": "13B: ~24 GB",
            "Architecture": "Decoder-only Transformer",
            "OpenSource": "Yes",
        },
        {
            "Model": "Mistral",
            "Size": "7B / 8x22B",
            "Weights": "7B: ~13 GB",
            "Architecture": "Transformer",
            "OpenSource": "Yes",
        },
        {
            "Model": "Gemma",
            "Size": "2B / 7B",
            "Weights": "7B: ~13 GB",
            "Architecture": "Decoder-only Transformer",
            "OpenSource": "Yes",
        },
        {
            "Model": "Phi",
            "Size": "1.3B / 2.7B",
            "Weights": "2.7B: ~5.2 GB",
            "Architecture": "Decoder-only Transformer",
            "OpenSource": "Yes",
        },
    ]

# --- MAIN DATA LOAD ---
df = load_csv(CSV_PATH)
healing_df = parse_healing_log()
dom_df = parse_dom_log()
stats_df = pd.concat(
    [load_csv(f) for f in glob.glob(os.path.join(TRAINED_STATS_DIR, "*.csv"))], ignore_index=True
) if os.path.isdir(TRAINED_STATS_DIR) else pd.DataFrame()
raw_data = load_csv(DATA_RAW_CSV)
clean_data = load_csv(DATA_CLEAN_CSV)

# --- METRICS (Execution) ---
if not df.empty:
    df['StartTime'] = pd.to_datetime(df['StartTime'], errors="coerce")
    df['EndTime'] = pd.to_datetime(df['EndTime'], errors="coerce")
    df['Duration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds()
    last_run = df['StartTime'].max()
else:
    last_run = None

# ... in your TAB 1:
if isinstance(last_run, pd.Timestamp) and not pd.isnull(last_run):
    last_run_val = last_run.strftime("%Y-%m-%d %H:%M:%S")
else:
    last_run_val = "N/A"
st.metric("Last Test Run", last_run_val)


# --- SIDEBAR ---
st.sidebar.header("Dashboard Settings")
tab_options = [
    "Testcase Execution",
    "Model Training & Data",
    "Model Comparison",
    "Knowledge Graph",
    "Healing Stats",
]
tabs = st.tabs(tab_options)

# ------------------- TAB 1: Testcase Execution ------------------- #
with tabs[0]:
    st.title("Testcase Execution: Full Stats & Visuals")
    if df.empty:
        st.warning("No test execution data found!")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Runs", len(df))
        c2.metric("Total Passed", (df["Status"] == "PASS").sum())
        c3.metric("Total Failed", (df["Status"] == "FAIL").sum())
        c4.metric("Total Healed", int(df.get("Healed_Count", 0).sum()) if "Healed_Count" in df else 0)
        if pd.isna(last_run) or last_run is None:
            st.metric("Last Test Run", "N/A")
        else:
            st.metric("Last Test Run", str(last_run.strftime("%Y-%m-%d %H:%M:%S")))

        st.subheader("Execution Table (scrollable)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader("Test Outcomes Distribution")
        st.plotly_chart(px.pie(df, names="Status", title="PASS/FAIL/HEALED Distribution"), use_container_width=True)
        st.subheader("Execution Duration Distribution")
        st.plotly_chart(px.histogram(df, x="Duration", color="Status", nbins=20, title="Test Duration (s)"), use_container_width=True)
        st.subheader("Test Execution Timeline")
        st.plotly_chart(px.scatter(df, x="StartTime", y="TestCase_ID", color="Status", title="Test Timeline"), use_container_width=True)

# ------------------- TAB 2: Model Training & Data (CRISP-DM etc) ------------------- #
with tabs[1]:
    st.title("Model Training, Dataset & CRISP-DM")
    st.markdown("#### CRISP-DM Summary")
    st.info("""
    1. **Business Understanding:** Automate locator healing using AI to reduce manual maintenance in UI test automation.
    2. **Data Understanding:** Gathered locator/DOM data, mapped breakages and healing attempts.
    3. **Data Preparation:** Cleaned raw CSV, handled missing/invalid locators, engineered features.
    4. **Modeling:** Trained models (BART, Llama, etc) to predict/correct broken locators.
    5. **Evaluation:** Compared models on healing accuracy, time to heal, error types, leaderboard stats.
    6. **Deployment:** Integrated model into healing controller, tracked outcomes, retrained as needed.
    7. **Feedback Loop:** Dashboards and logs feed back into improving model/data quality.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Raw Dataset (first 20 rows)")
        st.dataframe(raw_data.head(20))
        st.metric("Raw Dataset Rows", len(raw_data))
    with col2:
        st.markdown("##### Cleaned Dataset (first 20 rows)")
        st.dataframe(clean_data.head(20))
        st.metric("Cleaned Dataset Rows", len(clean_data))

    if not stats_df.empty:
        st.subheader("Model Training Metrics (All Epochs)")
        for metric in [c for c in stats_df.columns if c not in ['Model', 'epoch']]:
            fig = px.line(stats_df, x="epoch", y=metric, color="Model", markers=True, title=f"{metric} over Epochs")
            st.plotly_chart(fig, use_container_width=True)

        # Show leaderboard: best ROUGE-L per model
        best_per_model = stats_df.copy()
        # Some models (GGUF) may have epoch as "N/A" or None, handle gracefully
        if best_per_model["epoch"].dtype != int:
            # For non-HF models, epoch may be string "N/A". For consistency, fill with 0.
            best_per_model["epoch"] = pd.to_numeric(best_per_model["epoch"], errors="coerce").fillna(0).astype(int)
        # Get best ROUGE-L per model
        best_per_model = best_per_model.sort_values(by="eval_rougeL", ascending=False).groupby("Model").head(1)

        st.subheader("Model Experiment Leaderboard (Best ROUGE-L)")
        cols = ["Model", "epoch"] + [c for c in stats_df.columns if "eval" in c]
        st.dataframe(best_per_model[cols].sort_values(by="eval_rougeL", ascending=False))

        # Optional: Bar chart for ROUGE-L and Exact Match (best per model)
        st.subheader("Bar Chart: ROUGE-L and Exact Match (Best per Model)")
        chart_df = best_per_model[["Model", "eval_rougeL", "eval_exact"]].sort_values(by="eval_rougeL", ascending=False)
        fig = px.bar(chart_df, x="Model", y=["eval_rougeL", "eval_exact"], barmode="group",
                     title="Best ROUGE-L and Exact Match per Model")
        st.plotly_chart(fig, use_container_width=True)

    # Visuals: word cloud, missing/null heatmap
    st.subheader("Frequent Locators (Word Cloud)")
    if "element_key" in clean_data:
        text = " ".join([str(x) for x in clean_data["element_key"].dropna()])
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    st.subheader("Missing Value Heatmap (Cleaned Data)")
    if not clean_data.empty:
        st.write(px.imshow(clean_data.isna(), aspect='auto', color_continuous_scale="blues"))

# ------------------- TAB 3: Model Comparison ------------------- #
with tabs[2]:
    st.title("Model Comparison & System Statistics")
    st.markdown("#### Model Table (static + live stats)")
    specs = pd.DataFrame(get_model_specs())
    # Get live model exec stats from healing logs
    exec_stats = []
    for model in specs["Model"]:
        subset = healing_df[healing_df["model"].str.lower() == model.lower()] if not healing_df.empty else pd.DataFrame()
        healed_subset = subset[subset["attempt_status"].str.lower() == "success"] if not subset.empty else pd.DataFrame()
        count = len(healed_subset)
        avg_time = healed_subset["healing_time"].mean() if not healed_subset.empty else None
        fastest = healed_subset["healing_time"].min() if not healed_subset.empty else None
        slowest = healed_subset["healing_time"].max() if not healed_subset.empty else None
        exec_stats.append({
            "Model": model,
            "Healed Items": count,
            "Avg Healing Time": f"{avg_time:.2f}" if avg_time else "N/A",
            "Fastest Heal": f"{fastest:.2f}" if fastest else "N/A",
            "Slowest Heal": f"{slowest:.2f}" if slowest else "N/A",
        })
    exec_stats = pd.DataFrame(exec_stats)
    sys_stats = pd.DataFrame([get_system_stats()])
    st.dataframe(specs.merge(exec_stats, on="Model", how="left"), use_container_width=True)
    st.markdown("#### System Stats (current machine)")
    st.dataframe(sys_stats)
    # Radar comparison
    if not exec_stats.empty:
        st.markdown("#### Healing Comparison (Radar Chart)")
        radar = exec_stats.set_index("Model")[["Healed Items"]].fillna(0)
        fig = px.line_polar(radar.reset_index(), r="Healed Items", theta="Model", line_close=True, title="Healed Items per Model")
        st.plotly_chart(fig, use_container_width=True)
    # Grouped bar
    if not exec_stats.empty:
        st.markdown("#### Avg Healing Time per Model (Bar Chart)")
        fig = px.bar(exec_stats, x="Model", y="Avg Healing Time", title="Avg Healing Time (s) per Model")
        st.plotly_chart(fig, use_container_width=True)

# ------------------- TAB 4: Knowledge Graph ------------------- #
with tabs[3]:
    st.title("Knowledge Graph: Model → Test → Locator")
    if healing_df.empty:
        st.info("No healing events found!")
    else:
        st.markdown("#### Interactive Graph")
        G = nx.MultiDiGraph()
        for _, row in healing_df.iterrows():
            G.add_node(row['model'], color='lightblue', title=f"Model: {row['model']}")
            G.add_node(row['test_case_id'], color='lightgreen', title=f"Test: {row['test_case_id']}")
            G.add_node(row['element_key'], color='orange', title=f"Locator: {row['element_key']}")
            G.add_edge(row['model'], row['test_case_id'], title="used in")
            G.add_edge(row['test_case_id'], row['element_key'], title="healed")
        nt = Network(height="450px", width="100%", bgcolor="#222", font_color="white")
        nt.from_nx(G)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        nt.save_graph(tmp_file.name)
        st.components.v1.html(open(tmp_file.name, 'r', encoding='utf-8').read(), height=500)

# ------------------- TAB 5: Healing Stats Deep Dive ------------------- #
with tabs[4]:
    st.title("Healing Stats: What, When, How")
    if healing_df.empty:
        st.warning("No healing logs found!")
    else:
        st.markdown("#### All Healing Events Table")
        st.dataframe(healing_df, use_container_width=True)
        st.markdown("#### Most Frequently Broken Element Keys")
        top_keys = healing_df["element_key"].value_counts().head(15)
        st.bar_chart(top_keys)
        st.markdown("#### Healing Time Distribution")
        st.plotly_chart(px.histogram(healing_df, x="healing_time", nbins=25, title="Healing Time Distribution (s)"), use_container_width=True)
        st.markdown("#### Healing Timeline (per test case)")
        st.plotly_chart(px.scatter(healing_df, x="timestamp", y="test_case_id", color="model", hover_data=["element_key", "healing_time"], title="Healing Events Over Time"), use_container_width=True)
        st.markdown("#### Leaderboard: Most Healed Testcase")
        heal_leaderboard = healing_df.groupby("test_case_id").size().sort_values(ascending=False)
        st.dataframe(heal_leaderboard.reset_index(name="Healing Events"))
        st.markdown("#### Word Cloud of Broken Locators")
        if not healing_df.empty:
            text = " ".join([str(x) for x in healing_df["element_key"].dropna()])
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        # Advanced: Join with dom_df for dom sent/model results if available
        if not dom_df.empty:
            st.markdown("#### Sample DOMs sent to models (first 10)")
            st.dataframe(dom_df.head(10))
        st.markdown("#### Healing per Iteration Table (if available)")
        st.dataframe(healing_df[["test_case_id", "element_key", "model", "selector", "healing_time"]].sort_values("test_case_id"))


# --- Footer ---
st.markdown("---")
st.markdown(
    f"<sub>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Full pipeline dashboard by Streamlit | For questions, see project docs.</sub>",
    unsafe_allow_html=True
)
