import os
import pandas as pd
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from difflib import ndiff
import Levenshtein

# ================= CONFIGURATION =================
st.set_page_config(page_title="BART vs T5 — Foundation vs Fine-Tuned", layout="wide")

# ==== DATASET PATH ====
TEST_CSV = "ai_selector_training_data_cleaned.csv"  # your cleaned dataset

# ---- MODEL PATHS (edit if your paths differ) ----
MODEL_PATHS = {
    # Foundation
    "BART (Foundation)": "facebook/bart-base",
    "T5 (Foundation)": "t5-base",
    # Fine-tuned (local)
    "BART (Fine-Tuned)": r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model",
    "T5 (Fine-Tuned)": r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model_t5",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_GEN_LEN = 128
MAX_INPUT_TOKENS = 512

# ================= HEADER =================
st.title("BART & T5: Foundation vs Fine‑Tuned — Deep Comparison Dashboard")

st.info(
    """
- **Foundation models** load fresh from Hugging Face Hub (no task knowledge).
- **Fine‑tuned models** load from your local paths (specialized for DOM→XPath).
- All models are evaluated on the **same input set** for a fair comparison.
    """
)

# ================= HELPERS =================
rouge = evaluate.load("rouge")

@st.cache_resource(show_spinner=False)
def load_seq2seq(model_id_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id_or_path)
    mdl = mdl.to(DEVICE)
    mdl.eval()
    return tok, mdl

@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    # Build inputs: input_text (+ optional [CLASS: ...])
    inputs = [
        f"{t} [CLASS: {c}]" if isinstance(c, str) and c.strip() else t
        for t, c in zip(df["input_text"], df.get("class", [""] * len(df)))
    ]
    targets = df["target_text_xpath"].tolist()
    return df, inputs, targets


def generate_predictions(tokenizer, model, inputs, max_length=MAX_GEN_LEN, device=DEVICE):
    preds, token_attn_scores = [], []
    total = len(inputs)
    progress = st.progress(0)
    for i, inp in enumerate(inputs):
        with torch.no_grad():
            tokens = tokenizer(
                [inp], return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_TOKENS
            ).to(device)
            out = model.generate(
                **tokens,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            preds.append(pred)
            token_attn_scores.append([])  # placeholder for parity with prior UI
        if (i + 1) % max(1, total // 100) == 0 or i + 1 == total:
            progress.progress((i + 1) / total)
    return preds, token_attn_scores


def compute_metrics(preds, refs):
    r = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    exact = sum([p.strip() == t.strip() for p, t in zip(preds, refs)]) / len(refs)
    return {
        "rouge1": r["rouge1"],
        "rouge2": r["rouge2"],
        "rougeL": r["rougeL"],
        "exact": exact,
    }


# ================= SIDEBAR CONTROLS =================
st.sidebar.header("Run Settings")
sample_n = st.sidebar.number_input("Evaluate first N rows (for speed)", min_value=10, max_value=5000, value=200, step=10)
max_len = st.sidebar.slider("Max generated tokens", min_value=32, max_value=256, value=MAX_GEN_LEN, step=16)
run_button = st.sidebar.button("Run Evaluation")

# ================= DATA LOAD =================
df, all_inputs, all_targets = load_dataset(TEST_CSV)
if sample_n < len(all_inputs):
    inputs = all_inputs[:sample_n]
    targets = all_targets[:sample_n]
else:
    inputs, targets = all_inputs, all_targets

st.caption(f"Device: **{DEVICE.upper()}** | Samples: **{len(inputs)}** | Dataset: `{TEST_CSV}`")

# ================= RUN EVALUATION =================
results = {}
per_sample_scores = {}  # model -> list of rougeL per sample
pred_tables = {}        # model -> DataFrame with input/target/pred

if run_button:
    with st.spinner("Loading models & generating predictions..."):
        for model_name, model_path in MODEL_PATHS.items():
            st.subheader(model_name)
            tok, mdl = load_seq2seq(model_path)
            preds, _ = generate_predictions(tok, mdl, inputs, max_length=max_len, device=DEVICE)
            m = compute_metrics(preds, targets)
            results[model_name] = m
            # Per-sample rougeL
            rl = [
                rouge.compute(predictions=[preds[i]], references=[targets[i]], use_stemmer=True)["rougeL"]
                for i in range(len(targets))
            ]
            per_sample_scores[model_name] = rl
            # Prediction table
            pred_tables[model_name] = pd.DataFrame({
                "input": inputs,
                "target": targets,
                "prediction": preds,
                "exact_match": [int(p.strip() == t.strip()) for p, t in zip(preds, targets)],
                "rougeL": rl,
            })
            st.success(
                f"ROUGE-L: **{m['rougeL']*100:.2f}%** · ROUGE-1: {m['rouge1']*100:.2f}% · ROUGE-2: {m['rouge2']*100:.2f}% · Exact: **{m['exact']*100:.2f}%**"
            )

    # ================= METRIC SUMMARY =================
    st.markdown("---")
    st.header("Model Performance Summary")
    summary_df = pd.DataFrame([
        {
            "Model": name,
            "ROUGE-L": met["rougeL"] * 100,
            "ROUGE-1": met["rouge1"] * 100,
            "ROUGE-2": met["rouge2"] * 100,
            "Exact Match": met["exact"] * 100,
        }
        for name, met in results.items()
    ]).sort_values("ROUGE-L", ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(9, 4))
        summary_df.set_index("Model")[
            ["ROUGE-L", "ROUGE-1", "ROUGE-2", "Exact Match"]
        ].plot.bar(ax=ax)
        ax.set_ylabel("Score (%)")
        ax.set_title("Model Performance Comparison")
        st.pyplot(fig)
    with c2:
        st.dataframe(summary_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ================= ROUGE-L DISTRIBUTION =================
    st.subheader("ROUGE‑L Score Distribution (per sample)")
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    palette = sns.color_palette("tab10", n_colors=len(per_sample_scores))
    for i, (name, scores) in enumerate(per_sample_scores.items()):
        sns.kdeplot(scores, label=name, ax=ax2, linewidth=2, clip=(0, 1), color=palette[i])
    ax2.set_xlabel("ROUGE‑L")
    ax2.set_xlim(0, 1)
    ax2.set_title("Per‑sample ROUGE‑L Distributions")
    ax2.legend()
    st.pyplot(fig2)

    # ================= SAMPLE‑WISE WIN MAP =================
    st.subheader("Per‑sample Winner Heatmap")
    # Determine which model wins per sample by rougeL (ties → first in order)
    model_order = list(MODEL_PATHS.keys())
    win_indices = []
    for i in range(len(targets)):
        best = -1
        best_model = 0
        for j, name in enumerate(model_order):
            score = per_sample_scores[name][i]
            if score > best:
                best = score
                best_model = j
        win_indices.append(best_model)
    heat = np.array(win_indices).reshape(1, -1)
    fig3, ax3 = plt.subplots(figsize=(16, 1.4))
    cmap = sns.color_palette("tab10", n_colors=len(model_order))
    sns.heatmap(heat, cmap=sns.color_palette(cmap, as_cmap=True), cbar=False, xticklabels=False, yticklabels=False, ax=ax3)
    # put tiny initials on top
    initials = [m.split(" ")[0][0] + ("F" if "Foundation" in m else "T") for m in model_order]
    for i, idx in enumerate(win_indices):
        ax3.text(i + 0.5, 0.5, initials[idx], ha='center', va='center', fontsize=6, color="black")
    ax3.set_title("Winner per test sample (B= BART, T= T5; F=Foundation, T=Fine‑tuned)")
    st.pyplot(fig3)

    # ================= ERROR CASE TABLES =================
    st.markdown("---")
    st.header("Top Error Cases (Hall of Shame)")
    tabs = st.tabs(list(MODEL_PATHS.keys()))
    for tab, (name, table) in zip(tabs, pred_tables.items()):
        with tab:
            wrong = table[table["exact_match"] == 0].copy()
            if wrong.empty:
                st.success("No errors for this model on the sampled set. 🎉")
            else:
                wrong = wrong.sort_values("rougeL").head(15)
                st.dataframe(wrong, use_container_width=True, hide_index=True)

    # ================= SAMPLE DIAGNOSTICS =================
    st.markdown("---")
    st.subheader("Inspect a specific sample")
    idx = st.number_input("Sample index", min_value=0, max_value=len(inputs)-1, value=0, step=1)
    colA, colB = st.columns(2)
    with colA:
        st.code(inputs[idx], language="text")
        st.write("**Target XPath**")
        st.code(targets[idx])
    with colB:
        for name in MODEL_PATHS.keys():
            st.write(f"**{name}** — ROUGE‑L: {per_sample_scores[name][idx]:.3f}")
            st.code(pred_tables[name].iloc[idx]["prediction"])
