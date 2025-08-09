# ***************************************************************************************************
# File        : evaluate_all_models.py
# Description : Multiprocessed model evaluation + Streamlit dashboard for comparing
#               Foundation BART, Fine-Tuned BART, Llama, Gemma, Mistral & Phi.
# Author      : Adapted for multi-model comparison dashboard
# Date        : 2025-08-06
# ***************************************************************************************************

import os
import sys
import time
import re
import gc
import platform
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import torch
import streamlit as st

from multiprocessing import Process, current_process
from sklearn.metrics import f1_score
from evaluate import load as load_metric

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_CC = SmoothingFunction().method1
except ImportError:
    import nltk
    nltk.download('punkt')
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _BLEU_CC = SmoothingFunction().method1

# ==== CONFIG ====
ORIGINAL_CSV = r"C:\Users\Autom\PycharmProjects\Automation AI\ai_selector_training_data_cleaned.csv"
OUTPUT_DIR = "Advanced_Model_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_MAP = {
    "Foundation_BART": "facebook/bart-base",
    "Trained_BART":    r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model",
    "Llama":           r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "Gemma":           r"C:\Users\Autom\PycharmProjects\Automation AI\Models\gemma-7b-it-Q5_K_M.gguf",
    "Mistral":         r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf",
    "Phi":             r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Phi-3-mini-4k-instruct.Q4_0.gguf",
}
BATCH_SIZE = 50

# ======================= METRIC FUNCTIONS =======================
rouge = load_metric("rouge")

def compute_rouge_l(pred, ref):
    try:
        return rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)["rougeL"]
    except:
        return 0.0

def char_level_f1(y_true, y_pred):
    y_true, y_pred = list(str(y_true)), list(str(y_pred))
    labels = list(set(y_true + y_pred))
    y_true_bin = [[c == l for l in labels] for c in y_true]
    y_pred_bin = [[c == l for l in labels] for c in y_pred]
    try:
        return f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    except:
        return 0.0

def exact_match(a, b):
    return int(str(a).strip() == str(b).strip())

def jaccard_similarity(a, b):
    tokens_a = set(re.findall(r'//?\w+|\[@?[\w\-]+[=\'"]?.*?[\'"]?\]', str(a)))
    tokens_b = set(re.findall(r'//?\w+|\[@?[\w\-]+[=\'"]?.*?[\'"]?\]', str(b)))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

def levenshtein(s1, s2):
    if s1 == s2: return 0
    if not s1: return len(s2)
    if not s2: return len(s1)
    v0 = list(range(len(s2)+1))
    v1 = [0]*(len(s2)+1)
    for i,ch1 in enumerate(s1):
        v1[0] = i+1
        for j,ch2 in enumerate(s2):
            cost = 0 if ch1==ch2 else 1
            v1[j+1] = min(v1[j]+1, v0[j+1]+1, v0[j]+cost)
        v0, v1 = v1, v0
    return v0[len(s2)]

def bleu_score(pred, ref):
    try:
        return sentence_bleu([str(ref)], str(pred), smoothing_function=_BLEU_CC)
    except:
        return 0.0

def partial_match(pred, ref):
    ps = str(pred); rs = str(ref)
    return int(ps in rs or rs in ps)

def log_resources(tag=""):
    ram = psutil.virtual_memory().percent
    try:
        gpu_mem = subprocess.check_output(
            ['nvidia-smi','--query-gpu=memory.used','--format=csv,nounits,noheader'],
            stderr=subprocess.DEVNULL
        ).decode().split('\n')[0]
    except:
        gpu_mem = "N/A"
    print(f"[RES] {tag} | RAM: {ram:.1f}% | GPU Mem: {gpu_mem} MB")

# ======================= MODEL LOADERS =======================
def generate_prediction_bart(model_dir, input_text, device="cpu", tokenizer=None, model=None):
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    toks = tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**toks, max_length=64, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

def generate_prediction_llama_cpp(model_path, input_text, llm=None):
    if llm is None:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)
    prompt = (
        "You are a Selenium locator expert. "
        "Given the HTML snippet and element key below, generate ONE valid XPath that uniquely identifies the target element.\n\n"
        f"HTML:\n{input_text}\n\nSelector:"
    )
    return llm(prompt, max_tokens=64, stop=["\n"])["choices"][0]["text"].strip()

# ======================= PROCESS (MULTIPROCESS) =======================
def process_model(model, model_path, df, out_csv):
    results = []
    print(f"\n[PROCESS] {model} on {current_process().name}")
    # choose generation function
    if model in ("Foundation_BART", "Trained_BART"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained(model_path)
        m  = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        gen = lambda inp: generate_prediction_bart(model_path, inp, device=device, tokenizer=tok, model=m)
    else:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)
        gen = lambda inp: generate_prediction_llama_cpp(model_path, inp, llm=llm)

    done = set()
    if os.path.exists(out_csv):
        done = set(pd.read_csv(out_csv)["element_key"].astype(str))
    for idx, row in df.iterrows():
        key = str(row["element_key"])
        if key in done: continue
        inp = row["input_text"]
        gt  = row["target_text_xpath"]
        t0 = time.time()
        try: pred = gen(inp)
        except Exception as e:
            print(f" [ERR] {model} failed: {e}")
            pred = ""
        lat = time.time()-t0

        # compute metrics
        rL = compute_rouge_l(pred, gt)
        em = exact_match(pred, gt)
        f1 = char_level_f1(gt, pred)
        tlos = jaccard_similarity(pred, gt)
        lev = levenshtein(pred, gt)
        blu = bleu_score(pred, gt)
        pm  = partial_match(pred, gt)

        retry, retry_lat = 0, None
        if not em:
            t1 = time.time()
            try: pred2 = gen(inp)
            except: pred2 = ""
            retry = exact_match(pred2, gt)
            retry_lat = time.time()-t1

        reason = ("EmptySelector" if not pred
                  else "InvalidXPath" if "//" not in pred
                  else "Wrong" if not em else "")

        results.append({
            "element_key": key,
            "input_len": len(inp),
            "Model": model,
            "Predicted XPath": pred,
            "Ground Truth": gt,
            "ROUGE-L": rL,
            "BLEU": blu,
            "F1": f1,
            "EM": em,
            "Levenshtein": lev,
            "PartialMatch": pm,
            "TLOS": tlos,
            "Latency(s)": lat,
            "RetrySuccess": retry,
            "RetryLatency(s)": retry_lat,
            "FailReason": reason,
            "class": row.get("class","")
        })

        if (len(results) % BATCH_SIZE == 0) or (idx == len(df)-1):
            pd.DataFrame(results)\
              .to_csv(out_csv, mode='a',
                      header=not os.path.exists(out_csv),
                      index=False)
            log_resources(f"{model} idx={idx+1}")
            results.clear()
            gc.collect()

    print(f"[DONE] {model}")

# ======================= EVALUATION MAIN =======================
def main():
    df = pd.read_csv(ORIGINAL_CSV)
    procs = []
    for m, path in MODEL_MAP.items():
        out = os.path.join(OUTPUT_DIR, f"{m}_results.csv")
        p = Process(target=process_model, args=(m, path, df, out))
        p.start(); procs.append(p)
    for p in procs: p.join()

    # consolidate
    dfs = []
    for m in MODEL_MAP:
        f = os.path.join(OUTPUT_DIR, f"{m}_results.csv")
        if os.path.exists(f): dfs.append(pd.read_csv(f))
    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv(os.path.join(OUTPUT_DIR, "model_truth_comparison.csv"), index=False)
    print("[ALL MODELS DONE]")

# ======================= STREAMLIT DASHBOARD =======================
def get_hardware_stats():
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.2)
    try:
        gpu = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
            stderr=subprocess.DEVNULL
        ).decode().split('\n')[0]
    except:
        gpu = "N/A"
    return {
        "OS": platform.system()+" "+platform.release(),
        "Machine": platform.machine(),
        "CPU Usage (%)": cpu,
        "RAM Usage (%)": mem.percent,
        "GPU": gpu
    }

def run_dashboard():
    st.set_page_config(page_title="AI XPath Model Comparison", layout="wide")
    consolidated = os.path.join(OUTPUT_DIR, "model_truth_comparison.csv")
    if not os.path.exists(consolidated):
        st.error("No consolidated results. Run evaluation first.")
        return
    df = pd.read_csv(consolidated)
    st.title("🕸 AI XPath Model Comparison Dashboard")
    st.markdown("Compare **Foundation BART**, **Fine-Tuned BART**, **Llama**, **Gemma**, **Mistral** & **Phi**")

    tabs = st.tabs([
        "Summary","Distribution","Heatmap",
        "Error Cases","Latency","Correlation","Hardware"
    ])

    metric_cols = ["ROUGE-L","BLEU","F1","EM","Levenshtein","PartialMatch","TLOS","Latency(s)","RetrySuccess"]

    # --- Summary ---
    with tabs[0]:
        st.subheader("Average Metrics by Model")
        agg = (
            df.groupby("Model")[metric_cols]
            .mean()
            .rename(columns={
                "EM": "Exact Match",
                "TLOS": "Jaccard(TLOS)",
                "RetrySuccess": "Retry Success"
            })
            .round(3)
            .reset_index()
        )
        st.dataframe(agg, use_container_width=True)

    # --- Distribution ---
    with tabs[1]:
        st.subheader("Metric Distribution")
        choice = st.selectbox("Select Metric", metric_cols, index=0)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(data=df, x="Model", y=choice, ax=ax)
        ax.set_title(f"{choice} by Model")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- Heatmap ---
    with tabs[2]:
        st.subheader("Jaccard Similarity (TLOS) vs Input Length")
        df["Len Bucket"] = pd.cut(df["input_len"],
            bins=[0,300,600,900,1200,np.inf],
            labels=["0-300","301-600","601-900","901-1200","1200+"])
        hm = df.pivot_table(
            index="Len Bucket", columns="Model", values="TLOS", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10,4))
        sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
        st.pyplot(fig)

    # --- Error Cases ---
    with tabs[3]:
        st.subheader("Top Error Cases (Lowest ROUGE-L, EM=0)")
        errs = df[df["EM"]==0].sort_values("ROUGE-L").head(15)
        st.dataframe(errs[[
            "element_key","Model","Predicted XPath","Ground Truth",
            "ROUGE-L","BLEU","F1","Levenshtein"
        ]], use_container_width=True)

    # --- Latency ---
    with tabs[4]:
        st.subheader("Latency Statistics")
        lat = (
            df.groupby("Model")["Latency(s)"]
            .agg(["mean", "min", "max", "std"])
            .rename(columns={
                "mean": "Mean(s)", "min": "Min(s)",
                "max": "Max(s)", "std": "Std(s)"
            })
            .round(3)  # ← round up‐front
            .reset_index()
        )
        st.dataframe(lat, use_container_width=True)

        st.markdown("#### Latency Distribution")
        fig, ax = plt.subplots(figsize=(8,3))
        sns.histplot(data=df, x="Latency(s)", hue="Model", bins=30, element="step", stat="density", ax=ax)
        st.pyplot(fig)

    # --- Correlation ---
    with tabs[5]:
        st.subheader("Metric Correlation Matrix")
        corr = df[metric_cols].corr()
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # --- Hardware ---
    with tabs[6]:
        st.subheader("Current Hardware Stats")
        hw = get_hardware_stats()
        st.metric("OS", hw["OS"])
        st.metric("Machine", hw["Machine"])
        st.metric("CPU Usage (%)", f"{hw['CPU Usage (%)']:.1f}")
        st.metric("RAM Usage (%)", f"{hw['RAM Usage (%)']:.1f}")
        st.metric("GPU", hw["GPU"])

        st.markdown("#### Inference Latency Boxplot (per Model)")
        fig, ax = plt.subplots(figsize=(8,3))
        sns.boxplot(data=df, x="Model", y="Latency(s)", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # download
    st.subheader("Download Full Results")
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="model_truth_comparison.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1].lower()=="dashboard":
        run_dashboard()
    else:
        main()
