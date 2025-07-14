import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from Self_Healing_AI.healer import (
    log_healing_event, improved_clean_selector,
    get_best_attribute_selector, fallback_best_match_selector
)

MODEL_MAP = {
    "BART": r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model",
    "T5":   r"C:\Users\Autom\PycharmProjects\Automation AI\Models\trained_xpath_t5",
    "GPT2": r"C:\Users\Autom\PycharmProjects\Automation AI\Models\trained_xpath_gpt2",
}

LOG_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\healing.log"
DOM_LOG_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\healing_dom_snippets.log"
TRAIN_DATA_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\ai_selector_training_data_cleaned.csv"

# Load Sentence-BERT model once
BERT_MODEL_NAME = "all-MiniLM-L6-v2"
_bert_model = None
def get_bert_model():
    global _bert_model
    if _bert_model is None:
        _bert_model = SentenceTransformer(BERT_MODEL_NAME)
    return _bert_model

# Load and cache training data + embeddings (+classes)
_train_data = None
_train_embeddings = {}
_train_classes = {}  # element_key -> class
def load_training_data():
    global _train_data, _train_embeddings, _train_classes
    if _train_data is None:
        _train_data = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8")
        for key, group in _train_data.groupby("element_key"):
            doms = group["input_text"].tolist()
            xpaths = group["target_text_xpath"].tolist()
            embeddings = get_bert_model().encode(doms, convert_to_tensor=True)
            _train_embeddings[key] = (doms, xpaths, embeddings)
            # Only take the first class (assume same for all group rows)
            try:
                class_val = group["class"].iloc[0]
                _train_classes[key] = class_val if isinstance(class_val, str) else ""
            except Exception:
                _train_classes[key] = ""
    return _train_data, _train_embeddings, _train_classes

def ai_selector_attempt_heal(driver, logical_key, last_failed_locator, dom_context, test_case_id, model_name="BART"):
    _, train_embeddings, train_classes = load_training_data()
    bert_model = get_bert_model()

    # 1. Semantic similarity healing (BERT)
    if logical_key in train_embeddings:
        doms, xpaths, embeddings = train_embeddings[logical_key]
        runtime_emb = bert_model.encode([dom_context], convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(runtime_emb, embeddings)[0]
        best_idx = int(torch.argmax(cos_scores))
        best_score = float(cos_scores[best_idx])
        healed_value = xpaths[best_idx]
        threshold = 1.1  # adjust this threshold as needed

        start_time = time.time()
        duration = 0

        if best_score >= threshold:
            healed_by = "xpath"
            duration = round(time.time() - start_time, 3)
            # --- UNIFIED LOGGING ---
            log_healing_event(
                test_case_id=test_case_id,
                element_key=logical_key,
                model="BERT_SIM",
                selector=f"{healed_by}:{healed_value}",
                healing_time=duration,
                attempt_number=1,
                inference_time=duration,
                dom_before=dom_context,
                dom_after=None,
                healed_locator=healed_value,
                broken_locator=last_failed_locator[1],
                attempt_status="success"
            )
            dom_log_line = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [TC:{test_case_id}] [Key:{logical_key}]\n"
                f"DOM:\n{dom_context}\n\n"
            )
            with open(DOM_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(dom_log_line)
            print(f"[AI SELECTOR] BERT_SIM healed: {healed_by} = {healed_value} (Sim: {best_score:.2f})")
            return healed_by, healed_value

    # 2. Fallback to LLM (BART/T5/GPT2) with class context
    print("[AI SELECTOR] [FALLBACK] Trying LLM model healing...")
    model_dir = MODEL_MAP.get(model_name)
    if not model_dir or not os.path.exists(model_dir):
        print(f"[ERROR] [AI SELECTOR] Model not found: {model_name} at {model_dir}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    except Exception as e:
        print(f"[ERROR] [AI SELECTOR] Failed to load model '{model_name}': {e}")
        return None

    # --- NEW: Add class as [CLASS: ...] to input if available ---
    class_str = train_classes.get(logical_key, "")
    if isinstance(class_str, str) and class_str.strip():
        input_text = f"{logical_key} | {dom_context} [CLASS: {class_str.strip()}]"
    else:
        input_text = f"{logical_key} | {dom_context}"

    inputs = tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                forced_bos_token_id=0,
                forced_eos_token_id=2,
            )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] [AI SELECTOR] Error during model inference: {e}")
        return None

    duration = round(time.time() - start_time, 3)

    # -- Robust Postprocessing (NEW) --
    cleaned_selector = improved_clean_selector(prediction)
    print("Raw model output:", prediction)
    print("Cleaned selector:", cleaned_selector)
    if not cleaned_selector or cleaned_selector.lower() in ["", "selector:", "insert your answer here"]:
        # fallback to attribute-based or best-match selector
        cleaned_selector = get_best_attribute_selector(dom_context, logical_key)
        if not cleaned_selector:
            cleaned_selector = fallback_best_match_selector(dom_context, logical_key)
    healed_by = "xpath" if cleaned_selector.startswith("//") else "css"
    healed_value = cleaned_selector

    # --- UNIFIED LOGGING ---
    log_healing_event(
        test_case_id=test_case_id,
        element_key=logical_key,
        model=model_name,
        selector=f"{healed_by}:{healed_value}",
        healing_time=duration,
        attempt_number=1,
        inference_time=duration,
        dom_before=dom_context,
        dom_after=None,
        healed_locator=healed_value,
        broken_locator=last_failed_locator[1],
        attempt_status="success"
    )
    dom_log_line = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [TC:{test_case_id}] [Key:{logical_key}]\n"
        f"DOM:\n{dom_context}\n\n"
    )
    with open(DOM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(dom_log_line)
    print(f"[AI SELECTOR] {model_name} healed: {healed_by} = {healed_value} in {duration}s")
    return healed_by, healed_value

# Manual test block
if __name__ == "__main__":
    dummy_result = ai_selector_attempt_heal(
        driver=None,
        logical_key="Logon_Button",
        last_failed_locator=("xpath", "//button[@data-qa-id='oauth-logo-button']"),
        dom_context="<button class='zds-button oauth-logon-view__button zds-button--primary zds-button--small' role='button' data-qa-id='oauth-logon-button'>LOG IN</button>",
        test_case_id="TC_DUMMY_001",
        model_name="BART"
    )
    print("Test Result:", dummy_result)
