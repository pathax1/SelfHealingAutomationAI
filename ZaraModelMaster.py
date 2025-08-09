import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    BartForConditionalGeneration, BartTokenizerFast,
    GPT2LMHeadModel, GPT2TokenizerFast,
    T5ForConditionalGeneration, T5TokenizerFast,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, set_seed
)
import evaluate
import optuna
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# ============== CONFIG ==============
SEED = 42
set_seed(SEED)
DATA_PATH = "ai_selector_training_data.csv"
#OUTPUT_DIR = "./trained_xpath_model"
OUTPUT_DIR = "./trained_xpath_model_t5"
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_SIZE = 4
EPOCHS = 10

#MODEL_NAME = "facebook/bart-base"
MODEL_NAME = "t5-base"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 300

# ============== DATA LOADING, CLEANING, & EDA ==============
def is_valid_xpath(x):
    x = str(x).strip()
    if not x.startswith("//"):
        return False
    if not x or x.lower() == "nan":
        return False
    if any([
        x.startswith("<"), x.startswith("."), x.startswith("#"),
        "{" in x, "}" in x, ":" in x[:5]
    ]):
        return False
    if len(x) < 6:
        return False
    return True

def is_generic_class(class_str):
    blacklist = [
        "link", "media", "container", "row", "col", "wrapper", "header", "footer",
        "input", "button", "title", "label", "text", "active", "open", "selected", "item"
    ]
    class_names = class_str.strip().split()
    return all(cls in blacklist for cls in class_names if cls)

def load_and_analyze_data(path):
    df = pd.read_csv(path)

    # =========== CLASS NAME BLACKLIST CLEANING ===========
    before_clean = len(df)
    df = df[~df['class'].fillna("").apply(is_generic_class)]
    df = df[df['class'].str.strip().astype(bool)]
    print(f"[CLEAN] Removed generic/empty class rows: {before_clean - len(df)}")

    print(df.info())
    print(df.isnull().sum())
    print("Total duplicate rows:", df.duplicated().sum())

    # --- STRICT XPATH FILTERING ---
    mask_valid = df["target_text_xpath"].apply(is_valid_xpath)
    num_bad = (~mask_valid).sum()
    if num_bad:
        print(f"WARNING: {num_bad} rows removed due to invalid XPaths (not starting with //, empty, CSS, or malformed).")
        print(df.loc[~mask_valid, ["element_key", "target_text_xpath"]])
    df = df[mask_valid].reset_index(drop=True)

    # --- DEDUPLICATE: keep only first occurrence of (element_key, target_text_xpath)
    before = len(df)
    df = df.drop_duplicates(subset=["element_key", "target_text_xpath"], keep="first").reset_index(drop=True)
    after = len(df)
    print(f"Rows before deduplication: {before}, after: {after}")

    # Save the cleaned CSV for reference
    clean_path = os.path.splitext(path)[0] + "_cleaned.csv"
    df.to_csv(clean_path, index=False)
    print(f" Cleaned file saved as {clean_path}")

    # Plot input/target length
    df["input_len"] = df["input_text"].apply(len)
    df["target_len"] = df["target_text_xpath"].apply(len)
    plt.figure(figsize=(8, 3))
    plt.hist(df["input_len"], bins=20, alpha=0.7, label="input_text")
    plt.hist(df["target_len"], bins=20, alpha=0.7, label="target_text_xpath")
    plt.title("Token/Char Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_lengths.png"))
    plt.close()
    return df

# ============== SPLIT DATA ==============
def split_data(df):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_size = int(0.72 * len(df))
    val_size = int(0.13 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    print(f"Train/Val/Test Split: {train_df.shape} {val_df.shape} {test_df.shape}")
    return train_df, val_df, test_df

# ============== MODEL/ TOKENIZER SELECTION ==============
def get_model_tokenizer(name):
    if "bart" in name:
        model = BartForConditionalGeneration.from_pretrained(name)
        tokenizer = BartTokenizerFast.from_pretrained(name)
        preprocess_func = lambda ex: preprocess_bart_t5(ex, tokenizer)
    elif "t5" in name:
        model = T5ForConditionalGeneration.from_pretrained(name)
        tokenizer = T5TokenizerFast.from_pretrained(name)
        preprocess_func = lambda ex: preprocess_bart_t5(ex, tokenizer)
    elif "gpt2" in name:
        model = GPT2LMHeadModel.from_pretrained(name)
        tokenizer = GPT2TokenizerFast.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        preprocess_func = lambda ex: preprocess_gpt2(ex, tokenizer)
    else:
        raise ValueError("Model not supported")
    return model, tokenizer, preprocess_func

def preprocess_bart_t5(examples, tokenizer):
    input_strings = []
    for t, c in zip(examples["input_text"], examples["class"]):
        if isinstance(c, str) and c.strip():
            input_strings.append(f"{t} [CLASS: {c}]")
        else:
            input_strings.append(t)
    model_inputs = tokenizer(
        input_strings, max_length=MAX_INPUT_LEN,
        truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["target_text_xpath"], max_length=MAX_TARGET_LEN,
        truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_gpt2(examples, tokenizer):
    input_strings = []
    for t, c in zip(examples["input_text"], examples.get("class", [""] * len(examples["input_text"]))):
        if isinstance(c, str) and c.strip():
            input_strings.append(f"{t} [CLASS: {c}]")
        else:
            input_strings.append(t)
    model_inputs = tokenizer(
        [s + tokenizer.eos_token + l for s, l in zip(input_strings, examples["target_text_xpath"])],
        max_length=MAX_INPUT_LEN+MAX_TARGET_LEN, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

# ============== METRICS ==============
rouge = evaluate.load("rouge")
def compute_exact_match(preds, labels):
    return np.mean([p.strip() == l.strip() for p, l in zip(preds, labels)])

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(x.strip().split()) for x in decoded_preds]
    decoded_labels = ["\n".join(x.strip().split()) for x in decoded_labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    exact = compute_exact_match(decoded_preds, decoded_labels)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    result["exact"] = round(exact * 100, 2)
    return result

# ============== HYPERPARAMETER SEARCH ==============
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 5, 12),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-4, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.3),
    }

# ============== ERROR VISUALIZATION ==============
def visualize_errors(test_df, preds_col="predicted_xpath"):
    mismatched = test_df[test_df["target_text_xpath"].str.strip() != test_df[preds_col].str.strip()]
    plt.figure(figsize=(8, 3))
    plt.hist(mismatched["target_len"], bins=20, alpha=0.7, label="Target Len")
    plt.hist(mismatched[preds_col].apply(len), bins=20, alpha=0.7, label="Pred Len")
    plt.title("Misprediction Lengths")
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_lengths.png"))
    plt.close()
    print(f" Mispredictions saved: {len(mismatched)}")

# ============== GGUF/LLM INFERENCE-ONLY EVALUATION ==============
def evaluate_gguf_model(model_name, model_path, test_inputs, test_targets, max_tokens=64):
    from llama_cpp import Llama
    from evaluate import load
    import numpy as np
    import pandas as pd

    print(f"Loading GGUF model: {model_name} from {model_path}")
    llm = Llama(model_path=model_path, n_ctx=1024)

    predictions = []
    for prompt in test_inputs:
        try:
            out = llm(prompt, max_tokens=max_tokens, stop=["\n"])
            predicted = out['choices'][0]['text'].strip()
        except Exception as e:
            predicted = ""
        predictions.append(predicted)

    # Compute metrics
    rouge = load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=test_targets, use_stemmer=True)
    exact = np.mean([p.strip() == l.strip() for p, l in zip(predictions, test_targets)])

    row = {
        "Model": model_name,
        "epoch": "N/A",
        "eval_loss": None,
        "eval_rouge1": round(rouge_result["rouge1"] * 100, 4),
        "eval_rouge2": round(rouge_result["rouge2"] * 100, 4),
        "eval_rougeL": round(rouge_result["rougeL"] * 100, 4),
        "eval_exact": round(exact * 100, 2),
    }
    stats_df = pd.DataFrame([row])
    STATS_DIR = "Model_Training_Stats"
    os.makedirs(STATS_DIR, exist_ok=True)
    stats_csv = os.path.join(STATS_DIR, f"{model_name}_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f" {model_name} stats saved to {stats_csv}")


def calc_metrics(targets, predictions):
    # Convert to lists of characters (for character-level metrics)
    target_chars = [list(str(t)) for t in targets]
    pred_chars = [list(str(p)) for p in predictions]

    # Flatten lists for micro-averaged scores
    y_true = sum(target_chars, [])
    y_pred = sum(pred_chars, [])

    # Pad lists to the same length
    max_len = max(len(y_true), len(y_pred))
    y_true += [''] * (max_len - len(y_true))
    y_pred += [''] * (max_len - len(y_pred))

    # Map to 1/0 for match (strict, per character)
    matches = [yt == yp for yt, yp in zip(y_true, y_pred)]
    acc = sum(matches) / len(matches)

    # For precision/recall/f1, treat each unique character as a class
    all_labels = list(set(y_true + y_pred))
    y_true_bin = [[c == label for label in all_labels] for c in y_true]
    y_pred_bin = [[c == label for label in all_labels] for c in y_pred]

    precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
# ============== MAIN TRAINING BLOCK ==============
if __name__ == "__main__":
    # ---- Data ----
    df = load_and_analyze_data(DATA_PATH)
    train_df, val_df, test_df = split_data(df)
    raw_datasets = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    # ---- Model/tokenizer/preprocessing ----
    model, tokenizer, preprocess_func = get_model_tokenizer(MODEL_NAME)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # ---- Tokenize datasets ----
    tokenized_datasets = raw_datasets.map(
        preprocess_func,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # ---- Training Args ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=20,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=0 if os.name == 'nt' else 2,
        report_to="none",
        seed=SEED,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        predict_with_generate=True,
    )

    # ---- Trainer ----
    trainer = Seq2SeqTrainer(
        model_init=lambda: model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- Optuna HP search ----
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        n_trials=5
    )
    print("Best hyperparameters found:", best_run.hyperparameters)
    for k, v in best_run.hyperparameters.items():
        setattr(trainer.args, k, v)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(" Training complete and model saved to", OUTPUT_DIR)

    # ===========================
    # NEW: Save Per-Epoch Training Stats to CSV for Streamlit
    # ===========================
    stats_rows = []
    for i, log in enumerate(trainer.state.log_history):
        if 'eval_loss' in log:
            row = {
                "Model": MODEL_NAME,
                "epoch": log.get("epoch", i + 1),
                "eval_loss": log.get("eval_loss"),
                "eval_rouge1": log.get("eval_rouge1"),
                "eval_rouge2": log.get("eval_rouge2"),
                "eval_rougeL": log.get("eval_rougeL"),
                "eval_exact": log.get("eval_exact"),
            }
            stats_rows.append(row)

    STATS_DIR = "Model_Training_Stats"
    os.makedirs(STATS_DIR, exist_ok=True)
    stats_df = pd.DataFrame(stats_rows)
    csv_name = MODEL_NAME.replace("/", "_").replace("-", "_") + "_stats.csv"
    stats_df.to_csv(os.path.join(STATS_DIR, csv_name), index=False)
    print(f" Training stats saved to {os.path.join(STATS_DIR, csv_name)}")

    # ---- Final test set predictions ----
    test_results = trainer.predict(tokenized_datasets["test"])
    preds = tokenizer.batch_decode(test_results.predictions, skip_special_tokens=True)
    test_df["predicted_xpath"] = preds
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print("Test predictions saved.")

    # --- Compute and log accuracy, precision, recall, F1 ---
    metrics = calc_metrics(
        test_df["target_text_xpath"].tolist(),
        test_df["predicted_xpath"].tolist()
    )
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Optionally: Save to CSV for Streamlit
    metrics_row = {**metrics, "Model": MODEL_NAME, "epoch": "FINAL"}
    stats_path = os.path.join("Model_Training_Stats", MODEL_NAME.replace("/", "_").replace("-", "_") + "_stats.csv")
    stats_df = pd.read_csv(stats_path)
    stats_df = pd.concat([stats_df, pd.DataFrame([metrics_row])], ignore_index=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"Final metrics added to {stats_path}")

    # ---- Save and visualize errors ----
    test_df["target_len"] = test_df["target_text_xpath"].apply(len)
    visualize_errors(test_df, preds_col="predicted_xpath")

    mispreds = test_df[test_df["target_text_xpath"].str.strip() != test_df["predicted_xpath"].str.strip()]
    mispreds.to_csv(os.path.join(OUTPUT_DIR, "mispredictions.csv"), index=False)
    corrects = test_df[test_df["target_text_xpath"].str.strip() == test_df["predicted_xpath"].str.strip()]
    corrects.to_csv(os.path.join(OUTPUT_DIR, "correct_predictions.csv"), index=False)

    # ===========================
    # GGUF/LLM Inference-Only Model Evaluation (Llama, Gemma, Mistral, Phi)
    # ===========================
    test_inputs = []
    for t, c in zip(test_df["input_text"], test_df["class"]):
        if isinstance(c, str) and c.strip():
            test_inputs.append(f"{t} [CLASS: {c}]")
        else:
            test_inputs.append(t)
    test_targets = test_df["target_text_xpath"].tolist()
    GGUF_MODELS = {
        "Llama":   r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Meta-Llama-3-8B-Instruct.Q4_0.gguf",
        "Gemma":   r"C:\Users\Autom\PycharmProjects\Automation AI\Models\gemma-7b-it-Q5_K_M.gguf",
        "Mistral": r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf",
        "Phi":     r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Phi-3-mini-4k-instruct.Q4_0.gguf",
    }
    for mname, mpath in GGUF_MODELS.items():
        evaluate_gguf_model(mname, mpath, test_inputs, test_targets, max_tokens=64)
