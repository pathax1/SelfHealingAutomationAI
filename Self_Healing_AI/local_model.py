# ***************************************************************************************************
# File        : local_model.py
# Description : Local LLM selector query for AI Self-Healing Automation Framework.
# Author      : Aniket Pathare | Self-Healing AI Framework (2025)
# ***************************************************************************************************

import os
import re
from datetime import datetime

from Self_Healing_AI.healer import get_best_attribute_selector

# Switch between GPT4All and llama_cpp as needed
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\gemma-7b-it-Q5_K_M.gguf"
#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf"
MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Phi-3-mini-4k-instruct.Q4_0.gguf"

# ---------- LLM Loader ----------
LLM = None
def load_model(n_ctx: int = 2048, force_reload: bool = False):
    global LLM
    if LLM is None or force_reload:
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        print(f"[LLM] Loading model: {model_path}")
        LLM = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            use_mlock=True,
            verbose=False,
            n_gpu_layers=-1  # CUDA: use all layers on GPU if available!
        )
        print("[LLM] Model loaded.")
    return LLM

def is_model_available():
    try:
        load_model()
        return True
    except Exception as e:
        print(f"[LLM] Model not available: {e}")
        return False

def log_healing_event(test_case_id,element_key, selector):
    log_dir = "Logs"
    log_path = os.path.join(log_dir, "healing.log")
    os.makedirs(log_dir, exist_ok=True)
    log_line = f"[{datetime.now()}][TC:{test_case_id}][Key:{element_key}] Healed using selector: {selector}\n"
    with open(log_path, "a") as log:
        log.write(log_line)
    print(log_line.strip())

# ---------- Improved Selector Cleaning ----------
def improved_clean_selector(raw_response: str) -> str:
    response = raw_response.replace("`", "").replace("Selector:", "").strip()
    # Remove after 'or'
    response = response.split(" or ")[0].strip()
    # Remove annotation like (CSS), (XPath)
    response = re.sub(r'\(.*?\)', '', response).strip()
    # Take only first line if model outputs multiple
    response = response.splitlines()[0].strip()
    # Remove quotes
    if (response.startswith('"') and response.endswith('"')) or (response.startswith("'") and response.endswith("'")):
        response = response[1:-1]
    # Ignore selectors for meta, head, script, etc.
    if any(tag in response.lower() for tag in ["meta", "head", "script"]):
        return ""
    # Accept only likely selector starters
    if response.startswith("//") or response.startswith(".") or response.startswith("#") or response.startswith("input") or response.startswith("button") or response.startswith("["):
        return response
    return response

def _build_prompt(html_snippet: str, locator_key: str) -> str:
    max_html_length = 1500
    if len(html_snippet) > max_html_length:
        html_snippet = html_snippet[:max_html_length] + "..."
    element_hints = ""
    key_lower = locator_key.lower()
    if any(word in key_lower for word in ["button", "btn", "submit", "click"]):
        element_hints = "This is likely a button element."
    elif any(word in key_lower for word in ["input", "field", "text", "email", "password"]):
        element_hints = "This is likely an input field."
    elif any(word in key_lower for word in ["link", "href", "nav"]):
        element_hints = "This is likely a link or navigation element."
    elif any(word in key_lower for word in ["error", "message", "alert"]):
        element_hints = "This is likely a message or alert element."
    prompt = f"""You are a Selenium locator expert. Given the HTML snippet and element key below, generate ONE valid CSS selector or XPath that uniquely identifies the target element.

Element Key: "{locator_key}"
{element_hints}

Rules:
1. Return ONLY the selector - no explanations, no quotes, no extra text
2. Prefer CSS selectors over XPath when possible
3. Use stable attributes (data-qa-id, id, class) over dynamic ones
4. Avoid position-based selectors like nth-child unless necessary
5. Only select elements that are actionable (button, a, input, or clickable divs).
6. Do not select meta tags, head tags, or any non-clickable elements.
7. Make sure the selector is specific enough to be unique

HTML:
{html_snippet}

Selector:"""
    return prompt

def ask_selector_via_llm(html_snippet: str, locator_key: str, max_retries: int = 2) -> str:
    print(f"\n[LLM DEBUG] HTML DOM snippet for '{locator_key}':\n{html_snippet}\n{'=' * 70}\n")
    print(f"[LLM-QUERY] Requesting selector for '{locator_key}'...")
    if not is_model_available():
        print("[LLM-ERROR] Model is not available")
        log_healing_event(locator_key, "FAILED (Model not available)")
        return ""
    llm = load_model()
    prompt = _build_prompt(html_snippet, locator_key)
    print(f"[LLM-PROMPT] Using prompt for '{locator_key}':\n{prompt}\n{'=' * 60}")
    for attempt in range(max_retries + 1):
        try:
            print(f"[LLM-QUERY] Attempt {attempt + 1}/{max_retries + 1}")
            temperature = 0.0 if attempt == 0 else 0.1
            result = llm(
                prompt,
                max_tokens=128,
                temperature=temperature,
                stop=["\n", "Explanation:", "Note:", "This selector"],
                repeat_penalty=1.1
            )
            raw_response = result["choices"][0]["text"].strip()
            print(f"[LLM-RESPONSE] Raw: '{raw_response}'")
            cleaned_selector = improved_clean_selector(raw_response)
            print(f"[LLM-RESPONSE] Cleaned: '{cleaned_selector}'")

            # Robust hallucination check for key variants
            key_variants = [
                locator_key.lower(),
                locator_key.lower().replace("_", "-"),
                locator_key.lower().replace("_", ""),
            ]
            for variant in key_variants:
                if variant and variant in cleaned_selector.lower():
                    print(f"[LLM-HEAL] Selector uses key variant '{variant}' as an attribute or id; ignoring as hallucination.")
                    cleaned_selector = ""
                    break

            # Fallback: Use best-match attribute from DOM if selector is empty
            if not cleaned_selector:
                alt_selector = get_best_attribute_selector(html_snippet, locator_key)
                if alt_selector:
                    print(f"[LLM-HEAL] Fallback to best-match attribute selector: {alt_selector}")
                    cleaned_selector = alt_selector

            if cleaned_selector and (cleaned_selector.startswith("//") or cleaned_selector.startswith(".") or cleaned_selector.startswith("#") or cleaned_selector.startswith("input") or cleaned_selector.startswith("button") or cleaned_selector.startswith("[")):
                print(f"[LLM-SUCCESS] Valid selector generated: '{cleaned_selector}'")
                log_healing_event(locator_key, cleaned_selector)
                return cleaned_selector
            else:
                print(f"[LLM-WARNING] Invalid selector format on attempt {attempt + 1}: '{cleaned_selector}'")
                if attempt < max_retries:
                    print("[LLM-RETRY] Retrying...")
        except Exception as e:
            print(f"[LLM-ERROR] Query failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                print("[LLM-RETRY] Retrying after error...")
            else:
                print("[LLM-FAILED] All retry attempts exhausted")
    print(f"[LLM-FAILED] Could not generate valid selector for '{locator_key}' after {max_retries + 1} attempts")
    # Log the failed attempt for full traceability
    log_healing_event(locator_key, "FAILED")
    return ""
