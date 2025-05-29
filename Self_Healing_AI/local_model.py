# ***************************************************************************************************
# File        : local_model.py
# Description : Local LLM selector query for AI Self-Healing Automation Framework.
# Author      : Aniket Pathare | Self-Healing AI Framework (2025)
# ***************************************************************************************************

import os
import re
from datetime import datetime

# Switch between GPT4All and llama_cpp as needed
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf"

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

def log_healing_event(element_key, selector):
    log_dir = "Logs"
    log_path = os.path.join(log_dir, "healing.log")
    os.makedirs(log_dir, exist_ok=True)
    log_line = f"[{datetime.now()}] Healed '{element_key}' using selector: {selector}\n"
    with open(log_path, "a") as log:
        log.write(log_line)
    print(log_line.strip())

# ---------- Selector Cleaning ----------
def _clean_selector(raw_response: str) -> str:
    if not raw_response:
        return ""
    response = raw_response.strip()
    patterns_to_remove = [
        r"^(Selector:\s*)",
        r"^(Answer:\s*)",
        r"^(Result:\s*)",
        r"^(CSS:\s*)",
        r"^(XPath:\s*)",
        r"^(The selector is:\s*)",
    ]
    for pattern in patterns_to_remove:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE)
    response = response.splitlines()[0].strip()
    if (response.startswith('"') and response.endswith('"')) or (response.startswith("'") and response.endswith("'")):
        response = response[1:-1]
    return response.strip()

def _is_valid_selector_format(selector: str) -> bool:
    if not selector:
        return False
    if selector.startswith("//") or selector.startswith("(//") or selector.startswith("./"):
        return True
    css_patterns = [
        r"^[a-zA-Z][\w-]*$",
        r"^\.[\w-]+",
        r"^#[\w-]+",
        r"^\[[\w\-\s=\"':\[\]]+\]",
        r"^[\w\-\s\.#\[\]=\"':>+~,\(\)]+$",
    ]
    for pattern in css_patterns:
        if re.match(pattern, selector):
            return True
    return False

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
5. Make sure the selector is specific enough to be unique

HTML:
{html_snippet}

Selector:"""
    return prompt

def ask_selector_via_llm(html_snippet: str, locator_key: str, max_retries: int = 2) -> str:
    print(f"[LLM-QUERY] Requesting selector for '{locator_key}'...")
    if not is_model_available():
        print("[LLM-ERROR] Model is not available")
        return ""
    llm = load_model()
    prompt = _build_prompt(html_snippet, locator_key)
    print(f"[LLM-PROMPT] Using prompt for '{locator_key}'")
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
            cleaned_selector = _clean_selector(raw_response)
            print(f"[LLM-RESPONSE] Cleaned: '{cleaned_selector}'")
            if _is_valid_selector_format(cleaned_selector):
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
    return ""
