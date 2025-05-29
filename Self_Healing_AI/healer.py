# ***************************************************************************************************
# File        : healer.py
# Description : Standalone healing logic using LLM for AI self-healing selectors.
# Author      : Aniket Pathare | Self-Healing AI Framework (2025)
# ***************************************************************************************************

from gpt4all import GPT4All
from selenium.webdriver.common.by import By
from datetime import datetime
import os

MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf"

def log_healing_event(element_key, selector):
    log_dir = "Logs"
    log_path = os.path.join(log_dir, "healing.log")
    os.makedirs(log_dir, exist_ok=True)
    log_line = f"[{datetime.now()}] Healed '{element_key}' using selector: {selector}\n"
    with open(log_path, "a") as log:
        log.write(log_line)
    print(log_line.strip())

def heal_locator_with_gpt4all(model, html_snippet, element_key):
    prompt = (
        f"You are a Selenium locator expert.\n"
        f"Given only the HTML snippet below and the logical key '{element_key}',\n"
        f"output exactly one selector (CSS or XPath) that uniquely identifies the element—\n"
        f"no extra text, no explanation, just the selector.\n\n"
        f"HTML:\n{html_snippet}\n\n"
        f"Selector:"
    )
    response = model.generate(prompt, max_tokens=64, temp=0)
    return response.strip()

def ai_only_attempt_heal(driver, key, original_locator_tuple, dom_snippet):
    """
    Try to heal locator using GPT4All LLM. Returns tuple: (by_type, selector) if successful.
    """
    print(f"[AI-HEAL] Starting healing process for '{key}'...")
    with GPT4All(model_name=MODEL_PATH) as model:
        selector = heal_locator_with_gpt4all(model, dom_snippet, key)
    print(f"[AI-HEAL] LLM suggestion for '{key}': {selector}")

    if not selector:
        print(f"[AI-HEAL] No selector returned by LLM for '{key}'.")
        return None

    by_type = "XPATH" if selector.startswith("//") else "CSS_SELECTOR"
    print(f"[AI-HEAL] Determined strategy: {by_type}")

    # Log healing event
    log_healing_event(key, selector)
    return (by_type, selector)
