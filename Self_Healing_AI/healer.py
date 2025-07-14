# ***************************************************************************************************
# File        : healer.py
# Description : Standalone healing logic using LLM for AI self-healing selectors, with robust fallback.
# Author      : Aniket Pathare | Self-Healing AI Framework (2025)
# ***************************************************************************************************

from gpt4all import GPT4All
from selenium.webdriver.common.by import By
from datetime import datetime
import os
import re
from bs4 import BeautifulSoup
import difflib

#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\gemma-7b-it-Q5_K_M.gguf"
#MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf"
MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\Phi-3-mini-4k-instruct.Q4_0.gguf"


def log_healing_event(test_case_id, element_key, selector, healing_time=None):
    log_dir = "Logs"
    log_path = os.path.join(log_dir, "healing.log")
    os.makedirs(log_dir, exist_ok=True)
    log_line = f"[{datetime.now()}][TC:{test_case_id}][Key:{element_key}] Healed using selector: {selector}"
    if healing_time is not None:
        log_line += f"[HealingTime:{healing_time}]"
    log_line += "\n"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(log_line)
    print(log_line.strip())


def log_dom_snippet_for_healing(key, dom_snippet):
    log_dir = "Logs"
    log_path = os.path.join(log_dir, "healing_dom_snippets.log")
    os.makedirs(log_dir, exist_ok=True)
    log_line = (
        f"{'='*80}\n"
        f"[{datetime.now()}] Healing key: {key}\n"
        f"DOM Snippet:\n{dom_snippet}\n"
    )
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(log_line)

def improved_clean_selector(raw_response: str) -> str:
    response = raw_response.replace("`", "").replace("Selector:", "").strip()
    response = response.split(" or ")[0].strip()
    response = re.sub(r'\(.*?\)', '', response).strip()
    lines = response.splitlines()
    if lines:
        response = lines[0].strip()
    else:
        response = ""
    if (response.startswith('"') and response.endswith('"')) or (response.startswith("'") and response.endswith("'")):
        response = response[1:-1]
    if any(tag in response.lower() for tag in ["meta", "head", "script"]):
        return ""
    if response.startswith("//") or response.startswith(".") or response.startswith("#") or response.startswith("input") or response.startswith("button") or response.startswith("["):
        return response
    return response

def get_best_attribute_selector(html_snippet, element_key):
    # CSS: Try data-qa-id and id attributes (pick best fuzzy match)
    matches = re.findall(r'data-qa-id\s*=\s*["\']([^"\']+)["\']', html_snippet)
    best = ""
    best_score = 0
    key_norm = element_key.replace("_", "").replace("-", "").replace(" ", "").lower()
    for attr_val in matches:
        attr_norm = attr_val.replace("_", "").replace("-", "").replace(" ", "").lower()
        score = 0
        if key_norm in attr_norm or attr_norm in key_norm:
            score = min(len(key_norm), len(attr_norm))
        else:
            score = sum(1 for c in key_norm if c in attr_norm)
        if score > best_score:
            best_score = score
            best = attr_val
    if best:
        return f'[data-qa-id="{best}"]'
    matches = re.findall(r'id\s*=\s*["\']([^"\']+)["\']', html_snippet)
    for attr_val in matches:
        attr_norm = attr_val.replace("_", "").replace("-", "").replace(" ", "").lower()
        score = 0
        if key_norm in attr_norm or attr_norm in key_norm:
            score = min(len(key_norm), len(attr_norm))
        else:
            score = sum(1 for c in key_norm if c in attr_norm)
        if score > best_score:
            best_score = score
            best = attr_val
    if best:
        return f'#{best}'
    return ""

def get_best_xpath_contains_selector(html_snippet, element_key):
    key_norm = element_key.replace("_", "").replace("-", "").replace(" ", "").lower()
    if "data-qa-id" in html_snippet:
        return f'//a[contains(translate(@data-qa-id, "ABCDEFGHIJKLMNOPQRSTUVWXYZ_ -", "abcdefghijklmnopqrstuvwxyz---"), "{key_norm}")]'
    # Fallback: contains on text (lowercase)
    return f'//a[contains(translate(text(), "ABCDEFGHIJKLMNOPQRSTUVWXYZ_ -", "abcdefghijklmnopqrstuvwxyz---"), "{key_norm}")]'

def fallback_best_match_selector(html_snippet, element_key):
    import difflib
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_snippet, "html.parser")
    key_norm = element_key.strip().lower().replace("_", "").replace("-", "")
    candidates = list(soup.find_all(['input', 'textarea', 'select', 'a', 'button']))
    best_score = 0
    best_elem = None

    # Fuzzy match on attributes or text
    for tag in candidates:
        if tag.name in ["input", "textarea", "select"]:
            attr_str = " ".join([tag.get(attr, "") for attr in ["id", "name", "type", "data-qa-id", "aria-label"]])
            score = difflib.SequenceMatcher(None, key_norm, attr_str.lower()).ratio()
        else:
            score = difflib.SequenceMatcher(None, key_norm, tag.get_text(strip=True).lower()).ratio()
        if score > best_score:
            best_score = score
            best_elem = tag

    if best_elem:
        # 1. Prefer id (always safe with XPath)
        if best_elem.has_attr("id"):
            return f"//{best_elem.name}[@id='{best_elem['id']}']"
        # 2. Next: data-qa-id
        if best_elem.has_attr("data-qa-id"):
            return f"//{best_elem.name}[@data-qa-id='{best_elem['data-qa-id']}']"
        # 3. Next: unique name
        if best_elem.has_attr("name"):
            name_val = best_elem['name']
            others = soup.find_all(best_elem.name, attrs={"name": name_val})
            if len(others) == 1:
                return f"//{best_elem.name}[@name='{name_val}']"
        # 4. Next: unique type
        if best_elem.has_attr("type"):
            type_val = best_elem['type']
            others = soup.find_all(best_elem.name, attrs={"type": type_val})
            if len(others) == 1:
                return f"//{best_elem.name}[@type='{type_val}']"
        # 5. For <a> and <button>: match descendant text (robust for nested spans, etc.)
        if best_elem.name in ["a", "button"]:
            visible_text = best_elem.get_text(strip=True)
            # Try descendant text (handles text in <span>, etc.)
            return f'//{best_elem.name}[.//text()[normalize-space(.)="{visible_text}"]]'
        # 6. Last resort: just tag
        return f"//{best_elem.name}"
    return ""

def heal_locator_with_gpt4all(model, html_snippet, element_key):
    # Improved prompt for multi-candidate context with strict real-attribute usage
    prompt = (
        f"You are a Selenium locator expert.\n"
        f"Given only the HTML snippet below and the logical key '{element_key}',\n"
        f"there may be multiple similar elements present (with the same class).\n"
        f"For input, textarea, or select elements:\n"
        f" - If there is an 'id', always use it for the XPath selector.\n"
        f" - If there is a 'name', use it if it is unique within the snippet.\n"
        f" - Use 'type' only if there is a single element with that type in the HTML snippet.\n"
        f"For buttons or links:\n"
        f" - When several elements are present, always select the one whose visible text best matches or contains the element key.\n"
        f"Your task is to output exactly one XPath selector that uniquely identifies the actionable element best matching the logical key—prefer elements whose visible text, id, data-qa-id, or name matches or is most similar to the element key.\n"
        f"Never use the element key as an attribute value or text in your selector unless it appears exactly in the HTML below.\n"
        f"Only use attribute values or text that are present verbatim in the HTML snippet below.\n"
        f"Do NOT select a child or nested element, only the actual <a>, <button>, <input>, <textarea>, or <select> matching the key.\n"
        f"Only output the XPath selector, with no explanation.\n"
        f"HTML:\n{html_snippet}\n\n"
        f"Selector:"
    )

    print(f"[AI-HEAL] Prompt sent for '{element_key}':\n{prompt}\n{'=' * 60}")
    response = model.generate(prompt, max_tokens=64, temp=0)
    print(f"[AI-HEAL] Raw LLM response for '{element_key}':\n{response.strip()}\n{'=' * 60}")

    cleaned_selector = improved_clean_selector(response)
    print(f"[AI-HEAL] Cleaned selector for '{element_key}': {cleaned_selector}")

    # Robust hallucination check for key variants and key-only attributes
    key_norm = element_key.replace("_", "").replace("-", "").replace(" ", "").lower()
    hallucinated = False
    if cleaned_selector:
        # Check for using the key as an attribute value, unless it actually exists in HTML
        if (
            f'data-qa-id="{key_norm}"' in cleaned_selector.replace("_", "").replace("-", "").replace(" ", "").lower()
            and key_norm not in html_snippet.replace("_", "").replace("-", "").replace(" ", "").lower()
        ):
            print(f"[AI-HEAL] Selector matches only normalized key as attribute; ignoring as hallucination.")
            cleaned_selector = ""
            hallucinated = True

    # --- Fallback for invalid or empty selectors or placeholders ---
    invalid_placeholders = {"[insert your answer here]", "selector:", "`", ""}
    if (
        not cleaned_selector
        or cleaned_selector.strip().lower() in invalid_placeholders
        or "insert your answer here" in cleaned_selector.lower()
        or cleaned_selector.strip().startswith("Selector")
    ):
        print(f"[AI-HEAL][FALLBACK] LLM returned invalid/placeholder. Using fallback logic.")
        cleaned_selector = get_best_attribute_selector(html_snippet, element_key)
        if not cleaned_selector:
            cleaned_selector = get_best_xpath_contains_selector(html_snippet, element_key)
        print(f"[AI-HEAL][FALLBACK] Final fallback selector: {cleaned_selector}")

    # --- POST-VALIDATION: If the selector doesn't match an actionable tag, fallback by visible text best-match ---
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_snippet, "html.parser")
    if cleaned_selector and cleaned_selector.startswith('['):
        found = soup.select_one(cleaned_selector)
        if not found or found.name not in {"a", "button", "input"}:
            print(f"[AI-HEAL][POST-VALIDATION] Selector does not match actionable element. Using fallback_best_match_selector.")
            cleaned_selector = fallback_best_match_selector(html_snippet, element_key)
            print(f"[AI-HEAL][POST-VALIDATION] Post-validation fallback selector: {cleaned_selector}")

    # --- FINAL: Ensure the selector actually matches something in the snippet ---
    valid = False
    try:
        if cleaned_selector and cleaned_selector.startswith("//"):
            import re
            match = re.match(r"//([a-zA-Z]+)\[@([a-zA-Z0-9\-]+)='([^']+)'\]", cleaned_selector)
            if match:
                tag, attr, value = match.groups()
                for el in soup.find_all(tag):
                    if el.get(attr) == value:
                        valid = True
                        break
        else:
            if cleaned_selector:
                found = soup.select_one(cleaned_selector)
                if found:
                    valid = True
    except Exception as e:
        valid = False

    if not valid:
        print(f"[AI-HEAL][FINAL VALIDATION] Selector '{cleaned_selector}' does not match any element in snippet. Forcing fallback.")
        cleaned_selector = fallback_best_match_selector(html_snippet, element_key)
        print(f"[AI-HEAL][FINAL VALIDATION] Fallback to: {cleaned_selector}")

    return cleaned_selector

def ai_only_attempt_heal(driver, key, original_locator_tuple, dom_snippet, test_case_id):
    import time
    log_dom_snippet_for_healing(key, dom_snippet)
    print(f"[AI-HEAL] HTML snippet sent for '{key}': (also written to healing_dom_snippets.log)\n{'=' * 70}")
    print(f"[AI-HEAL] Starting healing process for '{key}'...")

    start_time = time.time()
    with GPT4All(model_name=MODEL_PATH) as model:
        selector = heal_locator_with_gpt4all(model, dom_snippet, key)
    duration = round(time.time() - start_time, 3)

    print(f"[AI-HEAL] LLM suggestion for '{key}': {selector}")

    if not selector:
        print(f"[AI-HEAL] Healing failed for '{key}'; no selector to log.")
        log_healing_event(test_case_id, key, "FAILED", healing_time=duration)
        return None

    by_type = "XPATH" if selector.startswith("//") else "CSS_SELECTOR"
    print(f"[AI-HEAL] Determined strategy: {by_type}")

    log_healing_event(test_case_id, key, selector, healing_time=duration)
    return (by_type, selector)

