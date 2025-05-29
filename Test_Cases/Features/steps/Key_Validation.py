from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from gpt4all import GPT4All
import time

MODEL_PATH = r"C:\Users\Autom\PycharmProjects\Automation AI\Models\mistral-7b-instruct-v0.2.Q8_0.gguf"

# Locators dict (as per your structure)
locators = {
    "Reject_Cookies": [
        { "by": "CSS_SELECTOR", "value": "#onetrust-reject-all-handler", "auto_healed": False },
        { "by": "XPATH",        "value": "//button[@id='onetrust-reject-all-handler']", "auto_healed": False }
    ],
    "Login_btn1": [
        {"by": "CSS_SELECTOR", "value": "a[data-qa-id='layout-header-user-logon']", "auto_healed": False},
        {"by": "XPATH", "value": "//a[@data-qa-id='layout-header-user-logon']", "auto_healed": False}
    ],
    "Login_btn2": [
        { "by": "CSS_SELECTOR", "value": "button[data-qa-id='oauth-logon-button']", "auto_healed": False },
        { "by": "XPATH",        "value": "//button[@data-qa-id='oauth-logon-button']", "auto_healed": False }
    ],
    "email_field": [
        { "by": "ID",           "value": "zd-:r4:", "auto_healed": False },
        { "by": "XPATH",        "value": "//input[@id='zd-:r4:']", "auto_healed": False }
    ]
}

# Mapping your custom "by" to Selenium's By
by_mapping = {
    "XPATH": By.XPATH,
    "CSS_SELECTOR": By.CSS_SELECTOR,
    "ID": By.ID
}

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

def try_click(driver, steps, model=None):
    for step in steps:
        key = step["key"]
        locator_list = locators[key]
        print(f"\n[ACTION] Trying '{key}'...")

        found = False
        for locator in locator_list:
            try:
                by = by_mapping[locator["by"]]
                print(f"  Trying: ({locator['by']}, '{locator['value']}')")
                elem = WebDriverWait(driver, 6).until(
                    EC.element_to_be_clickable((by, locator["value"]))
                )
                elem.click()
                print(f"  [SUCCESS] Clicked '{key}' with ({locator['by']}, '{locator['value']}')")
                found = True
                break
            except Exception as e:
                print(f"  [FAILED] {key} with ({locator['by']}, '{locator['value']}'): {e}")

        # For email_field: Heal if not found!
        if not found and key == "email_field":
            print("  [HEALING] Attempting AI healing for 'email_field'...")
            html_snippet = driver.page_source[:1800]  # You can scope this further if desired
            healed_selector = heal_locator_with_gpt4all(model, html_snippet, "email_field")
            print(f"  [AI SUGGESTION]: {healed_selector}")
            if healed_selector:
                healed_by = By.XPATH if healed_selector.startswith("//") else By.CSS_SELECTOR
                try:
                    elem = WebDriverWait(driver, 8).until(
                        EC.element_to_be_clickable((healed_by, healed_selector))
                    )
                    elem.clear()
                    elem.send_keys("test@example.com")
                    print(f"  [HEALED] Typed in 'email_field' using ({healed_by}, '{healed_selector}')")
                    found = True
                except Exception as e:
                    print(f"  [FAILED] AI-healed selector did not work: {e}")
            else:
                print("  [FAILED] AI could not provide selector for 'email_field'")
        elif not found:
            print(f"  [FAILED] '{key}' could not be found and was not healed.")

def main():
    service = Service(r"C:\Users\Autom\PycharmProjects\Automation AI\chromedriver.exe")
    driver = webdriver.Chrome(service=service)
    driver.maximize_window()
    driver.get("https://www.zara.com/ie/en/logon")
    time.sleep(2)

    # Load GPT4All model once!
    with GPT4All(model_name=MODEL_PATH) as model:
        steps = [
            {"key": "Reject_Cookies"},
            {"key": "Login_btn1"},
            {"key": "Login_btn2"},
            {"key": "email_field"}
        ]
        try_click(driver, steps, model)
        time.sleep(3)
   # driver.quit()

if __name__ == "__main__":
    main()
