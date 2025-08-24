#  Self-Healing Automation AI Framework

This project is an **AI-powered, self-healing test automation framework** that intelligently repairs broken DOM locators (like XPaths, CSS Selectors) using trained transformer models. It is designed to assist testers and developers by reducing flaky tests, minimizing manual rework, and improving UI test resilience.

The solution uses a combination of:
-  DOM structure analysis + heuristic filtering
-  Transformer-based models (BART, T5, GPT2) trained on `input_text` and `target_xpath`
-  Streamlit-powered dashboards for live XPath inference and evaluation
-  Performance tracking with Rouge-L, Exact Match, Latency, Retry logs, and more

This repository is structured into core functional modules:
- **Self_Healing_AI/** – AI-driven healing logic using local models or GGUF inference
- **Streamlit_Dashboard/** – Interactive visualization and test input interface
- **Test_Cases/features/** – BDD-based UI automation tests for real-world websites
- **Common_Functions/** – Shared utilities for logging, healing, and model I/O
- **Comparison_BART.py, evaluate_all_models.py** – Model evaluation & visualization scripts
- **ZaraModelMaster.py** – Full training script supporting BART/T5/GPT2 with Optuna tuning

 The framework supports both *fine-tuned transformer models* and *local LLMs (GGUF via llama.cpp)* for healing. It was designed and implemented as part of an MSc dissertation to demonstrate real-world applications of generative AI in software testing.

---

###  Dependencies
selenium==4.20.0  
behave==1.2.6  
openpyxl==3.0.10  
transformers==4.38.2  
torch==2.2.1  
