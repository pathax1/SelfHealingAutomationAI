from llama_cpp import Llama

llm = Llama(
    model_path="C:/Users/Autom/PycharmProjects/Automation AI/Models/mistral-7b-instruct-v0.2.Q8_0.gguf",
    n_ctx=512,
    n_gpu_layers=-1  # Use GPU for all layers
)

# Example inference
output = llm.create_completion(
    prompt="Q: What is the capital of Ireland? A:",
    max_tokens=32,
    stream=True,
)
for token in output:
    print(token["choices"][0]["text"], end="", flush=True)
