from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

input_text = 'Reject_Optional_Cookies | <div id="onetrust-button-group-parent" class="ot-sdk-three ot-sdk-columns has-reject-all-button"><div id="onetrust-button-group"><button id="onetrust-pc-btn-handler" aria-label="Cookies Settings, Opens the preference center dialog">Cookies Settings</button> <button id="onetrust-reject-all-handler">Reject optional cookies</button> <button id="onetrust-accept-btn-handler">Accept All Cookies</button></div></div>'
inputs = tokenizer([input_text], return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(
    **inputs,
    max_length=64,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Predicted:", prediction)
