from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_dir = r"C:\Users\Autom\PycharmProjects\Automation AI\trained_xpath_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

input_text = 'LOG_IN | <a class="layout-header-action__link layout-header-action__link--type-text link" href="https://www.zara.com/ie/en/logon" data-qa-id="layout-header-user-logon" tabindex="0">LOG IN</a>'
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
