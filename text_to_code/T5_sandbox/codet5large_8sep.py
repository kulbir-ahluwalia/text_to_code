from transformers import AutoTokenizer, T5ForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-large")
text = "Generate python: sort a list of numbers"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=16)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))



