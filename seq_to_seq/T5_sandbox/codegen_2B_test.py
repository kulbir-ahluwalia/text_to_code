from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, os, argparse
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration

# tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
# model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')


tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2B-multi")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))




