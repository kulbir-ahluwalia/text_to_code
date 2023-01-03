from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments
from datasets import load_metric, Dataset
from huggingface_hub import notebook_login
import json
import pandas as pd
import numpy as np
import torch

# model_name = "Salesforce/codegen-2B-mono"
# checkpoint = "Salesforce/codegen-350M-mono"
checkpoint = 't5-small'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# HYPER-PARAMETERS
batch_size = 128
model_name = checkpoint.split("/")[-1]
training_args = Seq2SeqTrainingArguments(output_dir=f"{model_name}-fine-tuned",
                                         evaluation_strategy="epoch",
                                         learning_rate=2e-3,
                                         per_device_train_batch_size=batch_size,
                                         per_device_eval_batch_size=batch_size,
                                         weight_decay=0.01,
                                         save_total_limit=1,
                                         num_train_epochs=5,
                                         predict_with_generate=True,
                                         push_to_hub=False,
)

with open("list_of_input_output_sequences_only.json", "r") as jsonDataset:
    nl_commands = []
    sequence_of_function_calls = []
    dataset = json.load(jsonDataset)
    for datapoint in dataset:
        nl_commands.append(datapoint["input_sequence"])
        sequence_of_function_calls.append(str(datapoint["output_sequence"]))
# print(f"nl_commands: {nl_commands}")
# print(f"sequence_of_function_calls: {sequence_of_function_calls}")

# create Pandas DataFrame ==> create "Dataset" object
text_labels_df = pd.DataFrame({'nl_command': nl_commands, 'sequence_of_function_calls': sequence_of_function_calls})
dataset = Dataset.from_pandas(text_labels_df).train_test_split(test_size=0.2)
tokenized_datasets = dataset.map(lambda examples: tokenizer(examples[0]), batched=True)

# training_set = dataset["train"]
# for example in training_set:
#     print(f"example is: {example}")
#     print(example['nl_command'], ", ", type(example['nl_command']))
#     print(tokenizer(example['nl_command']))
#     inference = model.generate(**tokenizer(example['nl_command'], return_tensors="pt"))
#     print(tokenizer.decode(inference[0]))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#  transformer's Seq2SeqTrainer might work better, but is not necessary here
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

metric = load_metric("accuracy")


if __name__ == '__main__':
    # notebook_login()
    trainer.train()
    trainer.evaluate()

