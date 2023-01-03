
# checkpoint = "Salesforce/codegen-350M-mono"
import torch
import torch.nn as nn

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from transformers import RobertaTokenizer, T5ForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeT5Summ(nn.Module):
    def __init__(self, gpu_id=0, pretrained_model="Salesforce/codegen-350M-mono"):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.gpu_id = gpu_id
        if pretrained_model == "Salesforce/codet5-base-multi-sum":
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.model.resize_token_embeddings(len(self.tokenizer))

        self.max_source_length = 512   # -1
        self.max_target_length = 256   # /40
        self.set_device(gpu_id)

    def set_device(self, gpu_id=0):
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_minibatch(self, code, desc):
        encoding = self.tokenizer(
            code,
            padding="longest",
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt"
        ).to(self.device)
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(
            desc,
            padding="longest",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        ).to(self.device)
        labels = target_encoding.input_ids
        print(len(labels), labels)
        # print(len(attention_mask), attention_mask)
        # labels.clone().detach()  # torch.tensorify the labels
        labels = torch.tensor(labels)
        labels[labels == self.tokenizer.pad_token_id] = -100
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        print("loss: ", loss)
        return loss

    def summarize(self, code):
        input_ids = self.tokenizer(code, truncation=True, max_length=self.max_source_length,
                                   return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_summarize(self, code):
        encoding = self.tokenizer(code, padding="longest", truncation=True, max_length=self.max_source_length,
                                  return_tensors="pt")
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        outputs = self.model.generate(input_ids, attention_mask=attention_mask)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def save(self, outpath):
        torch.save(self.model, outpath)


