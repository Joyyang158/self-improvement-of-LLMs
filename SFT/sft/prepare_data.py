from datasets import DatasetDict, load_dataset
import re
import random
from multiprocessing import cpu_count
from transformers import AutoTokenizer
import torch
import config

raw_datasets = load_dataset(config.dataset_name)
raw_datasets = raw_datasets.shuffle(seed=42)
size = 20000
print(raw_datasets)
dataset_dict = {"train": raw_datasets["train_sft"].select(range(size)),
                "test": raw_datasets["test_sft"].select(range(size))}

raw_datasets = DatasetDict(dataset_dict)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id


tokenizer.model_max_length = 2048

# Set reasonable default for models without max length
# if tokenizer.model_max_length > 2048:
#   tokenizer.model_max_length = 1536

# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    # We add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)

    return example

column_names = list(raw_datasets["train"].features)
raw_datasets = raw_datasets.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)



def split_data(text, max_length = 4096):
    segments = []
    current_segment = ''

    for t in text.split('</s>\n<|user|>\n')[1:]:
        if len(tokenizer.encode(f"{current_segment}</s>\n<|user|>\n{t}"))  < max_length:
            current_segment = f"{current_segment}</s>\n<|user|>\n{t}"
        else:
            segments.append(f"<|system|>\n{current_segment}")
    if current_segment:
        segments.append(f"<|system|>\n{current_segment}")
    
    return segments
            
def map_function(batch):
    new_examples = {'text': []}
    for strings in batch['text']:
        sequences = split_data(strings, max_length = 1792)
        new_examples['text'].extend(sequences)
    return new_examples



raw_datasets = raw_datasets.map(map_function,
                                num_proc=cpu_count(),
                                batched = True,
                                remove_columns = ["text"],
                                desc="Applying data splitting")



train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]


train_dataset.save_to_disk(f"{config.dataset_output_dir}/train_data")
eval_dataset.save_to_disk (f"{config.dataset_output_dir}/test_data")








