from datasets import DatasetDict, load_dataset
import re
import random
from multiprocessing import cpu_count
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM
import torch
from trl import SFTTrainer
import config
from accelerate import Accelerator
import torch.optim as optim
import argparse


####################  Parameter Setting #################### 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter1-ckpt')
parser.add_argument('--input_dir', type=str, default='generated/llama-7b/iter0/synthetic')
parser.add_argument('--output_dir', type=str, default='sft_loss_test/llama-7b/iter0')
parser.add_argument('--data_type', type=str, default='real')
args = parser.parse_args()


####################  Set up #################### 
device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None

tokenizer = AutoTokenizer.from_pretrained(args.model, return_tensors='pt', padding = 'longest', trust_remote_code = True)

# set pad_token_id equal to the eos_token_id if not set
if tokenizer.pad_token_id is None:
  tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

accelerator = Accelerator()


#################### Prepare the data ####################

raw_dataset = load_dataset(args.input_dir)
# Set chat template
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

def apply_chat_template(example, tokenizer):
    messages = example[args.data_type]
    # if messages[0]["role"] != "system":
    #     messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

column_names = list(raw_dataset["train"].features)
raw_dataset = raw_dataset.map(apply_chat_template,
                                num_proc=cpu_count(),
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)





# train_dataset = load_from_disk(f"{config.dataset_output_dir}/train_data")

# eval_dataset =  load_from_disk(f"{config.dataset_output_dir}/test_data")
# train_dataset = train_dataset.select(range(10))
# eval_dataset = eval_dataset.select(range(10))




# tokenizer = AutoTokenizer.from_pretrained(config.model_name, return_tensors='pt', padding = 'max_length')

# # set pad_token_id equal to the eos_token_id if not set
# if tokenizer.pad_token_id is None:
#   tokenizer.pad_token_id = tokenizer.eos_token_id

# # Set reasonable default for models without max length
# if tokenizer.model_max_length > 100_000:
#   tokenizer.model_max_length = 2048

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=use_4bit,
#     bnb_4bit_quant_type=bnb_4bit_quant_type,
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=use_nested_quant,
# )

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    # quantization_config=bnb_config,
    # device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# model_kwargs = dict(
#     # attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
#     torch_dtype="auto",
#     use_cache=False, # set to False as we're going to use gradient checkpointing
#     device_map=device_map,
#     # quantization_config=quantization_config,
# )




# model = AutoModelForCausalLM.from_pretrained(
#     config.model_name,
#     **model_kwargs
# )

# optimizer = optim.RMSprop(model.parameters(), lr = 5.0e-07)

# based on config
training_args = TrainingArguments(
    bf16=config.bf16, # specify bf16=True instead when training on GPUs that support bf16
    do_eval=config.do_eval,
    evaluation_strategy=config.evaluation_strategy,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    gradient_checkpointing=config.gradient_checkpointing,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=config.learning_rate,
    log_level=config.log_level,
    logging_steps=config.logging_steps,
    # logging_strategy=config.logging_strategy,
    lr_scheduler_type=config.lr_scheduler_type,
    # max_steps= config.max_steps,
    num_train_epochs=config.num_train_epochs,
    output_dir=args.output_dir,
    overwrite_output_dir=config.overwrite_output_dir,
    per_device_eval_batch_size=config.per_device_eval_batch_size, # originally set to 8
    per_device_train_batch_size=config.per_device_train_batch_size, # originally set to 8
    # push_to_hub=True,
    # hub_model_id="zephyr-7b-sft-lora",
    # hub_strategy="every_save",
    # report_to="tensorboard",
    save_strategy=config.save_strategy,
    save_total_limit=config.save_total_limit,
    seed=42,
    report_to="tensorboard",
    save_only_model = config.save_only_model
)


trainer = SFTTrainer(
        model = model,
        # model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=raw_dataset['train'],
        eval_dataset=raw_dataset['test'],
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=tokenizer.model_max_length,
        # optimizers = (optimizer, None)
    )


accelerator.prepare(trainer)

trainer.train()

trainer.save_model(f"{args.output_dir}/final")

# metrics = train_result.metrics
# max_train_samples = training_args.max_train_samples if training_args.max_train_samples is not None else len(train_dataset)
# metrics["train_samples"] = min(max_train_samples, len(train_dataset))
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()
