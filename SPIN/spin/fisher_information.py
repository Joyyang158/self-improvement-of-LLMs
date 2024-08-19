from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
import argparse
from accelerate.utils import InitProcessGroupKwargs
import torch.optim as optim
import torch
from tqdm import tqdm
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import gather_object
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter1-ckpt')
parser.add_argument('--input_dir', type=str, default='generated/zephyr-7b-sft-full/iter2/synthetic')
parser.add_argument('--output_dir', type=str, default='fisher_loss/zephyr-7b-sft-full/iter2')
parser.add_argument('--split', type=str, default='train')

args = parser.parse_args()
model_path = args.model

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, truncation = True, max_length = 4096)   
tokenizer.pad_token = tokenizer.eos_token
data = load_dataset(args.input_dir, split=args.split)
generated_data = data['generated']

loss_fn  = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id, reduction = "none")



accelerator.wait_for_everyone()    
with accelerator.split_between_processes(generated_data) as generated:
    feature_informaion = []
    loss_ls = []
    for each in tqdm(generated):
        model.eval()
        model.zero_grad()
        
        question = each[0]['content']
        answer = each[1]['content']
        tokenized_question = tokenizer(question, return_tensors = 'pt').to("cuda")
        input_text = question + tokenizer.eos_token + answer
        tokenized_input = tokenizer(input_text, return_tensors='pt').to("cuda")

        output = model(**tokenized_input)
        logits = output.logits

        input_ids = tokenized_input['input_ids']
        labels = input_ids[..., 1:]
        shift_logits = logits[..., :-1, :]
        loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), labels.reshape(-1))
        loss = loss[tokenized_question['input_ids'].shape[-1] + 1:].sum()
        
        loss.backward()

        total_grad_square_sum = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_square_sum = torch.sum(param.grad ** 2).item()
                total_grad_square_sum += grad_square_sum
        feature_informaion.append(total_grad_square_sum)
        loss_ls.append(loss.item())
results_gathered_fisher = gather_object(feature_informaion)
results_gathered_loss = gather_object(loss_ls)



if accelerator.is_local_main_process:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    filename = f"{args.output_dir}/fisher.npy"
    np.save(filename, results_gathered_fisher)
    filename = f"{args.output_dir}/loss.npy"
    np.save(filename, results_gathered_loss)