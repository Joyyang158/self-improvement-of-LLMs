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
import torch.nn.functional as F
import random

import warnings
warnings.filterwarnings("ignore")


kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter1-ckpt')
parser.add_argument('--input_dir', type=str, default='generated/llama-7b/iter0/synthetic')
parser.add_argument('--output_dir', type=str, default='sft_loss_test/llama-7b/iter0')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--data_type', type=str, default='real')

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
generated_data = data[args.data_type]

### sample 100 - gpt
# random.seed(42)
# generated_data  = random.sample(generated_data, 100)
###

loss_fn  = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id, reduction = "none")

df = pd.DataFrame(columns=['Question', 'R_Answer', 'G_Answer', 'R_Score', 'G_Score'])
accelerator.wait_for_everyone()    
with accelerator.split_between_processes(generated_data) as generated:
    feature_informaion = []
    log_prob_ls = []
    for each in tqdm(generated):
        model.eval()
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
        log_probs = F.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs.gather(dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
        sum_log_prob = target_log_probs[:, tokenized_question['input_ids'].shape[-1] + 1:].sum()
        log_prob_ls.append(sum_log_prob.item())
results_gathered_log_prob = gather_object(log_prob_ls)



if accelerator.is_local_main_process:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    filename = f"{args.output_dir}/log_prob_{args.input_dir.split('/')[-2]}_{args.data_type}.npy"
    np.save(filename, results_gathered_log_prob)