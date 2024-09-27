import json
import os
from datasets import load_dataset
import pyarrow.parquet as pq
import random
import shutil
random.seed(42)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_fracs', type=int, default=25)
parser.add_argument('--input_dir', type=str, default='generated/Llama-2-7b-ultrachat200k/iter4')
parser.add_argument('--output_dir', type=str, default='synthetic')

args = parser.parse_args()
num_fracs = args.num_fracs
input_dir = args.input_dir
output_dir = args.output_dir

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

data = []
for i in range(num_fracs):
    with open(f'{input_dir}/train-raw-generated/loser_{i}.jsonl', 'r') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        result['generated'][1]['content'] = result['generated'][1]['content'].lstrip()
        data.append(result)

print(len(data))
test_data = []

for i in range(num_fracs):
    with open(f'{input_dir}/test-raw-generated/loser_{i}_test.jsonl', 'r') as f:
        json_list = list(f)

    for json_str in json_list:
        result = json.loads(json_str)
        result['generated'][1]['content'] = result['generated'][1]['content'].lstrip()
        test_data.append(result)

print(len(test_data))

if not os.path.exists(f'{input_dir}/synthetic'):
    os.makedirs(f'{input_dir}/synthetic')

with open(f'{input_dir}/train.json', 'w') as f:
    json.dump(data, f, indent=4)
with open(f'{input_dir}/test.json', 'w') as f:
    json.dump(test_data, f, indent=4)

dataset = load_dataset('json', data_files=f'{input_dir}/train.json',split='train')
dataset_test = load_dataset('json', data_files=f'{input_dir}/test.json',split='train')



print(len(dataset))
print(len(dataset_test))


target_folder = f'{input_dir}/generated_data'
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

folder1 = f'{input_dir}/test-raw-generated'
folder2 = f'{input_dir}/train-raw-generated'

shutil.move(folder1, target_folder)
shutil.move(folder2, target_folder)

# pq.write_table(dataset.data.table, f'{output_dir}/train_prefs-00000-of-00001.parquet')
# pq.write_table(dataset_test.data.table, f'{output_dir}/test_prefs-00000-of-00001.parquet')