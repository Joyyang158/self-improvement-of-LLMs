from datasets import load_dataset
import argparse
import numpy as np
import sys
import json
import random
import shutil
import os


parser = argparse.ArgumentParser()
parser.add_argument('--fisher_information_input_dir', type=str, default='fisher_loss/zephyr-7b-sft-full/iter0_candidate')
parser.add_argument('--generated_data_input_dir', type=str, default='generated/zephyr-7b-sft-full/iter0_candidate')
parser.add_argument('--test_data_copy_dir', type=str, default='generated/zephyr-7b-sft-full/iter0/synthetic')


args = parser.parse_args()
fisher_information = np.load(f'{args.fisher_information_input_dir}/fisher.npy')
print(len(fisher_information))

with open (f'{args.generated_data_input_dir}/synthetic/synthetic_train.json', 'r') as f:
    data = json.load(f)

max_indices = [max(enumerate(sublist), key=lambda x: x[1])[0] for sublist in fisher_information]
for i, j in zip(max_indices,range(len(data))):
    data[j]['generated'][1]['content'] = data[j]['generated'][1]['content'][i]


if not os.path.exists(f'{args.generated_data_input_dir}/synthetic_filter'):
    os.makedirs(f'{args.generated_data_input_dir}/synthetic_filter')

with open(f'{args.generated_data_input_dir}/synthetic_filter/synthetic_train.json', 'w') as f:
    json.dump(data, f, indent=4)

source_file = f'{args.test_data_copy_dir}/synthetic_test.json'
destination_file = f'{args.generated_data_input_dir}/synthetic_filter/synthetic_test.json'
shutil.copy(source_file, destination_file)

print(len(data))





