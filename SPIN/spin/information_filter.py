from datasets import load_dataset
import argparse
import numpy as np
import sys
import json
import random
import shutil
import os

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gradient_input_dir', type=str, default='generated/zephyr-7b-sft-full/iter0/information')
parser.add_argument('--generated_input_dir', type=str, default='generated/zephyr-7b-sft-full/iter0/synthetic')

args = parser.parse_args()
gradient_data = np.load(f'{args.gradient_input_dir}/res.npy')
print(len(gradient_data))
with open (f'{args.generated_input_dir}/synthetic_train.json', 'r') as f:
    data = json.load(f)


threshold = np.percentile(gradient_data, 75)
top_25_indices = np.where(gradient_data >= threshold)[0]

filtered_data = [data[i] for i in top_25_indices]

########################### Combined data ###########################

# combined_generated_data = data + filtered_data
# random.shuffle(combined_generated_data)


# if not os.path.exists(f'{args.gradient_input_dir}/synthetic'):
#     os.makedirs(f'{args.gradient_input_dir}/synthetic')

# with open(f'{args.gradient_input_dir}/synthetic/synthetic_train.json', 'w') as f:
#     json.dump(combined_generated_data, f, indent=4)

# source_file = f'{args.generated_input_dir}/synthetic_test.json'
# destination_file = f'{args.gradient_input_dir}/synthetic/synthetic_test.json'
# shutil.copy(source_file, destination_file)

# print(len(combined_generated_data))


########################### Only filtered data ###########################

if not os.path.exists(f'{args.gradient_input_dir}/synthetic_filter'):
    os.makedirs(f'{args.gradient_input_dir}/synthetic_filter')

with open(f'{args.gradient_input_dir}/synthetic_filter/synthetic_train.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)

source_file = f'{args.generated_input_dir}/synthetic_test.json'
destination_file = f'{args.gradient_input_dir}/synthetic_filter/synthetic_test.json'
shutil.copy(source_file, destination_file)

print(len(filtered_data))





