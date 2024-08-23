import pandas as pd
import os
import json
from datasets import load_dataset


dataset_file_name = "gpt_score/Llama-2-7b-ultrachat200k/iter0.csv"
df_real = pd.read_csv('gpt_score/Llama-2-7b-ultrachat200k/iter0.csv')
df_generated = pd.read_csv(dataset_file_name)

iter_n = dataset_file_name.split('/')[-1].replace(".csv","") 

data = []
count_r = 0
count_g = 0
for i in range(len(df_real)):
    if df_real['R_Score'][i] - df_generated['G_Score'][i] <= -5:
        data.append({"real": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_real['R_Answer'][i])}], "generated": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_generated['G_Answer'][i])}]})
        count_r += 1
    else:
        data.append({"real": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_generated['G_Answer'][i])}], "generated": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_real['R_Answer'][i])}]})
        count_g += 1
print(count_r)
print(count_g)

input_dir = f"generated-gpt-preference/Llama-2-7b-ultrachat200k/{iter_n}_v2"
if not os.path.exists(f'{input_dir}/synthetic'):
    os.makedirs(f'{input_dir}/synthetic')


with open(f'{input_dir}/synthetic/synthetic_train.json', 'w') as f:
    json.dump(data, f, indent=4)
# with open(f'{input_dir}/synthetic/synthetic_test.json', 'w') as f:
#     json.dump(test_data, f, indent=4)

dataset = load_dataset('json', data_files=f'{input_dir}/synthetic/synthetic_train.json',split='train')
# dataset_test = load_dataset('json', data_files=f'{input_dir}/synthetic/synthetic_test.json',split='train')


