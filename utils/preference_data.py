import pandas as pd
import os
import json
from datasets import load_dataset
import shutil

model_name = "zephyr-7b-sft-full"
base_path = f"/blue/yonghui.wu/sgao1/haoyan"
dataset_real_file_name = "iter0.csv"
dataset_generated_file_name = "iter3.csv"
df_real = pd.read_csv(f"{base_path}/data/gpt-score-{model_name}/{dataset_real_file_name}")
df_generated = pd.read_csv(f"{base_path}/data/gpt-score-{model_name}/{dataset_generated_file_name}")

data = []
count_r = 0
count_g = 0
for i in range(len(df_real)):
    if df_generated['G_Score'][i] - df_real['R_Score'][i] >= 5:
        data.append({"real": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_generated['G_Answer'][i])}], "generated": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_real['R_Answer'][i])}]})
        count_g += 1
    else:
        data.append({"real": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_real['R_Answer'][i])}], "generated": [{"role": "user", "content": str(df_generated['Question'][i])}, {"role": "assistant", "content": str(df_generated['G_Answer'][i])}]})
        count_r += 1
        
print(count_g)
print(count_r)

iter_n = dataset_generated_file_name.replace(".csv","")
output_dir  = f"{base_path}/gpt-preference-data-5/{model_name}/{iter_n}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


with open(f'{output_dir}/train.json', 'w') as f:
    json.dump(data, f, indent=4)

src_path = f"{base_path}/data/SPIN-generated-{model_name}/{iter_n}/test.json"
dst_path = f"{output_dir}/test.json"
shutil.copy(src_path, dst_path)


dataset = load_dataset('json', data_files=f'{output_dir}/train.json',split='train')
print(dataset)


