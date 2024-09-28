import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Llama-2-7b-ultrachat200k')
parser.add_argument('--iteration', type=str, default='iter1')

args = parser.parse_args()

spin_file_path = f"/blue/yonghui.wu/sgao1/haoyan/data/gpt-score-{args.model_name}"
noise_file_path = f"/blue/yonghui.wu/sgao1/haoyan/data/gpt-score-trainable-noise-{args.model_name}"
spin_data = pd.read_csv(f"{spin_file_path}/{args.iteration}.csv")
noise_data = pd.read_csv(f"{noise_file_path}/{args.iteration}.csv")
win_count, tie_count, lose_count = 0, 0 ,0



for i,j in zip(noise_data['G_Score'], spin_data['G_Score']):
    if int(i) > int(j):
        win_count += 1
    elif int(i) == int(j):
        tie_count += 1
    else:
        lose_count += 1

print(f"Win Rate:{np.round(win_count / 20000,4)}")
print(f"Tie Rate:{np.round(tie_count / 20000,4)}")
print(f"Lose Rate:{np.round(lose_count / 20000,4)}")