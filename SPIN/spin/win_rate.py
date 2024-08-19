import numpy as np
import random
import json
import pandas as pd 

def win_rate_two_list(g_ls,r_ls):
    count = 0
    for i in range(len(g_ls)):
        if g_ls[i] > r_ls[i]:
            count += 1
    return count / len(g_ls)

def save_win_index(g_ls,r_ls):
    count = 0




random.seed(42)
random_range = random.sample(range(20000), 2000)

model = "zephyr-7b-sft-full" #Llama-2-7b-ultrachat200k, zephyr-7b-sft-full
key = "iter1-1step"

path_dict = {
    "vanilla": "log_prob_iter0",
    "iter0": "log_prob_iter1",
    "iter1-1step": "log_prob_iter2",
    "iter2-1step": "log_prob_iter3"
}


data_generated = np.load(f'logprob_sum_self/{model}/{key}/{path_dict[key]}_generated.npy')
data_real = np.load(f'logprob_sum_self/{model}/{key}/{path_dict[key]}_real.npy')
data_generated_next = np.load(f'logprob_sum_pre/{model}/{key}/{path_dict[key]}_generated.npy')
data_real_next = np.load(f'logprob_sum_pre/{model}/{key}/{path_dict[key]}_real.npy')

gpt_data = pd.read_csv(f"gpt_score/{model}/zephyr_reward_iter2.csv")
gpt_data_r = pd.read_csv(f"gpt_score/{model}/zephyr_reward_iter0.csv")
gpt_real_data = gpt_data_r['R_Score']
gpt_generated_data = gpt_data['G_Score']
count_1, count_2,count_3,count_4 = 0,0,0,0
for i,j in enumerate(random_range):

# for i in range(-1):
# data_generated = np.load(f'logprob_sum_self/Llama-2-7b-ultrachat200k/iter1-1step/log_prob_iter2_generated.npy')
# data_real = np.load(f'logprob_sum_self/Llama-2-7b-ultrachat200k/iter1-1step/log_prob_iter2_real.npy')
# data_generated_next = np.load(f'logprob_sum_pre/Llama-2-7b-ultrachat200k/iter1-1step/log_prob_iter2_generated.npy')
# data_real_next = np.load(f'logprob_sum_pre/Llama-2-7b-ultrachat200k/iter1-1step/log_prob_iter2_real.npy')

    generated_reward = data_generated_next[j] - data_generated[j]
    real_reward = data_real_next[j] - data_real[j]
    if (generated_reward < real_reward and gpt_generated_data[i] < gpt_real_data[i]):
        count_1 += 1
    elif (generated_reward > real_reward and gpt_generated_data[i] < gpt_real_data[i]):
        count_2 += 1
    elif (generated_reward < real_reward and gpt_generated_data[i] > gpt_real_data[i]):
        count_3 += 1
    elif (generated_reward > real_reward and gpt_generated_data[i] > gpt_real_data[i]):
        count_4 += 1
    else:
        print('none')
print(count_1, count_2, count_3, count_4)

# print(f'Iter - {0}: {win_rate_two_list(generated_reward,real_reward)}')
