import numpy as np

data_1 = np.load('logprob_sum/zephyr-7b-sft-full/iter0/log_prob_iter1_generated.npy')
print(np.mean(data_1))
data_2 = np.load('logprob_sum/zephyr-7b-sft-full/iter0/log_prob_iter1_real.npy')
print(np.mean(data_2))
