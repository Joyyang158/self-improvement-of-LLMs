import pandas as pd
import numpy as np


df = pd.read_csv('gpt_compare_res/processed_100_preference_zephyr.csv')
generated_logprob = np.load('logprob/zephyr-7b-sft-full/vanilla/sample_log_prob_iter0_generated.npy')
real_logprob = np.load('logprob/zephyr-7b-sft-full/vanilla/sample_log_prob_iter0_real.npy')
df['Generated_logprob'] = generated_logprob
df['Real_logprob'] = real_logprob
df.to_csv('gpt_compare_res/processed_100_preference_zephyr_logprob.csv')