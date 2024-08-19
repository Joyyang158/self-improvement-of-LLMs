import numpy as np
import matplotlib.pyplot as plt

color1 = "#d62728"
color2 = "#1f77b4"
color3 = "#9467bd"

model_zephyr = 'zephyr-7b-sft-full' 
model_llama = 'Llama-2-7b-ultrachat200k'
labels = {'0' : 'vanilla',
'1' : 'iter0',
'2' : 'iter1-1step',
'3' : 'iter2-1step',
'4' : 'iter3-1step',
}

def plot(index, label, index_SPIN, label_SPIN, model):
    data_generated = np.load(f'logprob_avg_new/{model}/{label}/log_prob_iter{index}_generated.npy')
    data_real = np.load(f'logprob_avg_new/{model}/{label}/log_prob_iter{index}_real.npy')
    data_combined = np.load(f'logprob_avg_new/{model}/{label_SPIN}/log_prob_iter{index_SPIN}_generated.npy')
    print(min(data_combined))
    print(min(data_real))
    print(min(data_generated))
    # bins = np.linspace(min(min(data_generated), min(data_real), min(data_combined)), max(max(data_generated), max(data_real), max(data_combined)), 200)
    bins = np.linspace(-2,0,100)
    print(f'{label}_generated - Mean: {np.mean(data_generated)}, Var: {np.var(np.array(data_generated))}')
    print(f'{label}_real - Mean: {np.mean(data_real)}, Var: {np.var(np.array(data_real))}')
    if index_SPIN == '4':
        data_combined_real = np.load(f'logprob_avg_new/{model}/{label_SPIN}/log_prob_iter{index_SPIN}_real.npy')
        print(f'{label_SPIN}_generated - Mean: {np.mean(data_combined)}, Var: {np.var(np.array(data_combined))}')
        print(f'{label_SPIN}_real - Mean: {np.mean(data_combined_real)}, Var: {np.var(np.array(data_combined_real))}')
    plt.figure(figsize=(10,6))
    plt.hist(data_generated, bins = bins , alpha = 0.7, color =color1, label = 'generated')
    plt.hist(data_real, bins = bins , alpha = 0.7, color =color2, label = 'real')
    plt.hist(data_combined, bins = bins , alpha = 0.7, color =color3, label = 'SPIN')

    plt.xlim(-2,0)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(f'{model} - {label_SPIN}' ,fontsize = 20)
    plt.xlabel('logprob value', fontsize = 18)
    plt.ylabel('Frequency', fontsize = 18)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logprob_avg_new/{model}/{label_SPIN}.png', dpi = 500)



def plot_reward(index, label, model):
    data_generated = np.load(f'logprob_sum_self/{model}/{label}/log_prob_iter{index}_generated.npy')
    data_real = np.load(f'logprob_sum_self/{model}/{label}/log_prob_iter{index}_real.npy')
    data_generated_next = np.load(f'logprob_sum_pre/{model}/{label}/log_prob_iter{index}_generated.npy')
    data_real_next = np.load(f'logprob_sum_pre/{model}/{label}/log_prob_iter{index}_real.npy')
    generated_reward = data_generated_next - data_generated
    real_reward = data_real_next - data_real
    bins = np.linspace(-200, 200, 400)
    print(f'{label}_generated_reward - Mean: {np.mean(generated_reward)}, Var: {np.var(np.array(generated_reward))}')
    print(f'{label}_real_reward - Mean: {np.mean(real_reward)}, Var: {np.var(np.array(real_reward))}')
    plt.figure(figsize=(10,6))
    plt.hist(generated_reward, bins = bins , alpha = 0.7, color =color1, label = 'generated_reward')
    plt.hist(real_reward, bins = bins , alpha = 0.7, color =color2, label = 'real_reward')

    plt.xlim(-200,200)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title(f'Reward - {model} - {label}' ,fontsize = 20)
    plt.xlabel('logprob value', fontsize = 18)
    plt.ylabel('Frequency', fontsize = 18)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logprob_sum_pre/{model}/{label}_reward.png', dpi = 500)

# plot(list(labels.keys())[3], labels['3'], list(labels.keys())[4], labels['4'], model_llama)
plot_reward(list(labels.keys())[3], labels['3'], model_zephyr)









def reward_research(index, label, index_next, label_next, index_next_next,label_next_next,  model):
    data_generated = np.load(f'logprob_sum/{model}/{label}/log_prob_iter{index}_generated.npy')
    data_real = np.load(f'logprob_sum/{model}/{label}/log_prob_iter{index}_real.npy')
    data_generated_next = np.load(f'logprob_sum/{model}/{label_next}/log_prob_iter{index_next}_generated.npy')
    data_real_next = np.load(f'logprob_sum/{model}/{label_next}/log_prob_iter{index_next}_real.npy')
    data_generated_next_next = np.load(f'logprob_sum/{model}/{label_next_next}/log_prob_iter{index_next_next}_generated.npy')
    data_real_next_next = np.load(f'logprob_sum/{model}/{label_next_next}/log_prob_iter{index_next_next}_real.npy')
    # index_current = int(len(data_generated) * 0.1)
    generated_reward = data_generated_next - data_generated
    real_reward = data_real_next - data_real

    generated_reward_next = data_generated_next_next - data_generated_next
    real_reward_next = data_real_next_next - data_real_next

    index_front = int(len(generated_reward) * 0.2)
    index_behind = int(len(generated_reward) * 0.8)


    index_data_generated_behind = np.argsort(generated_reward)[index_behind:]
    index_data_real_behind = np.argsort(real_reward)[index_behind:]

    index_data_generated_next_front = np.argsort(generated_reward_next)[:index_front]
    index_data_real_next_front = np.argsort(real_reward_next)[:index_front]

    print(f'generated: {len(set(index_data_generated_behind).intersection(set(index_data_generated_next_front))) / 2000}' )
    print(f'real: {len(set(index_data_real_behind).intersection(set(index_data_real_next_front))) / 2000}')



# reward_research(list(labels.keys())[0], labels['0'], list(labels.keys())[1], labels['1'], list(labels.keys())[2], labels['2'], model_zephyr)