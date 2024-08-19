# import matplotlib.pyplot as plt
import numpy as np

# data = np.load('generated/Llama-2-7b-chat-hf/iter0/information/res.npy')

# def filter_data(data):
#     threshold = np.percentile(data,10)

#     filtered_data = [x for x in data if x>]
#     Q1 = np.percentile(data,25)
#     Q3 = np.percentile(data,75)
#     IQR  = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

#     return filtered_data


# print(data)
# Q1 = np.percentile(data,25)
# Q3 = np.percentile(data,75)
# IQR  = Q3 - Q1

# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

# plt.hist(filtered_data, bins=10, edgecolor = 'black')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.savefig('information_fig_llama2-7b-new.png')



# fisher_iter0 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter2/fisher.npy')
# loss_iter0 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter2/loss.npy')

# # fisher_iter1 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter1/fisher.npy')
# # loss_iter1 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter1/loss.npy')

# # fisher_iter2 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter2/fisher.npy')
# # loss_iter2 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter2/loss.npy')

# # fisher_sft = np.load('fisher_loss/zephyr-7b-sft-full/sft/fisher.npy')
# # loss_sft = np.load('fisher_loss/zephyr-7b-sft-full/sft/loss.npy')

# # plt.scatter(fisher_iter0, loss_iter0, color = 'blue', alpha = 1, label = 'Iter0')
# # plt.scatter(fisher_iter1, loss_iter1, color = 'red', alpha = 0.3, label = 'Iter1')
# # plt.scatter(fisher_sft, loss_sft, color = 'orange', alpha = 0.3, label = 'SFT')
# # plt.scatter(fisher_iter2, loss_iter2, color = 'orange', alpha = 0.3, label = 'Iter2')

# plt.xlim(1,1e10)

# plt.scatter(fisher_iter0, loss_iter0)

# plt.ylabel('Loss')
# plt.xlabel('Fisher Information')

# plt.legend()


# plt.savefig('fisher_loss_figs/TinyLlama-1.1B-Chat-v1.0//fisher_loss_iter2.png')


fisher1 = np.load('fisher_loss/TinyLlama-1.1B-Chat-v1.0/iter0_candidate/fisher.npy')
# fisher = fisher1[:100]
# x = [item for item in list(range(len(fisher))) for _ in range(5)]
# print(len(x))
# y = [item for sublist in fisher for item in sublist]
# print(len(y))

# plt.scatter(x,y)
# plt.savefig('bbb.png')

max_value = [max(sublist) for sublist in fisher1]

# max_value = fisher1 

max_fisher = max(max_value)
min_fisher = min(max_value)
average_fisher = np.average(max_value)


print(f"Max: {max_fisher:.2e}")
print(f"Min: {min_fisher:.2e}")
print(f"Average: {average_fisher:.2e}")

