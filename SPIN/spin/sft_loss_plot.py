import numpy as np
import math
from datasets import load_dataset


# model_names = ["zephyr-7b-sft-full", "WizardLM-2-7B", "Llama-2-7b-chat-hf", "llama-7b"]


# # 2441 14729 nan, function of sft_loss_test
# def read_clean_data_sft(model_name, split):
#     vanilla = np.load(f'sft_loss/{model_name}/vanilla/loss_{split}.npy')
#     iter0 = np.load(f'sft_loss/{model_name}/iter0/loss_{split}.npy')
#     iter1 = np.load(f'sft_loss/{model_name}/iter1/loss_{split}.npy')
#     iter2 = np.load(f'sft_loss/{model_name}/iter2/loss_{split}.npy')
#     iter3 = np.load(f'sft_loss/{model_name}/iter3/loss_{split}.npy')
#     avg_vanilla = np.average([value for value in vanilla if not math.isnan(value)])
#     avg_0 = np.average([value for value in iter0 if not math.isnan(value)])
#     avg_1 = np.average([value for value in iter1 if not math.isnan(value)])
#     avg_2 = np.average([value for value in iter2 if not math.isnan(value)])
#     avg_3 = np.average([value for value in iter3 if not math.isnan(value)])

#     return [avg_vanilla, avg_0, avg_1, avg_2, avg_3]

# # function of sft_loss_generated
# def read_clean_data_generated(model_name, split):
#     vanilla = np.load(f'generated_loss/{model_name}/vanilla/loss_iter0_{split}.npy')
#     avg_vanilla = np.average([value for value in vanilla if not math.isnan(value)])
#     iter0_0 = np.load(f'generated_loss/{model_name}/iter0/loss_iter0_{split}.npy')
#     avg_iter0_0 = np.average([value for value in iter0_0 if not math.isnan(value)])

#     iter0_1 = np.load(f'generated_loss/{model_name}/iter0/loss_iter1_{split}.npy')
#     avg_iter0_1 = np.average([value for value in iter0_1 if not math.isnan(value)])
#     iter1_1 = np.load(f'generated_loss/{model_name}/iter1/loss_iter1_{split}.npy')
#     avg_iter1_1 = np.average([value for value in iter1_1 if not math.isnan(value)])

#     iter1_2 = np.load(f'generated_loss/{model_name}/iter1/loss_iter2_{split}.npy')
#     avg_iter1_2 = np.average([value for value in iter1_2 if not math.isnan(value)])
#     iter2_2 = np.load(f'generated_loss/{model_name}/iter2/loss_iter2_{split}.npy')
#     avg_iter2_2 = np.average([value for value in iter2_2 if not math.isnan(value)])

#     iter2_3 = np.load(f'generated_loss/{model_name}/iter2/loss_iter3_{split}.npy')
#     avg_iter2_3 = np.average([value for value in iter2_3 if not math.isnan(value)])
#     iter3_3 = np.load(f'generated_loss/{model_name}/iter3/loss_iter3_{split}.npy')
#     avg_iter3_3 = np.average([value for value in iter3_3 if not math.isnan(value)])

#     return [avg_iter0_0  -  avg_vanilla , avg_iter1_1 - avg_iter0_1, avg_iter2_2 - avg_iter1_2 , avg_iter3_3 - avg_iter2_3]


# plot_data = {}
# for model in model_names:
#     plot_data[model] = read_clean_data_sft(model, 'train')


# import matplotlib.pyplot as plt

# x_labels_sft = ['Vanilla', 'Iter0', 'Iter1', 'Iter2', 'Iter3']
# # x_label_generated = ['V-0', '0-1', '1-2', '2-3']
# plt.figure()

# for label, values in plot_data.items():
#     plt.plot(x_labels_sft, values, marker = 'o', label = label)

# plt.legend()

# # plt.savefig('sft_loss_test/real_train.png')

# plt.savefig('sft_loss/real_train.png')



vanilla = np.load('sft_loss/Llama-2-7b-chat-hf_ultrachat/vanilla/loss_iter0_test.npy')
avg_vanilla = np.average([value for value in vanilla if not math.isnan(value)])
print(avg_vanilla)