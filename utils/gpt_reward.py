import os
from openai import OpenAI
import pandas as pd
import json
import random
from tqdm import tqdm
# from together import Together
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_file_path', type=str, default='/blue/yonghui.wu/sgao1/haoyan/data/trainable-noise-zephyr-7b-sft-full/iter1/train.json')
args = parser.parse_args()


GPT_API_KEY = os.environ["OPENAI_API_KEY"]

client = OpenAI(api_key = GPT_API_KEY)
def gpt_inference(prompt, model):
    response = client.chat.completions.create(
        model = model,
        messages = [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': prompt}
        ]
    )
    output = response.choices[0].message.content
    return output

prompt = """ You are tasked with evaluating the quality of the given answer based on the provided question. Your task is to assign a score between 0 and 100, where 0 indicates very poor quality, and 100 indicates excellent quality. You should use a 1-point increment scale, meaning the score can be any whole number between 0 and 100 (e.g. 73,91,68) and avoiding scores that are always multiples of 5. Consider factors such as relevance, clarity, accuracy, and completeness. Provide only the score without any explanation.

Question: {question}

Answer: {answer}

Score:
"""

# model = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
model = 'gpt-4o-mini'
data_file_path = args.data_file_path
with open(data_file_path, 'r') as file:
    data = json.load(file)


# random.seed(42)
# sample_data = random.sample(data, 2000)

file_path = f'/blue/yonghui.wu/sgao1/haoyan/data/gpt-score-trainable-noise-zephyr-7b-sft-full'
if not os.path.exists(file_path):
    os.makedirs(file_path)

csv_file = f'{file_path}/{data_file_path.split("/")[-2]}.csv'
print(csv_file)

# if file_path.split("/")[1] == "iter0_synthetic":
#     header = ['Question', 'R_Answer', 'G_Answer', 'R_Score', 'G_Score']
# else:
#     header = ['Question', 'G_Answer', 'G_Score']
# df = pd.DataFrame(columns=['Question', 'R_Answer', 'G_Answer', 'R_Score', 'G_Score'])
# with open(csv_path, 'w', newline='') as file:
#     pd.DataFrame(columns=header).to_csv(file, index = False)

file_exists = os.path.isfile(csv_file)


save_batch_size = 500
if data_file_path.split("/")[-2] == "iter0":
    count = 0
    total_data = []
    for each_sample in tqdm(data):
        question = each_sample['real'][0]['content']
        sft_answer = each_sample['real'][1]['content']
        generated_answer = each_sample['generated'][1]['content']
        r_score = gpt_inference(prompt.format(question = question, answer = sft_answer), model)
        g_score = gpt_inference(prompt.format(question = question, answer = generated_answer), model)
        new_row = {"Question": question, "R_Answer":sft_answer, "G_Answer":generated_answer, "R_Score":r_score, "G_Score":g_score}
        total_data.append(new_row)
        count += 1
        if count % save_batch_size == 0:
            batch_data = pd.DataFrame(total_data)
            with open (csv_file, 'a' if file_exists else 'w', newline='') as f:
                batch_data.to_csv(f, header=not file_exists, index=False)
            total_data = []
            file_exists = True
            print("---------------Save Complete--------------")
else:
    count = 0
    total_data = []
    for each_sample in tqdm(data):
        question = each_sample['real'][0]['content']
        sft_answer = each_sample['real'][1]['content']
        generated_answer = each_sample['generated'][1]['content']
        g_score = gpt_inference(prompt.format(question = question, answer = generated_answer), model)
        new_row = {"Question": question, "G_Answer":generated_answer, "G_Score":g_score}
        total_data.append(new_row)
        count += 1
        if count % save_batch_size == 0:
            batch_data = pd.DataFrame(total_data)
            with open (csv_file, 'a' if file_exists else 'w', newline='') as f:
                batch_data.to_csv(f, header=not file_exists, index=False)
            total_data = []
            file_exists = True
            print("---------------Save Complete--------------")
