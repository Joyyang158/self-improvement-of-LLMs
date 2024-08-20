from transformers import AutoTokenizer, DataCollator, pipeline, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm
import torch
from datasets import load_dataset
import sys
import bitsandbytes as bnb
import gc
import argparse
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np


class PromptDataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query = [example['query'] for example in batch]
        ground_truth = [example['real_response'] for example in batch]
        generated_input_ids = [self.tokenizer.encode(q, return_tensors = "pt").squeeze(0) for q in query]
        # real_input_ids = [self.tokenizer.encode(g, return_tensors = "pt").squeeze(0) for g in ground_truth]

        # tokenized_query = prepare_prompts(query, self.tokenizer)
        return {
            "query" : query,
            "generated_input_ids": generated_input_ids,
            "ground_truth": ground_truth,
            # "real_input_ids": real_input_ids
        }

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def apply_chat_template(example):

    prompt_messages = example["real"][0]['content']
    real_response = example["real"][1]['content']
    modified_prompt = f"""<|system|>\n<|user|>\n{prompt_messages}\n<|assistant|>\n"""
    example['query'] = modified_prompt
    example['real_response'] = real_response

    return example

def get_reward(model, tokenizer, questions, answers, device):
    inputs = tokenizer(questions, answers, return_tensors='pt', padding = True).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
    rewards = [torch.tensor(output.item()) for output in outputs]

    return rewards

def get_cosine_similarity(model, generated_answers, real_answers):
    cosine_similarity = []
    for g, r in zip(generated_answers, real_answers):
        embeddings = model.encode([g, r])
        vec_a, vec_b = np.array(embeddings[0]), np.array(embeddings[1])
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        consie_sim = dot_product / (norm_a * norm_b)
        cosine_similarity.append(torch.tensor(np.round(consie_sim,4)))
    
    return cosine_similarity
    

def main(config_path):
    config = load_config(config_path)

    ppo_config = PPOConfig(
        model_name = config['model_name'],
        query_dataset = config['dataset_name'],
        reward_model = config['reward_model_name'],
        learning_rate = config['learning_rate'],
        batch_size = config['batch_size'],
        # mini_batch_size = config['mini_batch_size'],
        ppo_epochs = config['ppo_epochs'],
        gradient_accumulation_steps = config['gradient_accumulation_steps'],
        log_with = config['log_with'],
        project_kwargs = {
            "logging_dir": config['logging_dir']
        },
        tracker_project_name = config['tracker_project_name'],
        remove_unused_columns = config['remove_unused_columns']
    )



    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ppo_config.model_name,
        revision = config['model_revision'],
        trust_remote_code = True,
        torch_dtype = torch.bfloat16,
        use_flash_attention_2 = True,
        use_cache = True,
        # load_in_8bit = True,
        # bnb_config = {"load_in_8bit":}
        )
    
    cosine_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token


    dataset = load_dataset(ppo_config.query_dataset, split = 'train')
    eval_dataset = load_dataset(ppo_config.query_dataset, split = 'test').select(range(20))
    # dataset = dataset.select(range(40))

    column_names = list(dataset.features)
    dataset = dataset.map(
        apply_chat_template,
        num_proc = config['preprocessing_num_workers'],
        remove_columns=column_names
        )
    
    

    data_collator = PromptDataCollator(tokenizer)
    optimizer = bnb.optim.Adam8bit(model.parameters(), min_8bit_size = 4096, lr = ppo_config.learning_rate)


    ppo_trainer = PPOTrainer(
        model = model,
        config = ppo_config,
        dataset = dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
        optimizer = optimizer
    )


    generation_kwargs = {
        "min_new_tokens": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "max_new_tokens": config['max_new_tokens'],
        "pad_token_id": tokenizer.eos_token_id,
    }


    device = ppo_trainer.accelerator.device
    reward_model = AutoModelForSequenceClassification.from_pretrained(ppo_config.reward_model, torch_dtype = torch.bfloat16).to(device)
    reward_tokenizer = AutoTokenizer.from_pretrained(ppo_config.reward_model)
    # reward_pipe = pipeline("text-classification", model= config.reward_model, device = device, torch_dtype = torch.bfloat16)




    # generated_count, real_count = 0, 0
    epochs = config['total_epochs']
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch['generated_input_ids']
        
            #### Get response from SFTModel
            # response_tensors = []
            # for query in query_tensors:
            #     response = ppo_trainer.generate(query, **generation_kwargs)
            #     response_tensors.append(response.squeeze())



            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            generated_response_tensors = [ tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(query_tensors, response_tensors)] 
            generated_answers = [tokenizer.decode(r.squeeze()).replace("</s>","").lstrip() for r in generated_response_tensors]
            real_answers= batch['ground_truth']
            # real_response_tensors = batch['real_input_ids']



            #### Compute reward score
            questions = [q.replace("</s>","").replace("<|user|>\n","").replace("\n\n<|assistant|>\n","").lstrip() for q in batch["query"]]
            # generated_answers = [r for r in batch['response']]
            device = ppo_trainer.accelerator.device
            generated_rewards = get_reward(reward_model, reward_tokenizer, questions, generated_answers, device)
            cosine_similarity = get_cosine_similarity(cosine_model, questions, generated_answers)

            # real_rewards = get_reward(reward_model, reward_tokenizer, questions, real_answers, device)
  
            chosen_rewards = [g + c for g, c in zip(generated_rewards, cosine_similarity)]
            # chosen_rewards = generated_rewards
            chosen_response_tensors = generated_response_tensors

             # chosen_response_tensors = []
            # for i in range(len(generated_rewards)):
            #     if real_rewards[i] >= generated_rewards[i]:
            #         chosen_rewards.append(torch.tensor(real_rewards[i]))
            #         chosen_response_tensors.append(real_response_tensors[i])
            #         real_count += 1
            #     else:
            #         chosen_rewards.append(torch.tensor(generated_rewards[i]))
            #         chosen_response_tensors.append(generated_response_tensors[i])
            #         generated_count += 1
            

            # print(f"real_count: {real_count}")
            # print(f"generated_count: {generated_count}")
            # print(f"real_rewards: {real_rewards}")
            # print(f"generated_rewards: {generated_rewards}")
            
            
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, chosen_response_tensors, chosen_rewards)

            ppo_trainer.log_stats(stats, batch, chosen_rewards)
        
        #### Save model
        if ppo_trainer.accelerator.is_main_process:
            state_dict = ppo_trainer.accelerator.get_state_dict(model)
            model.save_pretrained(f"{config['output_dir']}/epoch{epoch}", state_dict=state_dict, safe_serialization = False)
            tokenizer.save_pretrained(f"{config['output_dir']}/epoch{epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = "config.yaml")
    args = parser.parse_args()
    main(args.config)