export CUDA_VISIBLE_DEVICES="0,1,2,3"
# model="/group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter0-ckpt_iterative/checkpoint-2500"
model="/group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/gpt-preference_outputs/iter1-ckpt/"
# model="/group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter2-ckpt"
# model="/group-volume/haoyan/models/zephyr-7b-sft-full/"

accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks arc_easy,arc_challenge --device cuda --batch_size 8 --num_fewshot 25
accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks truthfulqa_mc1,truthfulqa_mc2 --device cuda --batch_size 8
accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks winogrande --device cuda --batch_size 8 --num_fewshot 5
accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks gsm8k --device cuda --batch_size 8 --num_fewshot 5
accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks hellaswag --device cuda --batch_size 8 --num_fewshot 10
accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks mmlu --device cuda --batch_size 8 --num_fewshot 5


# accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks triviaqa --device cuda --batch_size 8
# accelerate launch lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks openbookqa --device cuda --batch_size 8