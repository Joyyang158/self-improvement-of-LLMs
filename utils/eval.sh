export CUDA_VISIBLE_DEVICES="1,2,3,5,7"

# model="/blue/yonghui.wu/sgao1/haoyan/spin-results/zephyr-7b-sft-full/gpt-preference-10-outputs/iter0"
model="/blue/yonghui.wu/sgao1/haoyan/spin-results/Llama-3-8B-Ultrachat-200K/SPIN/iter2"


accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks arc_easy,arc_challenge --device cuda --batch_size 8 --num_fewshot 25
accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks truthfulqa_mc1,truthfulqa_mc2 --device cuda --batch_size 8
accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks winogrande --device cuda --batch_size 8 --num_fewshot 5
accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks gsm8k --device cuda --batch_size 8 --num_fewshot 5
accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks hellaswag --device cuda --batch_size 8 --num_fewshot 10
accelerate launch /home/sgao1/haoyan/lm-evaluation-harness/lm_eval --model hf --model_args pretrained=$model,dtype='bfloat16' --tasks mmlu --device cuda --batch_size 8 --num_fewshot 5