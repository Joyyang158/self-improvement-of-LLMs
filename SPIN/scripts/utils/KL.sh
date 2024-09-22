export CUDA_VISIBLE_DEVICES=3,4,5

# source /home/user/miniconda3/etc/profile.d/conda.sh
# conda activate evalenv
accelerate launch --num_processes 3 --main_process_port=2950 SPIN/spin/KL.py
accelerate launch --num_processes 3 --main_process_port=2950 SPIN/spin/KL.py --base_model joyfine/Llama-2-7b-ultrachat200k-SPIN-iter0 --spin_model joyfine/Llama-2-7b-ultrachat200k-SPIN-iter1 --base_input_file iter1.csv --spin_input_file iter2.csv --output_file iter1.csv
accelerate launch --num_processes 3 --main_process_port=2950 SPIN/spin/KL.py --base_model joyfine/Llama-2-7b-ultrachat200k-SPIN-iter1 --spin_model joyfine/Llama-2-7b-ultrachat200k-SPIN-iter2 --base_input_file iter2.csv --spin_input_file iter3.csv --output_file iter2.csv

#### zephyr ####
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter0-ckpt --input_dir generated/zephyr-7b-sft-full/iter0/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/vanilla --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter0-ckpt --input_dir generated/zephyr-7b-sft-full/iter0/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/vanilla --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter1-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter1/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter0 --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter1-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter1/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter0 --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter2/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter1-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter2/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter1-1step --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter3-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter2-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter3-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum_pre/zephyr-7b-sft-full/iter2-1step --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/reproduce_outputs/iter1-ckpt_iterative --input_dir generated/zephyr-7b-sft-full-reproduce/iter2/synthetic --output_dir logprob/zephyr-7b-sft-full/iter1-2steps --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/reproduce_outputs/iter1-ckpt_iterative --input_dir generated/zephyr-7b-sft-full-reproduce/iter2/synthetic --output_dir logprob/zephyr-7b-sft-full/iter1-2steps --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter2-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter2-1step --split train --data_type generated

#### Llama ####
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model neuralmagic/Llama-2-7b-ultrachat200k --input_dir generated/Llama-2-7b-ultrachat200k/iter0/synthetic --output_dir logprob_sum/Llama-2-7b-ultrachat200k/vanilla --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model neuralmagic/Llama-2-7b-ultrachat200k --input_dir generated/Llama-2-7b-ultrachat200k/iter0/synthetic --output_dir logprob_sum/Llama-2-7b-ultrachat200k/vanilla --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter0-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter0/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/vanilla --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter0-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter0/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/vanilla --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter1-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter1/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter0 --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter1-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter1/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter0 --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter2-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter2/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter1-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter2-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter2/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter1-1step --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter3-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter3/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter2-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/Llama-2-7b-ultrachat200k/outputs/iter3-ckpt --input_dir generated/Llama-2-7b-ultrachat200k/iter3/synthetic --output_dir logprob_sum_pre/Llama-2-7b-ultrachat200k/iter2-1step --split train --data_type generated



# sft-generated
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/sft_results/zephyr-7b-sft-full/outputs_generated/iter0/final --input_dir generated/zephyr-7b-sft-full-sft-generated/iter1/synthetic --output_dir logprob_sum/zephyr-7b-sft-full_sft_generated/iter0 --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/sft_results/zephyr-7b-sft-full/outputs_generated/iter0/final --input_dir generated/zephyr-7b-sft-full-sft-generated/iter1/synthetic --output_dir logprob_sum/zephyr-7b-sft-full_sft_generated/iter0 --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/sft_results/Llama-2-7b-ultrachat200k/outputs_generated/iter0/final --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter2-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/sft_results/Llama-2-7b-ultrachat200k/outputs_generated/iter0/final --input_dir generated/zephyr-7b-sft-full-new/iter3/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter2-1step --split train --data_type generated

# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter3-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter4/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter3-1step --split train --data_type real
# accelerate launch --num_processes 4 --main_process_port=2950 spin/KL.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/new_outputs/iter3-ckpt --input_dir generated/zephyr-7b-sft-full-new/iter4/synthetic --output_dir logprob_sum/zephyr-7b-sft-full/iter3-1step --split train --data_type generated

