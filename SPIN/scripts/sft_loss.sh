export CUDA_VISIBLE_DEVICES=0,1,2,3,4

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate spinenv


# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/models/Llama-2-7b-chat-hf --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/vanilla --split train
# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter0-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/iter0 --split train
# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter1-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/iter1 --split train
# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter2-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/iter2 --split train
# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter3-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/iter3 --split train

# WizardLM
# accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/models/WizardLM-2-7B --input_dir generated/WizardLM-2-7B/iter0/synthetic --output_dir generated_loss/WizardLM-2-7B/vanilla --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter0-ckpt --input_dir generated/WizardLM-2-7B/iter0/synthetic --output_dir generated_loss/WizardLM-2-7B/iter0 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter0-ckpt --input_dir generated/WizardLM-2-7B/iter1/synthetic --output_dir generated_loss/WizardLM-2-7B/iter0 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter1-ckpt --input_dir generated/WizardLM-2-7B/iter1/synthetic --output_dir generated_loss/WizardLM-2-7B/iter1 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter1-ckpt --input_dir generated/WizardLM-2-7B/iter2/synthetic --output_dir generated_loss/WizardLM-2-7B/iter1 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter2-ckpt --input_dir generated/WizardLM-2-7B/iter2/synthetic --output_dir generated_loss/WizardLM-2-7B/iter2 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter2-ckpt --input_dir generated/WizardLM-2-7B/iter3/synthetic --output_dir generated_loss/WizardLM-2-7B/iter2 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/WizardLM-2-7B/outputs/iter3-ckpt --input_dir generated/WizardLM-2-7B/iter3/synthetic --output_dir generated_loss/WizardLM-2-7B/iter3 --split train


# zephyr
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/models/zephyr-7b-sft-full --input_dir generated/zephyr-7b-sft-full/iter0/synthetic --output_dir generated_loss/zephyr-7b-sft-full/vanilla --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter0-ckpt --input_dir generated/zephyr-7b-sft-full/iter0/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter0 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter0-ckpt --input_dir generated/zephyr-7b-sft-full/iter1/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter0 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter1-ckpt --input_dir generated/zephyr-7b-sft-full/iter1/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter1 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter1-ckpt --input_dir generated/zephyr-7b-sft-full/iter2/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter1 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full/iter2/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter2 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter2-ckpt --input_dir generated/zephyr-7b-sft-full/iter3/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter2 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/zephyr-7b-sft-full/outputs/iter3-ckpt --input_dir generated/zephyr-7b-sft-full/iter3/synthetic --output_dir generated_loss/zephyr-7b-sft-full/iter3 --split train

# llama

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model huggyllama/llama-7b --input_dir generated/llama-7b/iter0/synthetic --output_dir generated_loss/llama-7b/vanilla --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter0-ckpt --input_dir generated/llama-7b/iter0/synthetic --output_dir generated_loss/llama-7b/iter0 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter0-ckpt --input_dir generated/llama-7b/iter1/synthetic --output_dir generated_loss/llama-7b/iter0 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter1-ckpt --input_dir generated/llama-7b/iter1/synthetic --output_dir generated_loss/llama-7b/iter1 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter1-ckpt --input_dir generated/llama-7b/iter2/synthetic --output_dir generated_loss/llama-7b/iter1 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter2-ckpt --input_dir generated/llama-7b/iter2/synthetic --output_dir generated_loss/llama-7b/iter2 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter2-ckpt --input_dir generated/llama-7b/iter3/synthetic --output_dir generated_loss/llama-7b/iter2 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/llama-7b/outputs/iter3-ckpt --input_dir generated/llama-7b/iter3/synthetic --output_dir generated_loss/llama-7b/iter3 --split train


# llama-2

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/models/Llama-2-7b-chat-hf --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/vanilla --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter0-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter0 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter0-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter1/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter0 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter1-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter1/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter1 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter1-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter2/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter1 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter2-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter2/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter2 --split train
accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter2-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter3/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter2 --split train

accelerate launch --num_processes 5 --main_process_port=2950 spin/sft_loss.py --model /group-volume/haoyan/spin_results/Llama-2-7b-chat-hf/outputs/iter3-ckpt --input_dir generated/Llama-2-7b-chat-hf/iter3/synthetic --output_dir generated_loss/Llama-2-7b-chat-hf/iter3 --split train