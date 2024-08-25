export CUDA_VISIBLE_DEVICES=0,1,2

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate spinenv


accelerate launch --num_processes 3 --main_process_port=2950 spin/sft_loss.py --model neuralmagic/Llama-2-7b-ultrachat200k --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss/Llama-2-7b-chat-hf_ultrachat/vanilla --split test
# accelerate launch --num_processes 2 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/sft_results/Llama-2-7b-chat-hf/outputs_generated/final --input_dir generated/Llama-2-7b-chat-hf/iter0/synthetic --output_dir sft_loss_test/Llama-2-7b-chat-hf/sft_generated --split test
# accelerate launch --num_processes 2 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/sft_results/WizardLM-2-7B/outputs_real/final --input_dir generated/WizardLM-2-7B/iter0/synthetic --output_dir sft_loss_test/WizardLM-2-7B/sft_real --split test
# accelerate launch --num_processes 2 --main_process_port=2950 spin/sft_loss.py --model  /group-volume/haoyan/sft_results/WizardLM-2-7B/outputs_generated/final --input_dir generated/WizardLM-2-7B/iter0/synthetic --output_dir sft_loss_test/WizardLM-2-7B/sft_generated --split test