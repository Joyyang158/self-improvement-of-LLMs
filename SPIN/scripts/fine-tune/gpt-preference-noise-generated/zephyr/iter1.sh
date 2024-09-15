# Set which GPU devices to be visible to the process, --num_processes should be adjusted accordingly
# source /home/sgao1/anaconda3/etc/profile.d/conda.sh
# conda activate spinenv


export CUDA_VISIBLE_DEVICES="3,4,5,6"


# Set the home directory for Hugging Face transformers library cache.
#export HF_HOME="${your_hf_home}"

# Set the logging level for the `accelerate` library to output informational messages.
ACCELERATE_LOG_LEVEL=info

# Launches a distributed training job with the `accelerate` CLI tool. Key parameters include:
# --config_file: Path to the DeepSpeed configuration file. This file defines distributed training options and optimizations.
# --num_processes: Sets the number of processes to launch, typically equal to the number of GPUs available for parallel training.
# Additional override options (specified at command line) that can alter settings defined in config.yaml:
# --num_train_epochs=6: Specifies the total number of training epochs.
# --learning_rate=1e-7: Sets the learning rate for the training process.
# --beta=0.1: Custom beta parameter value.
# --warmup_ratio=0.1: Defines the warmup ratio for learning rate scheduling.
# --output_dir="${path_to_save_checkpoint}": Directory where training checkpoints will be saved.
# Execution command: Runs 'spin/run_spin.py' with 'configs/config.yaml' as its configuration.

# microsoft/Phi-3-mini-4k-instruct
# /group-volume/haoyan/models/gemma-1.1-2b-it
# alignment-handbook/zephyr-7b-sft-full


nohup accelerate launch --config_file SPIN/configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port 2950 SPIN/spin/run_spin.py SPIN/configs/gpt-preference-0/zephyr/iter1.yaml --num_train_epochs=3 --output_dir="/blue/yonghui.wu/sgao1/haoyan/spin-results/zephyr-7b-sft-full/generated-noised-gpt-preference-0-outputs/iter1" > job_output_noise_iter1.log 2>&1 &
