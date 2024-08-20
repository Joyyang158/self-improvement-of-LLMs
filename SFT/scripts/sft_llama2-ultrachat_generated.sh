source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate myenv


export CUDA_VISIBLE_DEVICES="0,1,2,3"

ACCELERATE_LOG_LEVEL=info

accelerate launch --config_file deepspeed_zero3.yaml --num_processes=4 --main_process_port 2950 sft.py --model neuralmagic/Llama-2-7b-ultrachat200k --input_dir /user-volume/SPIN/generated/Llama-2-7b-ultrachat200k/iter0/synthetic --output_dir /group-volume/haoyan/sft_results/Llama-2-7b-ultrachat200k/outputs_generated/iter0 --data_type generated