source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate myenv


export CUDA_VISIBLE_DEVICES="0,1,2,3"

ACCELERATE_LOG_LEVEL=info

accelerate launch --config_file deepspeed_zero3.yaml --num_processes=4 --main_process_port 2950 sft.py --model /group-volume/haoyan/models/zephyr-7b-sft-full --input_dir /user-volume/SPIN/generated/zephyr-7b-sft-full/iter0/synthetic --output_dir /group-volume/haoyan/sft_results/zephyr-7b-sft-full/outputs_generated/iter0 --data_type generated