source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate myenv

pip install sentencepiece protobuf

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

ACCELERATE_LOG_LEVEL=info

accelerate launch --config_file deepspeed_zero3.yaml --num_processes=6 --main_process_port 2950 sft_mix.py --model /group-volume/haoyan/models/WizardLM-2-7B --input_dir /user-volume/SPIN/generated/WizardLM-2-7B/iter0/synthetic --output_dir /group-volume/haoyan/sft_results/WizardLM-2-7B/outputs_mix