export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate spinenv


accelerate launch --config_file configs/deepspeed_zero2.yaml auxiliary-exp/PPO/main.py --config configs/config.yaml



