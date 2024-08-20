accelerate launch --config_file configs/deepspeed_zero3.yaml \
    auxiliary-exp/PPO/main_v2.py \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path /group-volume/haoyan/models/zephyr-7b-sft-full \
    --sft_model_path /group-volume/haoyan/models/zephyr-7b-sft-full \
    --reward_model_path OpenAssistant/reward-model-deberta-v3-large-v2 \
    --local_rollout_forward_batch_size 1 \
    --non_eos_penalty \