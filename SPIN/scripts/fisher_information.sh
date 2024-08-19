export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# source /home/user/miniconda3/etc/profile.d/conda.sh
# conda activate spinenv


accelerate launch --num_processes 8 --main_process_port=2950 spin/fisher_information_candidate.py

# FRAC_LEN=800
# TOTAL_RECORDS=20000

# NUM_BATCHES=$((TOTAL_RECORDS / FRAC_LEN))

# for ((DATA_FRAC=0; DATA_FRAC < NUM_BATCHES; DATA_FRAC++))
# do 
#     echo "Processing batch $DATA_FRAC of $NUM_BATCHEs..."
#     accelerate launch --num_processes 2 --main_process_port=2950 spin/feature_information.py --frac_len $FRAC_LEN --data_frac $DATA_FRAC

# done
# echo "Train - All batches processed"


# for ((DATA_FRAC=0; DATA_FRAC < NUM_BATCHES; DATA_FRAC++))
# do 
#     echo "Processing batch $DATA_FRAC of $NUM_BATCHEs..."
#     accelerate launch --num_processes 4 --main_process_port=2950 spin/generate.py --model "/group-volume/haoyan/models/stablelm-2-1_6b-chat" --batch_size 8 --frac_len $FRAC_LEN --data_frac $DATA_FRAC --split test --output_dir generated/stablelm-2-1_6b-chat/iter0/test-raw-generated

# done
# echo "Test - All batches processed"


# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 1 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 2 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 3 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 4 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 5 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 6 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 7 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 8 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 9 --output_dir generated/iter0
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 10 --output_dir generated/iter0

# # Generate for the test split as well
# accelerate launch --main_process_port=2950 spin/generate.py --model alignment-handbook/zephyr-7b-sft-full --input_dir UCLA-AGI/SPIN_iter0 --batch_size 8 --frac_len 800 --data_frac 0 --split test --output_dir generated/iter0