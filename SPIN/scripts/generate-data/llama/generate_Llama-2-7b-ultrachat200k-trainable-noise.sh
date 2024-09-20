export CUDA_VISIBLE_DEVICES=0,3,6,7

FRAC_LEN=800
TOTAL_RECORDS=20000

NUM_BATCHES=$((TOTAL_RECORDS / FRAC_LEN))

for ((DATA_FRAC=0; DATA_FRAC < NUM_BATCHES; DATA_FRAC++))
do   
    echo "Train Data -- Processing batch $DATA_FRAC of $NUM_BATCHEs..."
    accelerate launch --num_processes 4 --main_process_port=2950 SPIN/spin/generate.py --model "/blue/yonghui.wu/sgao1/haoyan/spin-results/Llama-2-7b-ultrachat200k/trainable-noised-gpt-preference-0-outputs/iter1" --batch_size 8 --frac_len $FRAC_LEN --data_frac $DATA_FRAC --output_dir /blue/yonghui.wu/sgao1/haoyan/data/trainable-noise-Llama-2-7b-ultrachat200k/iter2/train-raw-generated

done
echo "Train - All batches processed"



for ((DATA_FRAC=0; DATA_FRAC < NUM_BATCHES; DATA_FRAC++))
do 
    echo "Test Data -- Processing batch $DATA_FRAC of $NUM_BATCHEs..."
    accelerate launch --num_processes 4 --main_process_port=2950 SPIN/spin/generate.py --model "/blue/yonghui.wu/sgao1/haoyan/spin-results/Llama-2-7b-ultrachat200k/trainable-noised-gpt-preference-0-outputs/iter1" --batch_size 8 --frac_len $FRAC_LEN --data_frac $DATA_FRAC --split test --output_dir /blue/yonghui.wu/sgao1/haoyan/data/trainable-noise-Llama-2-7b-ultrachat200k/iter2/test-raw-generated

done
echo "Test - All batches processed"


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