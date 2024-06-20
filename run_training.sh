# pip3 install -r requirements.txt
export NCCL_IB_SL=1
export NCCL_IB_HCA="mlx5_0:1,mlx5_5:1"
# export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export OMP_NUM_THREADS=26

export WANDB_MODE=offline
export WANDB_API_KEY="450f5f137524092429c1579743d3941e8d31ac5d"
export WANDB_PROJECT="lm-test"
export WANDB_NAME='test'
# export WANDB_NOTES=$run_name
# export WANDB_TAGS="$exp_group"
export WANDB_DIR="."
export WANDB_SERVICE_WAIT=300

lora_trainable="embed_tokens,lm_head,q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"

echo ===== PRETRAINING =====
# Node that MODEL_PATH can be local folder path
MODEL_PATH=./models/Meta-Llama-3-8B
TITLE=llama-7b-pretrain
DATA=data/dbg

OUTPUT_DIR=outputs/test
mkdir $OUTPUT_DIR

echo ===== current OUTPUT_DIR is $OUTPUT_DIR =====
echo ===== MODEL_PATH is $MODEL_PATH =====

torchrun --nproc_per_node=4 --master_port=9919 pretrain.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --logging_steps 50 \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --use_peft True\
    --lora_trainable $lora_trainable \
    --modules_to_save $modules_to_save \
    --fsdp "full_shard auto_wrap" \
    --model_max_length 2048