
export OMP_NUM_THREADS=26

export NCCL_IB_SL=1
export NCCL_IB_HCA="mlx5_0:1,mlx5_5:1"
# export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export OMP_NUM_THREADS=26

export WANDB_MODE=online
export WANDB_API_KEY="450f5f137524092429c1579743d3941e8d31ac5d"
export WANDB_PROJECT="tamil-trial"
export WANDB_NAME='tamil-pretrain-v0.4-modified-vocab'

# v0.3: long context, new data, new lr schedule
# export WANDB_NOTES=$run_name
# export WANDB_TAGS="$exp_group"
export WANDB_DIR="."
export WANDB_SERVICE_WAIT=300

lr=1e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

#pretrained_model=./models/Sailor-4B-Chat
pretrained_model=./models/EDU-4B-Chat-New-Tokenizer-Base
tokenizer_name_or_path=${pretrained_model}
dataset_config_file=./config/data_config/tamil-pretraining-conf-v0.4.json
data_cache=.cache
per_device_train_batch_size=1
gradient_accumulation_steps=256
block_size=4096
output_dir=outputs/tamil-pretraining-conf-v0.4-modified-vocab

torchrun --nnodes 1 --nproc_per_node 8 main/pretrain.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${tokenizer_name_or_path} \
    --dataset_config_file ${dataset_config_file} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --low_cpu_mem_usage \
    --seed $RANDOM \
    --bf16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --learning_rate ${lr} \
    --warmup_ratio 0.25 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 100 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_total_limit 10 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 32 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype bfloat16 \
    --run_name $WANDB_NAME \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False

        # --fsdp "full_shard auto_wrap" \
