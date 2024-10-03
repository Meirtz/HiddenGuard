#!/bin/bash

export WANDB_MODE=offline
export MASTER_PORT=$((29000 + RANDOM % 1000))
export CUBLAS_WORKSPACE_CONFIG=:16:8

### Llama-3-8B Config ###
#model_name_or_path=meta-llama/Llama-2-7b-chat-hf
model_name_or_path=MODEL_PATH_HERE
lorra_alpha=10
layers="10,20"
#layers="20"
#transform_layers="-1"
transform_layers="30"
output_dir="./out/Mistral-7b_HG"

echo "model_name_or_path=$model_name_or_path"
echo "output_dir=$output_dir"

accelerate launch --config_file configs/accelerate_zero1.yaml \
    --num_processes 1 --main_process_port $MASTER_PORT --deepspeed_hostfile ds_hostfile \
    hidden_guard.py \
    --model_name_or_path $model_name_or_path \
    --target_layers $layers \
    --transform_layers $transform_layers \
    --lorra_alpha $lorra_alpha \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir  $output_dir \
    --overwrite_output_dir \
    --max_steps 150 \
    --bf16 True \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --use_refusal_retain \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 1000  \
    --save_total_limit 0 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 8192 \
    --q_lora False \
    --gradient_checkpointing True \
    --report_to none \
    --log_every 1