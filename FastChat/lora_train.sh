#!/bin/bash

deepspeed FastChat/fastchat/train/train_lora.py \
    --model_name_or_path Llama-2-7b-chat-hf \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path data/LTE_train_data.json \
    --bf16 True \
    --output_dir output_lte_lora_llama-2_7b_chat \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --q_lora False \
    --deepspeed FastChat/playground/deepspeed_config_s2.json \