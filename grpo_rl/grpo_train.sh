wandb online
export WANDB_PROJECT=GRPO_Korean_QA_RAG_2025
wandb login "your_token"

export CUDA_VISIBLE_DEVICES=0,1

python src/train.py \
  --model_name "K-intelligence/Midm-2.0-Base-Instruct" \
  --load_best_model_at_end \
  --eval_strategy epoch \
  --eval_steps 100 \
  --save_steps 100 \
  --metric_for_best_model eval_loss \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --train_data_path "data/(train)Midm-2.0-Mini-Instruct.cleaned.json" \
  --eval_data_path "data/korean_language_rag_V1.0_dev.json" \
  --few_shot_data_path "data/(train)Midm-2.0-Mini-Instruct.cleaned.json" \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --use_quant \
  --num_few_shot_data 5 \
  --beta 0.1 \
  --quant_type "8bit"