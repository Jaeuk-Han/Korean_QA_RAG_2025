wandb login "your_token"
wandb online

export CUDA_VISIBLE_DEVICES=0,1

MODEL_NAME="Qwen/Qwen3-Embedding-4B"
GRAMMAR_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/data/GrammarBook_structured.json"
QA_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/src/jaeuk/data/qa_train.json"
VAL_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/src/jaeuk/data/qa_eval.json"
OUTPUT_DIR="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/src/jaeuk/outputs/dual_encoder3"
BATCH_SIZE=1
EPOCHS=10
LR=2e-5
MAX_LEN=256
SEED=42
LORA_R=8
LORA_ALPHA=16
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"
BNB_4BIT_QUANT_TYPE="nf4"

# 실행 명령
python src/train.py \
    --grammar_path "$GRAMMAR_PATH" \
    --qa_path "$QA_PATH" \
    --val_path "$VAL_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --max_length "$MAX_LEN" \
    --seed "$SEED" \
    --use_lora \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --use_quant \
    --bnb_4bit_quant_type "$BNB_4BIT_QUANT_TYPE"