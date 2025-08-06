INPUT_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/src/GRPO/infer_result"
LABEL_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/data/korean_language_rag_V1.0_dev.json"

python src/evaluate_json.py \
    --input_dir "$INPUT_PATH" \
    --label_path "$LABEL_PATH"