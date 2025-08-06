# 실행 환경 설정
GRAMMAR_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/data/GrammarBook_structured.json"
QA_PATH="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/getPerplexity/result/lowest_ppl_results(kanana_basic_top5).json"
OUT_PATH="dual_rerank_results.json"
DEVICE="cuda:0"

# 모델 이름 (한국어에 적합한 Bi-encoder + Cross-encoder)
BIENCODER="dragonkue/multilingual-e5-small-ko-v2"
CROSSENCODER="BM-K/KoSimCSE-roberta"

# 파이프라인 실행
python main.py \
  --grammar "$GRAMMAR_PATH" \
  --qa "$QA_PATH" \
  --device "$DEVICE" \
  --top_k 200 \
  --top_n 30 \
  --out "$OUT_PATH" \
  --biencoder_name "$BIENCODER" \
  --cross_name "$CROSSENCODER"