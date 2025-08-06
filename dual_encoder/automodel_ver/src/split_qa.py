import random
from util import load_json, save_json

def split_dataset(
    input_path: str, 
    train_path: str, 
    eval_path: str, 
    eval_ratio: float = 0.2
):

    data = load_json(input_path)
    random.shuffle(data)

    split_idx = int(len(data) * (1 - eval_ratio))
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    save_json(train_data, train_path)
    save_json(eval_data, eval_path)

    print(f"[split] Total: {len(data)} | Train: {len(train_data)} | Eval: {len(eval_data)}")

if __name__ == "__main__":
    split_dataset(
        input_path="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/getPerplexity/result/lowest_ppl_results(kanana_basic_top5).json",
        train_path="data/qa_train.json",
        eval_path="data/qa_eval.json",
        eval_ratio=0.2
    )