import os
import json
import argparse

from reward import (
    EM_reward_fn,
    BERTScore_reward_fn,
    ROUGE_1_reward_fn,
)

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True, help="infer_result")
parser.add_argument("--label_path", type=str, required=True, help="answer")
args = parser.parse_args()

with open(args.label_path, "r", encoding="utf-8") as f:
    label_data = {item["id"]: item["output"]["answer"] for item in json.load(f)}

all_results = []

for filename in os.listdir(args.input_dir):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(args.input_dir, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [item["input"]["question"] for item in data]
    completions = [item["output"]["answer"] for item in data]
    references = [label_data.get(item["id"], "") for item in data]

    try:
        rouge_scores = ROUGE_1_reward_fn(prompts, completions, references)
        bert_scores = BERTScore_reward_fn(prompts, completions, references)
        em_scores = EM_reward_fn(prompts, completions, references)

        semantic_scores = [(r + b) / 2 for r, b in zip(rouge_scores, bert_scores)]
        final_scores = [(s + e) / 2 for s, e in zip(semantic_scores, em_scores)]

        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_bert = sum(bert_scores) / len(bert_scores)
        avg_em = sum(em_scores) / len(em_scores)
        avg_semantic = sum(semantic_scores) / len(semantic_scores)
        avg_final = sum(final_scores) / len(final_scores)

        print(f"\n File: {filename}")
        print(f"   - ROUGE-1        : {avg_rouge:.4f}")
        print(f"   - BERTScore      : {avg_bert:.4f}")
        print(f"   - EM             : {avg_em:.4f}")
        print(f"   - (ROUGE+BERT)/2 : {avg_semantic:.4f}")
        print(f"   - Final Mean     : {avg_final:.4f}")

        all_results.append((filename, avg_final))

    except Exception as e:
        print(f"[ERROR] {filename}: {e}")

if all_results:
    print("\n Total Final Mean:")
    all_results.sort(key=lambda x: x[1], reverse=True)
    for name, score in all_results:
        print(f"   - {name:40s}: {score:.4f}")

    best_file, best_score = all_results[0]
    print("\n Best Final Mean:")
    print(f"   File       : {best_file}")
    print(f"   Final Mean : {best_score:.4f}")
