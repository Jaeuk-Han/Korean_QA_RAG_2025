import os
import json
import random
from typing import List, Dict
from sentence_transformers import InputExample
from datasets import DatasetDict, Dataset
from collections import defaultdict

# 파일 존재하는지 확인하는 유틸
def ensure_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] File not found: {path}")
    return path

# json 문서 읽어오는 유틸
def load_json(path: str):
    with open(ensure_path(path), encoding="utf-8") as f:
        return json.load(f)

# json 문서 저장하는 유틸
def save_json(data: List | Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Context 딕셔너리에서 실제 Context("description") 추출하는 유틸
def load_grammar_contexts(grammar_book: List[Dict]):
    return [rule["description"] for rule in grammar_book]

# Gold + Sampling(4 Negative) 데이터셋 만들어주는 유틸
def load_qa_examples_for_softmax(
    qa_data: List[Dict],
    grammar_pool: List[str],
    num_candidates: int = 5
):
    
    examples = []

    for qa in qa_data:
        question = qa["Question"]
        gold_context = qa["Top-3_Lowest_context_PPL"][0]["description"]

        # Gold 제외한 문맥에서 네거티브 샘플링
        negatives = random.sample(
            [ctx for ctx in grammar_pool if ctx != gold_context],
            k=num_candidates - 1
        )

        candidates = [gold_context] + negatives
        random.shuffle(candidates)

        label = candidates.index(gold_context)

        examples.append({
            "question": question,
            "candidates": candidates,
            "label": label
        })

    return Dataset.from_list(examples)


# mnr용 QA + Gold 데이터셋 만들어주는 유틸 for softmax
def load_qa_examples_for_mnr(
    qa_data: List[Dict],
    grammar_pool: List[str]
):
    examples = []

    for qa in qa_data:
        question = qa["Question"]
        gold_context = qa["Top-3_Lowest_context_PPL"][0]["description"]

        # MNRLoss는 query, positive pair만 필요하며, 배치 내 다른 샘플들을 자동으로 negative로 사용하는것 확인함.
        examples.append(InputExample(texts=[question, gold_context]))

    return examples


# evaluation 후 성능 평가 유틸
def evaluate_retrieval_metrics(
    predictions: List[Dict], 
    qa_data: List[Dict], 
    top_k_values: List[int] = [1, 5, 10, 30]):

    qid2gold_ctx = {
        qa["Question_ID"]: qa["Top-3_Lowest_context_PPL"][0]["description"]
        for qa in qa_data
    }

    rank_list = []
    topk_hits = defaultdict(int)

    for ex in predictions:
        qid = ex["Question_ID"]
        retrieved = ex["Retrieved_Context"]
        gold = qid2gold_ctx.get(qid, None)

        if gold is None:
            continue

        ranks = [ctx["description"] for ctx in retrieved]
        try:
            rank = ranks.index(gold) + 1  # index + 1 = ranking
        except ValueError:
            rank = None

        if rank is not None:
            rank_list.append(rank)
            for k in top_k_values:
                if rank <= k:
                    topk_hits[k] += 1
        else:
            rank_list.append(9999)

    total = len(rank_list)
    print(f"\nRetrieval Evaluation Results (Total: {total})")
    for k in top_k_values:
        acc = topk_hits[k] / total * 100
        print(f"Top-{k} Accuracy: {acc:.2f}%")

    avg_rank = sum(rank_list) / total
    print(f"Gold average ranking: {avg_rank:.2f}")


# Hard Negative 기반 데이터셋 생성하는 유틸 (정말 틀린 context만 포함)
def load_qa_examples_with_hard_negatives(
    qa_data: List[Dict],
    grammar_book: List[Dict],
    num_candidates: int = 5
):
    examples = []

    # description 빠르게 찾기 위한 dict
    desc_by_category = defaultdict(list)
    for rule in grammar_book:
        desc_by_category[rule["category"]].append(rule["description"])

    for qa in qa_data:
        question = qa["Question"]
        gold_context = qa["Top-3_Lowest_context_PPL"][0]["description"]
        gold_category = qa["Top-3_Lowest_context_PPL"][0]["category"]

        # Hard Negative 후보: 같은 카테고리 내 다른 규칙들
        all_same_cat = desc_by_category.get(gold_category, [])
        hard_negatives = [ctx for ctx in all_same_cat if ctx != gold_context]

        # 후보 수 부족할 경우 전체 grammar에서 보충
        if len(hard_negatives) < (num_candidates - 1):
            remaining = num_candidates - 1 - len(hard_negatives)
            all_desc = [rule["description"] for rule in grammar_book if rule["description"] != gold_context and rule["description"] not in hard_negatives]
            extra_negatives = random.sample(all_desc, k=remaining)
            negatives = hard_negatives + extra_negatives
        else:
            negatives = random.sample(hard_negatives, k=num_candidates - 1)

        candidates = [gold_context] + negatives
        random.shuffle(candidates)
        label = candidates.index(gold_context)

        examples.append({
            "question": question,
            "candidates": candidates,
            "label": label
        })

    return Dataset.from_list(examples)