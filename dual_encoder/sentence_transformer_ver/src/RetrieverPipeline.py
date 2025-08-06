import json, torch, pathlib, os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import load_json, ensure_path

class RetrieverPipeline:

    def __init__(
        self,
        grammar_path: str,
        qa_path: str,
        biencoder_name: str = "jhgan/ko-sroberta-multitask",
        cross_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        device: str = "cuda:0"
    ):

        grammar_path = ensure_path(grammar_path)
        qa_path = ensure_path(qa_path)
        self.grammar_book = load_json(grammar_path)
        self.qa_dataset = load_json(qa_path)
        print(f"Context: {len(self.grammar_book)}, QA: {len(self.qa_dataset)}")


        self.device = device
        
        # 초기 검색을 위해 Dense(Bi-Encoder 사용 예정)
        # Reranker로는 Cross-Encoder 사용
        # 추후 한글에 적합한 Bi-Encoder / Cross-Encoder 실험적으로 탐색 예정.

        self.biencoder = SentenceTransformer(biencoder_name, device=device)
        self.cross_tokenizer = AutoTokenizer.from_pretrained(cross_name)
        self.cross_model = AutoModelForSequenceClassification.from_pretrained(cross_name).eval().to(device)

        self._build_context_index()

    def _build_context_index(self):
        self.context_texts = [rule['description'] for rule in self.grammar_book]
        # 결과 저장용 식별자 미리 만들어두기 '범주'|'번호' 형식으로
        self.context_ids = [f"{rule['category']}|{rule['rule_id']}" for rule in self.grammar_book]
        
        # Bi-Encoder Embedding 파트 (Context에 대해)
        self.context_embs = self.biencoder.encode(
            self.context_texts,
            batch_size=256,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        print("Context Embedding finished!")

    def retrieve(
        self,
        question_text: str,
        top_k: int = 50
    ): # COS_SIM 기반으로 상위 50개 거름 top_k는 추후 결과 보고 조절 예정 (Cross-Encoder의 Computation cost issue)
        
        if self.context_embs.shape[0] == 0: # 컨텍스트 자체가 없을 때 거름
            return []
        
        question_emb = self.biencoder.encode(
            question_text,
            convert_to_tensor=True,
        )
        # question_emb와 각각의 context_embs안의 item들 사이의 코사인 유사도 탐색
        hits = util.semantic_search(
            question_emb,
            self.context_embs,
            top_k=top_k
        )[0]

        return [
            {
                "context_id": self.context_ids[h['corpus_id']],
                "context_text": self.context_texts[h['corpus_id']],
                "bi_score": float(h['score'])
            }
            for h in hits
        ]

    def rerank(
    self,
    question_text: str,
    candidates: list,
    top_n: int = 30,
    ):
        if not candidates:
            return []

        pairs = [(question_text, cand['context_text']) for cand in candidates]
    
        device = self.cross_model.device
        inputs = self.cross_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # 흥미롭게도 "BM-K/KoSimCSE-roberta-cross-encoder"은 
        # AutoModelForSequenceClassification 기반의 num_labels = 2로 구성된
        # binary classification 모델이라 index error 발생.
        # 이후 다른 모델 실험시 모델 구조 체크 확인후 코드 수정하거나 더 범용적인 코드 구조 필요해보임.

        with torch.no_grad():
            logits = self.cross_model(**inputs).logits

            if logits.dim() == 2 and logits.size(1) == 2:
                logits = logits[:, 1] # label 중 positive 클래스만 추출해 사용
            elif logits.dim() == 1:
                pass
            else:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

        num_logits = logits.size(0)

        # 디버그용 코드들

        if num_logits != len(candidates):
            print(f"[ERROR] logits.size: {num_logits} != candidates: {len(candidates)}", flush=True)
            return []

        print(f"DEBUG logits.size(0): {num_logits}, top_n: {top_n}", flush=True)

        if num_logits == 0:
            return []

        k = min(top_n, num_logits)
        if k <= 0:
            return []

        top_vals, top_idx = torch.topk(logits, k=k)

        top_idx = top_idx.cpu().tolist()
        top_vals = top_vals.cpu().tolist()

        ranked = []
        for rank_pos, (idx, score) in enumerate(zip(top_idx, top_vals), start=1):
            cand = candidates[idx]
            ranked.append({
                "context_id": cand["context_id"],
                "context_text": cand["context_text"],
                "bi_score": cand["bi_score"],
                "cross_score": score,
                "rank": rank_pos
            })

        return ranked


    def run(
        self,
        top_k: int = 50,
        top_n: int = 30,
        save_path: str = "dual_rerank_results.json"
    ):

        # 저장 경로 지정 파트
        base_dir = os.path.dirname(os.path.abspath(__file__))
        result_dir = os.path.join(base_dir, "result")
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir, save_path)

        results = []

        for qa in tqdm(self.qa_dataset):
            q_id = qa['Question_ID']
            q_text = qa['Question']
            # 'Top-3_Lowest_context_PPL' 실제로는 5개 이름 수정 필요해보임
            gold_contexts = {c['rule_id'] for c in qa['Top-3_Lowest_context_PPL']}

            bi_cands = self.retrieve(q_text, top_k=top_k)
            ranked = self.rerank(q_text, bi_cands, top_n=top_n)

            selected_ids = [c['context_id'] for c in ranked]
            gold_hit = any(rule in ctx_id for ctx_id in selected_ids for rule in gold_contexts)

            results.append({
                "Question_ID": q_id,
                "Question": q_text,
                "Selected_Contexts": ranked,
                "Gold_rule_ids": list(gold_contexts),
                "Gold_in_topN": gold_hit
            })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 평가용 recall 계산

        recall = sum(r['Gold_in_topN'] for r in results) / len(results)
        print(f"Saved {save_path} | Recall(top {top_n}): {recall:.2%}")
        
        return results

        # 결론적으로 recall 89% 정도 나옴