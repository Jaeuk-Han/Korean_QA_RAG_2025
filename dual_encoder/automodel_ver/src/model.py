import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from transformers import AutoModel
from torch import Tensor
import numpy as np

class Dualcollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 128
    ):
        self.tk, self.L = tokenizer, max_length
        self.task_description = 'Given a web search query, retrieve relevant passages that answer the query'

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery: {query}'

    def __call__(self, batch):
        questions = [self.get_detailed_instruct(ex['question']) for ex in batch]
        contexts = [ex["candidates"] for ex in batch]

        labels = [ex["label"] for ex in batch] # [label(int)]

        flat_c = sum(contexts, [])  # B * K (B - Batch, K - Candidate) / context should be list for tokenizer
        B, K = len(questions), len(contexts[0])

        q_enc = self.tk(questions, padding=True, truncation=True,
                        max_length=self.L, return_tensors="pt")
        c_enc = self.tk(flat_c, padding=True, truncation=True,
                          max_length=self.L, return_tensors="pt")

        for k in c_enc: # unflatten context
            c_enc[k] = c_enc[k].view(B, K, -1)

        return {
            "input_ids_q": q_enc["input_ids"],
            "attention_mask_q": q_enc["attention_mask"],
            "input_ids_c": c_enc["input_ids"],
            "attention_mask_c": c_enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class DualEncoder(nn.Module):
    def __init__(
        self,
        model: AutoModel,
        temperature: float = 1.0
    ):
        super().__init__()
        self.model = model
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature))))
    
    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def _encode(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        pooled = self.last_token_pool(last_hidden, attention_mask)
        return pooled

    def forward(
            self,
            input_ids_q, attention_mask_q,
            input_ids_c, attention_mask_c,
            labels=None
    ):
        
        B, K, L = input_ids_c.size()

        q = self._encode(input_ids_q, attention_mask_q) # QA - CLS / mean polling
        
        c = self._encode(
            input_ids_c.view(B*K, L),
            attention_mask_c.view(B*K, L)
        ).view(B, K, -1) # Context - CLS / / mean polling
        
        q = F.normalize(q, p=2, dim=-1)
        c = F.normalize(c, p=2, dim=-1)

        tau = torch.exp(self.log_tau)  # temperature parameter
        logits = (q.unsqueeze(1) * c).sum(dim=-1) / tau # COS-SIM == torch.matmul
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        
        return {"loss": loss, "logits": logits}

def top_acc(eval_pred): # metric for trainer
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    
    # top 2 안에 있는지 확인
    top2 = np.argsort(-logits, axis=1)[:, :2]     # 내림차순 상위 30
    topk_acc = (top2 == labels[:, None]).any(axis=1).mean()
    
    # 정답이 예측된 순위 몇 위인지 확인
    ranks = np.argsort(-logits, axis=1)  # 내림차순 정렬
    rank_indices = np.where(ranks == labels[:, None])  # 정답의 인덱스 찾기
    rank = rank_indices[1]

    return {"accuracy": float(acc), "top_k_accuracy": float(topk_acc), "mean_rank": float(rank.mean())}