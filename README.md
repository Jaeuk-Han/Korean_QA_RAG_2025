# KOREAN_QA_RAG_2025 — Methodology Overview

This repository implements a **Korean QA / Grammar RAG** system with two major components:

1) **Retriever (Dual-Encoder)**: retrieves the most relevant *grammar-rule evidence* for each question  
2) **Generator (GRPO RL)**: produces final answers with training signals driven by verifiable / metric-based rewards

The emphasis of this README is **methodology**, not command-level execution details.

---

## 1. High-level Goal

The task requires generating **reliable Korean answers** (often with strict output style constraints) by grounding generation on **grammar-rule evidence**.  
We therefore treat the system as a two-stage pipeline:

- **Retrieval**: find the best matching grammar rule(s) given a question
- **Generation**: answer strictly in the required format, leveraging retrieved evidence when available
- **Evaluation**: quantify correctness and faithfulness using automatic metrics (and any competition-defined rules)

---

## 2. Directory Overview

- `dual_encoder/`
  - `sentence_transformer_ver/`: fast baseline retrieval (bi-encoder + cross-encoder reranking)
  - `automodel_ver/`: trainable custom dual-encoder retriever (AutoModel-based)
- `grpo_rl/`: GRPO-based RL training/inference/evaluation for the generator

---

## 3. Retriever Methodology

### 3.1 Baseline Retriever (SentenceTransformer version)

**Motivation**  
Start from a strong, simple baseline that is easy to run and interpret, and that already yields competitive retrieval recall.

**Approach**  
A two-step ranking pipeline:

1) **Bi-encoder retrieval**
   - Encode all grammar-rule descriptions into a vector index
   - Encode each question into a vector
   - Retrieve Top-K candidates by cosine similarity

2) **Cross-encoder reranking**
   - For each (question, candidate rule) pair, compute a relevance score using a cross-encoder
   - Rerank Top-K → keep Top-N as final evidence

**Why this works**
- Bi-encoder is efficient for scanning a large rule set
- Cross-encoder provides higher precision due to joint encoding of (question, evidence)

**Primary metric for retriever quality**
- *Recall@N*: whether the gold/target rule is present in the top-N retrieved results  
  (This aligns well with downstream generation success when evidence is required.)

---

### 3.2 Trainable Retriever (AutoModel Dual-Encoder version)

**Motivation**  
A baseline ST pipeline can be limited by fixed embeddings or weak domain adaptation.  
We therefore train a custom dual-encoder to better match the dataset distribution and the grammar-rule space.

**Core model**
- Two encoders (often weight-shared) produce:
  - `q = Enc(question)`
  - `c_i = Enc(context_i)` for candidate contexts
- Similarity is computed (typically cosine / dot product), producing logits over candidates.

**Training objective: softmax classification**
- For each question, construct a candidate set:
  - 1 **gold** context (correct rule)
  - N **negative** contexts (incorrect rules)
- Train with cross-entropy so the gold context receives the highest probability.

**Negative sampling strategies**
1) **Random negatives**
   - Simple and broad coverage
2) **Hard negatives**
   - Prefer negatives from the same category or semantically similar rules
   - Improves fine-grained discrimination and reduces “easy-negative” bias

**Efficiency & adaptation**
- Supports low-memory training options like:
  - **4-bit quantization (QLoRA)**
  - **LoRA adapters**
- This enables training strong retrievers even on constrained GPU setups.

**Primary retriever evaluation**
- Recall@K / Recall@N, plus ranking-based signals (mean rank, top-1/top-2 accuracy)

---

## 4. Generator Methodology (GRPO RL)

### 4.1 Why RL here?

In this task, the final output must satisfy both:
- **Correctness** (answer should be right)
- **Strict formatting / constraints** (output must obey a specified template)

Supervised fine-tuning alone often struggles with:
- reliably enforcing formatting constraints
- preventing small-but-fatal output deviations

GRPO-based RL allows us to directly optimize **verifiable reward signals** that reflect the competition requirements.

---

### 4.2 Reward Design (Multi-reward)

The generator reward is a weighted composition of signals such as:

- **EM (Exact Match)**  
  Captures strict correctness when the expected answer is deterministic.

- **ROUGE (lexical overlap)**  
  Helpful when there is variation in wording but the core content should match.

- **BERTScore (semantic similarity)**  
  Improves robustness against paraphrases by measuring semantic overlap.

- **Format reward (constraint verification)**  
  A rule-based validator that checks whether the output matches required patterns  
  (e.g., quoting rules, fixed endings, option-selection phrasing, etc.).

**Key insight**  
Even if semantic rewards are high, a format violation can cause real competition score drop.  
A dedicated format reward stabilizes training toward constraint-following outputs.

---

### 4.3 Training behavior and stability considerations

**Sampling vs deterministic decoding**
- Sampling (`do_sample`, temperature/top-p/top-k) increases candidate diversity
- However, too much randomness can break formatting and collapse format-related rewards  
→ decoding parameters must be tuned carefully.

**Practical stability approach**
- Use conservative sampling or partial determinism to keep outputs within the valid format region
- Gradually relax sampling only if reward remains stable

---

## 5. System Integration (Retriever → Generator)

Two typical integration modes:

1) **Retrieval-augmented prompting**
   - Inject Top-N retrieved grammar rules into the generator prompt
   - The generator is instructed to answer grounded in the evidence

2) **Fallback strategy**
   - If retrieval confidence is low, answer with minimal assumptions (or request clarification if allowed)
   - If confidence is high, enforce evidence-backed reasoning

The key objective is to minimize:
- hallucinations
- rule mismatch (wrong evidence)
- formatting drift

---

## 6. Minimal Execution

This repository is organized around shell entry points.

- Retriever:
  - `dual_encoder/automodel_ver/train.sh`
  - `dual_encoder/sentence_transformer_ver/train.sh`

- Generator:
  - `grpo_rl/grpo_train.sh`
  - `grpo_rl/grpo_test.sh`
  - `grpo_rl/grpo_evaluate.sh`

In practice:
```bash
bash <script>.sh
```

---

## 7. What to Report (Recommended)

If you write an experiment log / report, the most informative items are:

- Retriever:
  - Recall@K/Recall@N, mean rank
  - ablations: random vs hard negatives, encoder choice, embedding dimension

- Generator:
  - EM / ROUGE / BERTScore
  - format validity rate
  - decoding parameter ablations (temperature/top-p/repetition penalty)

- End-to-end:
  - performance with/without retrieval evidence
  - failure mode analysis: retrieval miss vs generation hallucination vs format break
