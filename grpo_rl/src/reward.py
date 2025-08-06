# BLEURT_reward_fn - tf때문에 뺌

import re
from rouge_metric import Rouge
import evaluate
import unicodedata

bert_scorer = evaluate.load('bertscore')
bert_model_type = 'bert-base-multilingual-cased'

def extract_answer_and_reason(text):
    split_patterns = ['가 옳다', '이 옳다']
    for pattern in split_patterns:
        if pattern in text:
            split_idx = text.find(pattern) + len(pattern)
            answer = text[:split_idx].strip()
            reason = text[split_idx:].strip().lstrip('., ')
            return answer, reason
    return text.strip(), ""

def normalize_answer_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    quote_map = {
        "“": '"',  "”": '"',  "„": '"',  "«": '"',  "»": '"',
        "‘": "'",  "’": "'",  "‹": "'",  "›": "'",
    }
    for k, v in quote_map.items():
        text = text.replace(k, v)
    text = re.sub(r'["\']', '', text)

    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def split_answer_reason(true_list, pred_list):
    true_answers = []
    pred_answers = []
    true_reasons = []
    pred_reasons = []

    for true, pred in zip(true_list, pred_list):
        true_ans, true_reason = extract_answer_and_reason(true)
        pred_ans, pred_reason = extract_answer_and_reason(pred)

        true_ans = normalize_answer_text(true_ans)
        pred_ans = normalize_answer_text(pred_ans)

        true_answers.append(true_ans)
        pred_answers.append(pred_ans)

        if true_reason and pred_reason:
            true_reasons.append(true_reason)
            pred_reasons.append(pred_reason)

    return true_answers, pred_answers, true_reasons, pred_reasons

# def detect_honorific(text):
#     honorific_patterns = [
#         r'(습니다|습니까|해요|드려요|주시겠어요|해주세요|하십니다|하십시오|입니다|하셨습니다)',
#         r'(하시겠어요|부탁드립니다|바래요|바랬어요|바라겠습니다)'
#     ]
#     return any(re.search(pattern, text) for pattern in honorific_patterns)

import re

def format_reward_fn(
    prompts,
    completions,
    references=None,
    samples=None,
    completion_ids=None,
    **kwargs
):
    print(f"[DEBUG] prompts: {type(prompts)}, completions: {type(completions)}, references: {type(references)}, samples: {type(samples)}")
    print(f"[DEBUG] prompts (ex): {prompts[:1]}")
    print(f"[DEBUG] completions (ex): {completions[:1]}")
    print(f"[DEBUG] references (ex): {references[:1]}")

    if references is None or completions is None:
        raise ValueError("references or completions is None")

    rewards = []

    for ref, comp in zip(references, completions):
        score = 0.0
        stripped = comp.strip()

        # 포맷이 맞으면 기본 점수 1.0 부여
        if re.match(r'^"[^"]+"[이가] 옳다[.]?', stripped):
            score = 1.0

        rewards.append(score)

    print(f"[DEBUG] format reward: {rewards[:1]}")
    return rewards



def EM_reward_fn(
    prompts,
    completions,
    references=None,
    samples=None,
    completion_ids=None,
    **kwargs
):
    # print(f"[DEBUG] prompts: {type(prompts)}, completions: {type(completions)}, references: {type(references)}, samples: {type(samples)}")
    # print(f"[DEBUG] prompts (ex): {prompts[:1]}")
    # print(f"[DEBUG] completions (ex): {completions[:1]}")
    # print(f"[DEBUG] references (ex): {references[:1]}")

    if references is None or completions is None:
        raise ValueError("references or completions is None")

    true_answers, pred_answers, _, _ = split_answer_reason(references, completions)

    rewards = []
    for true, pred in zip(true_answers, pred_answers):
        print(f"[DEBUG] true_answer: {true}")
        print(f"[DEBUG] pred_answer: {pred}")
        acceptable_answers = true.split('#')
        reward = 1.0 if any(pred.strip() == ans.strip() for ans in acceptable_answers) else 0.0
        rewards.append(reward)

    print(f"[DEBUG] EM: {rewards[:1]}")
    return rewards

def BERTScore_reward_fn(
    prompts,
    completions,
    references=None,
    samples=None,
    completion_ids=None,
    **kwargs
):
    if references is None or completions is None:
        raise ValueError("references or completions is None")

    rewards = []
    for ref, comp in zip(references, completions):
        _, _, true_reason, pred_reason = split_answer_reason([ref], [comp])
        if true_reason and pred_reason:
            score_dict = bert_scorer.compute(
                predictions=pred_reason, references=true_reason, model_type=bert_model_type
            )
            f1_score = score_dict["f1"][0]

            rewards.append(f1_score)

        else:
            rewards.append(0.0)


    print(f"[DEBUG] BERTscore reward: {rewards[:1]}")
    return rewards


def ROUGE_1_reward_fn(
    prompts,
    completions,
    references=None,
    samples=None,
    completion_ids=None,
    **kwargs
):
    # print(f"[DEBUG] prompts (ex): {prompts[:1]}")
    # print(f"[DEBUG] completions (ex): {completions[:1]}")
    # print(f"[DEBUG] references (ex): {references[:1]}")

    if references is None or completions is None:
        raise ValueError("references or completions is None")

    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=False,
        apply_best=False,
        alpha=0.5,
        weight_factor=1.0,
    )

    rewards = []
    for ref, comp in zip(references, completions):
        _, _, true_reason, pred_reason = split_answer_reason([ref], [comp])
        if true_reason and pred_reason:
            scores = rouge_evaluator.get_scores(pred_reason, true_reason)
            if isinstance(scores, dict) and isinstance(scores["rouge-1"], list):
                reward = scores["rouge-1"][0]["f"][0]
            elif isinstance(scores, list):
                reward = scores[0]["rouge-1"]["f"]
            else:
                print("[ERROR] Unexpected ROUGE output format.")
                reward = 0.0
        else:
            reward = 0.0
        rewards.append(reward)

    print(f"[DEBUG] ROUGE-1: {rewards[:1]}")
    return rewards

def multi_reward(prompts, completions, references=None, **kwargs):
    r1 = BERTScore_reward_fn(prompts, completions, references, **kwargs)
    r2 = EM_reward_fn(prompts, completions, references, **kwargs)
    r3 = format_reward_fn(prompts, completions, references, **kwargs)
    r4 = ROUGE_1_reward_fn(prompts, completions, references, **kwargs)

    return [(b + c + (a + 3 * d) / 2) / 3 for a, b, c, d in zip(r1, r2, r3, r4)]