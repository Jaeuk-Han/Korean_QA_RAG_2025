import json
import random
import torch
from torch.utils.data import Dataset

class FewShotGenerater:
    def __init__(self, data_path, num_few_shot=3):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.num_few_shot = num_few_shot

    def generate_few_shot_messages(self):
        messages = []
        sampled_indices = random.sample(range(len(self.data)), self.num_few_shot)

        for i in sampled_indices:
            ex = self.data[i]
            question = ex.get("input", {}).get("question") or ex.get("Question")
            answer = ex.get("output", {}).get("answer") or ex.get("Answer")

            if question and answer:
                messages.append({"role": "user", "content": f"[질문]: {question}"})
                messages.append({"role": "assistant", "content": answer})

        return messages


class Category_FewShotGenerater:
    def __init__(self, data_path, num_few_shot=3):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.num_few_shot = num_few_shot
        self.by_type = {"선택형": [], "교정형": []}

        for ex in self.data:
            q_type = ex.get("Question_Type") or ex.get("input", {}).get("question_type")
            if q_type in self.by_type:
                self.by_type[q_type].append(ex)

    def generate_few_shot_messages(self, question_type: str):
        messages = []
        data_pool = self.by_type.get(question_type, [])

        if len(data_pool) < self.num_few_shot:
            raise ValueError(f"Few-shot 샘플이 부족합니다: {question_type} 유형에서 {len(data_pool)}개만 존재함")

        sampled = random.sample(data_pool, self.num_few_shot)

        for ex in sampled:
            question = ex.get("input", {}).get("question") or ex.get("Question")
            answer = ex.get("output", {}).get("answer") or ex.get("Answer")

            if question and answer:
                messages.append({"role": "user", "content": f"[질문]: {question}"})
                messages.append({"role": "assistant", "content": answer})

        return messages


class CustomDataset(Dataset):
    def __init__(
        self, 
        data_path,  
        tokenizer, 
        instruct, 
        few_shot_generater=None,
    ):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.instruct = instruct
        self.few_shot_generater = few_shot_generater
        self.printed_once = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        question = data.get("question") or data.get("input", {}).get("question") or data.get("Question")
        reference_answer = data.get("answer") or data.get("output", {}).get("answer") or data.get("Answer")

        if reference_answer is None:
            raise ValueError(f"[reference_answer 없음] idx={idx}, data={data}")

        q_type = data.get("Question_Type") or data.get("input", {}).get("question_type")
        if q_type == "선택형":
            system_prompt = self.instruct["prompt"] + self.instruct["selection_prompt"]
        elif q_type == "교정형":
            system_prompt = self.instruct["prompt"] + self.instruct["correction_prompt"]
        else:
            raise ValueError(f"[알 수 없는 question_type] idx={idx}, type={q_type}")

        messages = [{"role": "system", "content": system_prompt}]

        if self.few_shot_generater:
            if isinstance(self.few_shot_generater, Category_FewShotGenerater):
                few_shot = self.few_shot_generater.generate_few_shot_messages(q_type)
            else:
                few_shot = self.few_shot_generater.generate_few_shot_messages()
            messages.extend(few_shot)

        messages.append({"role": "user", "content": f"[질문]: {question}"})

        input_tensor = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = input_tensor[0]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        if not self.printed_once:
            print("[DEBUG] === 디코딩된 프롬프트 ===")
            for msg in messages:
                role = msg["role"].upper()
                print(f"[{role}]\n{msg['content']}\n")
            self.printed_once = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompt_text,
            "references": reference_answer,
            "samples": data
        }

