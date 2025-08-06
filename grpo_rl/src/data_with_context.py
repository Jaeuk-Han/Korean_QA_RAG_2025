import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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
            description = ex.get("Description", "")

            if question and answer:
                messages.append({"role": "user", "content": f"[질문]\n{question}"})
                messages.append({"role": "assistant", "content": f"[답변]\n{answer}"})
                if description: # data 내에 description이 존재하는 경우 포함
                    messages.append({"role": "assistant", "content": f"[관련 문법 조항]\n{description}"}) 
        
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
            description = ex.get("Description", "")

            if question and answer:
                messages.append({"role": "user", "content": f"[질문]\n{question}"})
                messages.append({"role": "assistant", "content": f"[답변]\n{answer}"})
                if description: # data 내에 description이 존재하는 경우 포함
                    messages.append({"role": "assistant", "content": f"[관련 문법 조항]\n{description}"})
        return messages


class CustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_path,  
        tokenizer, 
        instruct, 
        few_shot_generater,
        ):

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.instruct = instruct # instruct는 {"prompt": "시스템 메시지 내용"} 형태라고 가정
        self.few_shot_generater = few_shot_generater
        self.printed_once = False
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        question = data.get("question") or data.get("input", {}).get("question") or data.get("Question")
        
        reference_answer = data.get("answer") or data.get("output", {}).get("answer") or data.get("Answer")

        if reference_answer is None:
            raise ValueError(f"[reference_answer 없음] idx={idx}, keys={list(data.keys())}, data={data}")

        # 1. 기본 메시지 구성
        q_type = data.get("Question_Type") or data.get("input", {}).get("question_type")
        if q_type == "선택형":
            system_prompt = self.instruct["prompt"] + self.instruct["selection_prompt"]
        elif q_type == "교정형":
            system_prompt = self.instruct["prompt"] + self.instruct["correction_prompt"]
        else:
            raise ValueError(f"[알 수 없는 question_type] idx={idx}, type={q_type}")

        # base_messages = [{"role": "system", "content": system_prompt}]

        # Category 여부 확인해서 few-shot 메시지 생성 방식 분기
        # if isinstance(self.few_shot_generater, Category_FewShotGenerater):
        #     few_shot_messages = self.few_shot_generater.generate_few_shot_messages(q_type)
        # else:
        #     few_shot_messages = self.few_shot_generater.generate_few_shot_messages()

        # base_messages.extend(few_shot_messages)
        # base_messages.append({"role": "user", "content": f"[질문]\n{question}"})

        base_messages = [{"role": "system", "content": system_prompt}]

        if self.few_shot_generater is not None:
            if isinstance(self.few_shot_generater, Category_FewShotGenerater):
                few_shot_messages = self.few_shot_generater.generate_few_shot_messages(q_type)
            else:
                few_shot_messages = self.few_shot_generater.generate_few_shot_messages()
            base_messages.extend(few_shot_messages)

        base_messages.append({"role": "user", "content": f"[질문]\n{question}"})

        chat_prompt = self.tokenizer.apply_chat_template(
            base_messages,
            add_generation_prompt=True,
            tokenize=False
        )

        if not self.printed_once:
            print(f"[DEBUG] prompt for idx {idx}:\n{chat_prompt}\n{'-'*80}")
            self.printed_once = True

        prompt_ids = self.tokenizer(chat_prompt, return_tensors="pt", padding=False).input_ids[0]

        return {
            "prompt": chat_prompt,
            "input_ids": prompt_ids,
            "attention_mask": (prompt_ids != self.tokenizer.pad_token_id).long(),
            "references": reference_answer,
            "samples": data
        }
