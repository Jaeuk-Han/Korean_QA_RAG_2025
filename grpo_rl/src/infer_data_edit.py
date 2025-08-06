import json
import torch
from torch.utils.data import Dataset

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. \
당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 또한 당신은 문법 교정 전문가입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
단, 동일한 문장을 절대 반복하지 마시오. 답변은 올바른 문장 전체를 반드시 큰따옴표(" ")로 감싸고, "가 옳다" 형식으로 작성하시오.'''

def load_fewshot_examples(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correction, selection = [], []
    for ex in data:
        q_type = ex.get("Question_Type", "")
        question = ex.get("Question", "").strip()
        answer = ex.get("Answer", "").strip()
        description = ex.get("Description", "").strip()
        full_answer = f"[관련 어문 규범] {description}\n[답변] {answer}"
        if q_type == "교정형":
            correction.append((question, full_answer))
        elif q_type == "선택형":
            selection.append((question, full_answer))
    return correction, selection

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, fewshot_json_path,
                 prompt=PROMPT, correction_prompt="", selection_prompt=""):
        self.inputs = []
        self.prompt = prompt
        self.correction_prompt = correction_prompt
        self.selection_prompt = selection_prompt
        self.tokenizer = tokenizer

        self.few_shot_examples_correction, self.few_shot_examples_selection = load_fewshot_examples(fewshot_json_path)

        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, example in enumerate(data):
            input_obj = example["input"]
            question_type = input_obj.get("question_type", "")
            current_question = input_obj.get("question", "")

            if question_type == "교정형":
                task_prompt = self.correction_prompt
                few_shot = self.few_shot_examples_correction
            elif question_type == "선택형":
                task_prompt = self.selection_prompt
                few_shot = self.few_shot_examples_selection
            else:
                task_prompt = ""
                few_shot = []

            message = [{"role": "system", "content": self.prompt}]
            if task_prompt:
                message.append({"role": "user", "content": task_prompt})

            for q, a in few_shot:
                message.append({"role": "user", "content": q})
                message.append({"role": "assistant", "content": a})

            message.append({"role": "user", "content": current_question})

            if i == 0:
                print(f"\n[DBG] ==== Example 0 Prompt ({question_type}) ====")
                for msg in message:
                    role = msg['role'].upper()
                    print(f"[{role}]\n{msg['content']}\n")

            input_ids = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            self.inputs.append(input_ids[0])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs[idx]
        }
