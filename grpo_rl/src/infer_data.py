import json
import torch
from torch.utils.data import Dataset

PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. \
당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 또한 당신은 문법 교정 전문가입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
단, 동일한 문장을 절대 반복하지 마시오. 답변은 올바른 문장 전체를 반드시 큰따옴표(" ")로 감싸고, "가 옳다" 형식으로 작성하시오. [답변]이나 줄바꿈을 절대 포함하지 마시오.'''

class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, prompt=PROMPT, correction_prompt="", selection_prompt=""):
        self.inputs = []
        self.prompt = prompt
        self.correction_prompt = correction_prompt
        self.selection_prompt = selection_prompt
        self.tokenizer = tokenizer

        # 교정형 few-shot 예시
        self.few_shot_examples_correction = [
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"부담없이 먹어라."''',
                '''"부담 없이 먹어라."가 옳다. 명사 '부담'과 부사 '없이'는 별개의 단어이므로 띄어 쓴다. '부담없이'라는 말은 하나의 단어가 아니다. 한편 '어처구니없이', '경황없이'와 같이 '없이'가 다른 말과 결합하여 하나의 단어를 이룬 경우도 있으므로 주의해야 한다.'''
            ),
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"아가씨는 윗채에 계십니다."''',
                '''"아가씨는 위채에 계십니다."가 옳다. '윗-'과 '웃-'은 명사 '위'에 맞추어 '윗-'으로 통일한 형태를 표준어로 쓰되, '아래, 위'의 대립이 없는 단어의 경우에는 '웃-'을 표준어로 삼는다. 단, 된소리나 거센소리 앞에서는 '윗-' 대신 '위-'를 쓴다. '위채'는 '아래채'와 같이 '아래, 위'의 대립이 존재하므로 '웃-'을 쓰지 않고, 거센소리인 'ㅊ' 앞에 붙기에 '윗-' 대신 '위-'를 쓴다. 따라서 '위채'로 적어야 한다.'''
            ),
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n― 행정과-2330(2002.1.6)호''',
                '''"행정과-2330(2002. 1. 6.)호"가 옳다. 아라비아 숫자만으로 연월일을 표시할 때 마침표를 쓴다. '일'을 나타내는 마침표를 생략하는 것은 글자로 치면 '일'을 쓰지 않는 것과 같다. 그러므로 '일'을 나타내는 마침표를 생략해서는 안 된다.'''
            ),
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"요청하신 대로 페이지당 내용을 늘였습니다."''',
                '''"요청하신 대로 페이지당 내용을 늘렸습니다."가 옳다. '늘이다'는 '더 길어지게 하다'의 뜻을 나타내고, '늘리다'는 '더 커지거나 많아지게 하다'의 뜻을 나타낸다. 페이지당 내용의 분량을 더 많아지게 했다는 뜻이므로 '늘리다'를 써야 옳다.'''
            ),
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"청소년의 경우에는 담배를 끊거나 흡연양을 줄이는 효과 이외에 흡연 시작을 예방하는 효과도 있다."''',
                '''"청소년의 경우에는 담배를 끊거나 흡연량을 줄이는 효과 이외에 흡연 시작을 예방하는 효과도 있다."가 옳다. '흡연량'은 한자어끼리 결합된 단어로, 이 경우 두음 법칙을 적용하지 않으므로 '량'이 옳다.'''
            ),
            (
                '''다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n"한 번만이라도 다시 볼 수 있다면 얼마나 좋으리요."''',
                '''"한 번만이라도 다시 볼 수 있다면 얼마나 좋으리오."가 옳다. 종결 어미 '-으리오'는 혼잣말에 쓰이는 것으로, '-요'와는 형태가 다르다. '-으리오'가 표준이다.'''
            ),
            (
                '''다음 문장이 어문 규범에 부합하도록 문장 부호를 추가하고, 그 이유를 설명하세요.\n― 우등생인 민수( )도, 까지, 조차, 마저( ) 불합격이라니 놀랍지 않을 수 없다.''',
                '''"우등생인 민수{도, 까지, 조차, 마저} 불합격이라니 놀랍지 않을 수 없다."가 옳다. 열거된 항목 중에서 어느 하나가 자유롭게 선택될 수 있음을 나타낼 때는 중괄호를 사용한다.'''
            ),
        ]

        # 선택형 few-shot 예시
        self.few_shot_examples_selection = [
            (
                '''"{캐찹/케첩}을 뿌렸다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"케첩을 뿌렸다."가 옳다. '케첩(ketchup)'의 발음은 [ˈkɛtʃəp]으로 '어'에 해당하는 소리이므로 '첩'으로 적는 것이 맞다.'''
            ),
            (
                '''"여름이라 더워서 몸이 축축 {느러진다/늘어진다}." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"여름이라 더워서 몸이 축축 늘어진다."가 옳다. '늘어지다'는 원형 '늘다'의 의미가 유지되는 경우이므로 원형을 밝혀 적는다.'''
            ),
            (
                '''"{거칠은/거친} 들판의 솔잎이 푸르르다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"거친 들판의 솔잎이 푸르르다."가 옳다. '거친'은 '거칠다'의 활용형이다. '거칠다'의 어간 끝 받침 'ㄹ'은 'ㄴ, ㅂ, ㅅ, -오, -ㄹ'로 시작하는 어미 앞에서 나타나지 않으면 나타나지 않는 대로 적는다. 따라서 '거치니', '거친', '거칩니다'가 옳다.'''
            ),
            (
                '''"우리끼리는 그냥 편하게 {불렀던가/불렀든가} 그랬죠." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"우리끼리는 그냥 편하게 불렀던가 그랬죠."가 옳다. 과거 경험을 말할 때는 '-던가'를 사용해야 한다.'''
            ),
            (
                '''"금성은 다른 말로 {샛별/새벽별}이라고 한다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"금성은 다른 말로 샛별이라고 한다."가 옳다. '샛별'만이 표준어이며 '새벽별'은 비표준어다.'''
            ),
            (
                '''"한 40분 정도 기다리셔야 {되요/돼요}." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"한 40분 정도 기다리셔야 돼요."가 옳다. '되다'의 활용형 '되어요'는 줄여서 '돼요'로 쓴다.'''
            ),
            (
                '''"집으로 {걸려 온/걸려온} 전화를 받았다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"집으로 걸려 온 전화를 받았다."가 옳다. 본용언과 본용언의 구성이므로 띄어 쓴다. 보조 용언일 경우에만 붙여 쓰는 것이 허용된다.'''
            ),
            (
                '''"우리는 컴퓨터로 {시뮬레이션/시뮬레이숀}을 했어요." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.''',
                '''"우리는 컴퓨터로 시뮬레이션을 했어요."가 옳다. 'simulation'의 발음 [ʃən]은 '션'으로 표기한다.'''
            ),
        ]

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
                message.append({"role": "user", "content": f"[질문]: {q}"})
                message.append({"role": "assistant", "content": a})

            message.append({"role": "user", "content": f"[질문]: {current_question}"})

            # 최초 1개 디버깅용 출력
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
