import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data import CustomDataset, FewShotGenerater
from peft import PeftModel
import unicodedata
import re

def normalize_quotes(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    quote_map = {
        "“": '"',  "”": '"',  "„": '"',  "«": '"',  "»": '"',
        "‘": "'",  "’": "'",  "‹": "'",  "›": "'",
    }
    for k, v in quote_map.items():
        text = text.replace(k, v)

    text = re.sub(r'\s+', ' ', text).strip()
    return text



def run_inference(
    input_path,
    output_path,
    model_id,
    tokenizer_id=None,
    device="auto",
    cache_dir=None,
    lora_weights_path=None,
    prompt=None,
    correction_prompt=None,
    selection_prompt=None,
    quant=None,
    few_shot_generater=None
):
    quantization_config = None
    if quant is not None:
        print(f"[Inference] Using quantization: {quant}")
        if quant == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        **model_kwargs
    )

    if lora_weights_path:
        print(f"[Inference] Loading LoRA adapter from: {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
    tokenizer.pad_token = tokenizer.eos_token

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    dataset = CustomDataset(
        data_path=input_path,
        tokenizer=tokenizer,
        instruct={
            "prompt": prompt,
            "correction_prompt": correction_prompt,
            "selection_prompt": selection_prompt
        },
        few_shot_generater=few_shot_generater,
    )

    with open(input_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset)), desc="Generating..."):
        item = dataset[idx]
        input_ids = item["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids.unsqueeze(0),
                max_new_tokens=1024,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.05,
                num_beams=1,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                use_cache=True,
            )

        output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        for prefix in ["[답변]\n", "[답변] ", "[답변]", "답변: ", "답변:"]:
            if output_text.startswith(prefix):
                output_text = output_text.replace(prefix, "", 1)

        # 따옴표 처리
        normalized_output = normalize_quotes(output_text)

        result[idx]["output"] = {
            "answer": output_text,
            "normalized_answer": normalized_output  # 보상 함수와 비교할 때 사용
        }
        print(f"[DBG] idx: {idx}, output: {output_text}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
