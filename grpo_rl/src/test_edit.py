import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from infer_data_edit import CustomDataset
from peft import PeftModel
import unicodedata
import re


def normalize_quotes(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    quote_map = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‹": "'", "›": "'",
    }
    for k, v in quote_map.items():
        text = text.replace(k, v)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_answer_from_output(output_text):
    prefixes = ["[답변]\n", "[답변] ", "[답변]", "답변: ", "답변:"]
    for prefix in prefixes:
        if prefix in output_text:
            start_idx = output_text.find(prefix) + len(prefix)
            return output_text[start_idx:].strip()
    return output_text.strip()


def main(args):
    quantization_config = None
    if args.quant == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
        "cache_dir": args.cache_dir,
        "trust_remote_code": True
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        **model_kwargs
    )
    if args.lora_weights_path:
        print(f"[DBG] Loading LoRA adapter from {args.lora_weights_path}")
        model = PeftModel.from_pretrained(model, args.lora_weights_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    fewshot_json_path="/home/nlplab/hdd1/juhyng/Korean_QA_RAG_2025/src/GRPO/data/fewshot_examples.json"

    dataset = CustomDataset(
        fname=args.input,
        tokenizer=tokenizer,
        prompt=args.system_prompt,
        fewshot_json_path=fewshot_json_path,
        correction_prompt=args.correction_prompt,
        selection_prompt=args.selection_prompt,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset)), desc="Generating..."):
        item = dataset[idx]
        input_ids = item["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids.unsqueeze(0),
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=True,
            )

        output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        output_text = extract_answer_from_output(output_text)
        output_text = normalize_quotes(output_text)

        result[idx]["output"] = {"answer": output_text}
        print(f"[DBG] idx: {idx}, output: {output_text}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--quant", type=str, choices=["4bit", "8bit"], default=None)
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, required=True)
    parser.add_argument("--correction_prompt", type=str, required=True)
    parser.add_argument("--selection_prompt", type=str, required=True)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    main(args)
