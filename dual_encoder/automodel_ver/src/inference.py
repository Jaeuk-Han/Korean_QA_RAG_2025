import argparse
import torch
import wandb

from transformers import (
    AutoTokenizer, Trainer, TrainingArguments, AutoModel, BitsAndBytesConfig
)

from model import DualEncoder, Dualcollator, top_acc

from util import (
    load_json,
    load_grammar_contexts,
    load_qa_examples_for_softmax,
    load_qa_examples_with_hard_negatives
)

from peft import LoraConfig, get_peft_model

from args import get_args


def main():
    args = get_args()
    
    grammar = load_json(args.grammar_path)
    grammar_pool = load_grammar_contexts(grammar)
    
    test_data = load_json(args.test_path)
    
    # 모델 및 Tokenizer 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    quantization_config = None
    # 양자화 사용이 활성화된 경우
    if args.use_quant:
        # compute_dtype 설정
        if args.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True, # 이중 양자화로 메모리 추가 절약
        )
    
    base_model = AutoModel.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            quantization_config=quantization_config,
            device_map={"": 0} # device_map 설정 필수
        )    
    # LoRA 설정
    if args.use_lora:
        
        lora_config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules.split(",")
        }
        
        base_model = get_peft_model(base_model, LoraConfig(**lora_config))
    
    emb_model = DualEncoder(base_model, temperature=args.temperature)
    emb_model.eval()  # 추론 모드로 설정
    
    grammar_pool = load_json(grammar_path)
    context_texts = load_grammar_contexts(self.grammar_pool)
    context_ids = [f"{g['category']}|{g['rule_id']}" for g in self.grammar_pool]
    
        