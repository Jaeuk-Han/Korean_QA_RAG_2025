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

    wandb.init(
        project="dual-encoder-rag",
        name=f"run_{args.model_name.replace('/', '_')}",
        config=vars(args)
    )

    # 데이터 로드
    grammar = load_json(args.grammar_path)
    qa_data = load_json(args.qa_path)
    val_data = load_json(args.val_path)
    grammar_pool = load_grammar_contexts(grammar)

    # 학습 데이터 생성
    train_examples = load_qa_examples_for_softmax(qa_data, grammar_pool)
    val_examples = load_qa_examples_for_softmax(val_data, grammar_pool)

    # train_examples = load_qa_examples_with_hard_negatives(qa_data, grammar)
    # val_examples = load_qa_examples_with_hard_negatives(val_data, grammar)

    # 모델 및 Tokenizer 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    
    quantization_config = None
    # 양자화 사용이 활성화된 경우
    if args.use_quant:
        print("Using Quantization for training")
        print(f"Quantization type: {args.bnb_4bit_quant_type}")
        
        # compute_dtype 설정
        if args.bnb_4bit_compute_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,   # ← bfloat16 → float16
            bnb_4bit_use_double_quant=True,
        )
    
    base_model = AutoModel.from_pretrained(
            args.model_name,
            # cache_dir=args.cache_dir,
            quantization_config=quantization_config,
            device_map={"": 0} # device_map 설정 필수
        )    
    # LoRA 설정
    if args.use_lora:
        print("Using LoRA for training")
        
        lora_config = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.lora_target_modules.split(",")
        }
        print(f"\n\nLoRA config: {lora_config}\n\n")
        
        base_model = get_peft_model(base_model, LoraConfig(**lora_config))
    
    emb_model = DualEncoder(base_model, temperature=args.temperature)
    
    # Trainer 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to="wandb",
        run_name=f"run_{args.model_name.replace('/', '_')}",
        remove_unused_columns=False,
        gradient_accumulation_steps=2 # OOM
    )

    trainer = Trainer(
        model=emb_model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        data_collator=Dualcollator(tokenizer, max_length=args.max_length),
        tokenizer=tokenizer,
        compute_metrics=top_acc
    )

    trainer.train()
    # trainer.save_model(args.output_dir)

    emb_model.model.save_pretrained(args.output_dir)   # 내부 모델 저장
    tokenizer.save_pretrained(args.output_dir)   # 토크나이저 저장

    wandb.finish()


if __name__ == "__main__":
    main()