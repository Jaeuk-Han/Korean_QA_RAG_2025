from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.distributed as dist
import json, random, torch
import numpy as np
import os

from args import parse_args
from data import CustomDataset, FewShotGenerater, Category_FewShotGenerater

from reward import multi_reward
from inference import run_inference

def set_seed(seed: int):
    """모든 라이브러리의 시드를 고정하는 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LoraSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if isinstance(model, PeftModel):
            save_path = os.path.join(args.output_dir, f"epoch_{int(state.epoch)}")
            os.makedirs(save_path, exist_ok=True)
            print(f"[LoRA Save Callback] Saving LoRA adapter to {save_path}")
            model.save_pretrained(save_path)
        else:
            print("[LoRA Save Callback] Warning: model is not a PeftModel, skipping save.")
        return control

class InferenceCallback(TrainerCallback):
    def __init__(
        self,
        model_id,
        input_path,
        output_path,
        device,
        tokenizer_id=None,
        cache_dir=None,
        lora_weights_base_path=None,
        prompt=None,
        correction_prompt=None,
        selection_prompt=None,
        quant=None,
        few_shot_generater=None
    ):
        self.model_id = model_id
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer_id = tokenizer_id
        self.device = device
        self.cache_dir = cache_dir
        self.lora_weights_base_path = lora_weights_base_path
        self.prompt = prompt
        self.correction_prompt = correction_prompt
        self.selection_prompt = selection_prompt
        self.quant = quant
        self.few_shot_generater = few_shot_generater

    def on_save(self, args, state, control, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        epoch = int(state.epoch)
        print(f"\n[Callback] Running inference AFTER save at epoch {epoch}")

        device_map = "auto" if torch.cuda.device_count() > 1 else "cuda"

        lora_weights_path = (
            os.path.join(self.lora_weights_base_path, f"epoch_{epoch}")
            if self.lora_weights_base_path is not None else None
        )

        run_inference(
            input_path=self.input_path,
            output_path=self.output_path.replace(".json", f"_epoch{epoch}.json"),
            model_id=self.model_id,
            tokenizer_id=self.tokenizer_id,
            device=device_map,
            cache_dir=self.cache_dir,
            lora_weights_path=lora_weights_path,
            prompt=self.prompt,
            correction_prompt=self.correction_prompt,
            selection_prompt=self.selection_prompt,
            quant=self.quant,
            few_shot_generater=self.few_shot_generater,
        )


def main():
    args = parse_args()
    
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # base few-shot
    # few_shot_generater = FewShotGenerater(data_path=args.few_shot_data_path, num_few_shot=args.num_few_shot_data)
    # category base few-shot
    few_shot_generater = Category_FewShotGenerater(data_path=args.few_shot_data_path, num_few_shot=args.num_few_shot_data)
    
    with open(args.instruct_path, 'r', encoding='utf-8') as f:
        instruct = json.load(f)
    
    train_dataset = CustomDataset(
        data_path=args.train_data_path,
        tokenizer=tokenizer,
        instruct=instruct,
        few_shot_generater=few_shot_generater
    )
    
    eval_dataset = CustomDataset(
        data_path=args.eval_data_path,
        tokenizer=tokenizer,
        instruct=instruct,
        few_shot_generater=few_shot_generater
    )
    
    # rf 로드
    reward_funcs=[multi_reward]

    if args.use_quant:
        # 5. BitsAndBytesConfig 설정
        print(f"Using quantization: {args.quant_type}")

        if args.quant_type == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,   # 8-bit 로드
                llm_int8_threshold=6.0
            )
        elif args.quant_type == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,   # 4-bit 로드
                bnb_4bit_quant_type="nf4",  # "fp4" or "nf4"
                bnb_4bit_use_double_quant=True,  # Double quantization
                llm_int8_threshold=6.0
            )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config if args.use_quant else None,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    
    if args.use_lora:
        print("Using LoRA for training")
        print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target_modules={args.lora_target_modules}")
        
        # 6. LoRA 설정 및 적용
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        base_model = get_peft_model(base_model, lora_config)

    grpo_config = GRPOConfig(
        num_generations=args.num_generations,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        remove_unused_columns=False,
        report_to="wandb"
    )

    # callback 사용
    inference_callback = InferenceCallback(
        model_id=args.model_name,
        input_path=args.eval_data_path,
        output_path=args.inference_output_path,
        device="auto",
        tokenizer_id=args.model_name,
        cache_dir=args.cache_dir,
        lora_weights_base_path=args.output_dir if args.use_lora else None, # 최신 LoRA adpt 로딩
        prompt=instruct["prompt"],
        correction_prompt=instruct["correction_prompt"],
        selection_prompt=instruct["selection_prompt"],
        quant=args.quant_type if args.use_quant else None,
        few_shot_generater=few_shot_generater
    )

    # GRPO Trainer 사용하게 수정
    trainer = GRPOTrainer(
        model=base_model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        callbacks=[
            LoraSaveCallback(),
            inference_callback
        ]
    )
    
    trainer.train()
    if args.use_lora:
        base_model.save_pretrained(args.output_dir)
    else:
        trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()