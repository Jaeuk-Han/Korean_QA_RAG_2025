import argparse
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
default_output_dir = f"./output/run_{timestamp}"
default_infer_output_path = f"./infer_result/inference_result_{timestamp}.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with custom dataset")
    parser.add_argument("--train_data_path", type=str, default="data/highest_with_question_type(train).json", help="Path to the training data")
    parser.add_argument("--eval_data_path", type=str, default="data/highest_with_question_type(dev).json", help="Path to the evaluation data")
    parser.add_argument("--few_shot_data_path", type=str, default="data/korean_language_rag_V1.0_train.json", help="Path to the few-shot examples data")
    parser.add_argument("--instruct_path", type=str, default="data/instruct.json", help="Path to the instruction file")
    parser.add_argument("--model_name", type=str, default="kakaocorp/kanana-nano-2.1b-instruct", help="Pretrained model name")
    parser.add_argument("--cache_dir", type=str, default="/media/nlplab/hdd1/cache_dir", help="Cache directory for model and tokenizer")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help="Output directory for model checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=6, help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate before a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Number of steps between saving checkpoints")
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"], help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    
    parser.add_argument("--num_few_shot_data", type=int, default=1, help="Number of few-shot examples to use")
    
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations to sample per input for reward computation")
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["steps", "epoch"], help="Checkpoint saving strategy")
    parser.add_argument("--load_best_model_at_end", action='store_true', help="Load the best model at the end of training")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss", help="Metric to use for determining the best model")
    parser.add_argument("--greater_is_better", action='store_true', help="Whether a higher metric value is better")
    parser.add_argument("--save_safetensors", action='store_true', help="Save model in safetensors format") # 추가함. RuntimeError: Some tensors share memory
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--use_lora", action='store_true', help="Whether to use LoRA for training")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=list, default=["q_proj", "v_proj", "k_proj", "o_proj"], help="Target modules for LoRA")
    
    parser.add_argument("--beta", type=float, default=1.0, help="Hyperparameter for loss scaling")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for candidate")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top_p for candidate")
    parser.add_argument("--top_k", type=int, default=50, help="Top_k for candidate")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="max_new_tokens for candidate")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="repetition_penalty for candidate")
    
    parser.add_argument("--use_quant", action='store_true', help="Whether to use quantization for the model")
    parser.add_argument("--quant_type", type=str, default="8bit", choices=["8bit", "4bit"], help="Type of quantization to use")

    parser.add_argument("--inference_output_path", type=str, default=default_infer_output_path, help="Path to save inference results after each epoch")

    return parser.parse_args()