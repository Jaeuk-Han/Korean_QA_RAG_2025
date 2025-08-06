import argparse

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_path", type=str, required=True)
    parser.add_argument("--qa_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="/media/nlplab/hdd1/cache_dir", help="Cache directory for model and tokenizer")
    
    parser.add_argument("--output_dir", type=str, default="./outputs/dual_encoder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    
    # quantization arguments
    parser.add_argument("--use_quant", action="store_true", help="Enable 4-bit quantization (QLoRA)")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16", help="Compute dtype (float16 or bfloat16)")
    
    
    return parser.parse_args()