"""Continued Pre-Training (CPT) script for Qwen3-8B using rocm-agent trajectory data.

This script converts multi-turn conversation data into training format and runs
supervised fine-tuning on Qwen3-8B.

Usage:
    python cpt_qwen3_8b.py \
        --data-file /path/to/large_dataset.json \
        --output-dir /path/to/output \
        --model Qwen/Qwen3-8B

For multi-GPU training:
    torchrun --nproc_per_node=8 cpt_qwen3_8b.py \
        --data-file /path/to/large_dataset.json \
        --output-dir /path/to/output
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


@dataclass
class CPTConfig:
    data_file: str = field(default="training_data/large_dataset.json")
    output_dir: str = field(default="outputs/qwen3-8b-cpt")
    model_name: str = field(default="Qwen/Qwen3-8B")
    max_length: int = field(default=32768)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=1e-5)
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: Optional[str] = field(default=None)
    min_reward_threshold: float = field(default=0.0)
    only_successful: bool = field(default=True)
    train_on_assistant_only: bool = field(default=True)
    val_split: float = field(default=0.05)


def load_trajectory_data(data_file: str, config: CPTConfig) -> list[dict]:
    """Load and filter trajectory data from JSON file."""
    data = json.loads(Path(data_file).read_text())
    examples = data["examples"]
    
    filtered = []
    for ex in examples:
        if config.only_successful and not ex.get("success", False):
            continue
        if ex.get("reward", 0) < config.min_reward_threshold:
            continue
        filtered.append(ex)
    
    print(f"Loaded {len(filtered)} examples from {len(examples)} total (filtered by reward >= {config.min_reward_threshold}, success={config.only_successful})")
    return filtered


def format_messages_to_text(messages: list[dict], tokenizer) -> str:
    """Convert messages to Qwen chat format using apply_chat_template."""
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def create_labels_mask(
    input_ids: list[int],
    messages: list[dict],
    tokenizer,
    train_on_assistant_only: bool = True,
) -> list[int]:
    """Create labels with -100 for non-assistant tokens (if train_on_assistant_only)."""
    if not train_on_assistant_only:
        return input_ids.copy()
    
    labels = [-100] * len(input_ids)
    
    # Encode each message separately to find boundaries
    current_pos = 0
    for i, msg in enumerate(messages):
        # Format single message
        single_msg = [msg]
        if i == 0:
            single_text = tokenizer.apply_chat_template(single_msg, tokenize=False, add_generation_prompt=False)
        else:
            # For subsequent messages, we need to compute the delta
            prev_msgs = messages[:i]
            prev_text = tokenizer.apply_chat_template(prev_msgs, tokenize=False, add_generation_prompt=False)
            curr_text = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
            single_text = curr_text[len(prev_text):]
        
        single_tokens = tokenizer.encode(single_text, add_special_tokens=False)
        msg_len = len(single_tokens)
        
        if msg["role"] == "assistant":
            # Include assistant response in loss computation
            for j in range(current_pos, min(current_pos + msg_len, len(labels))):
                labels[j] = input_ids[j]
        
        current_pos += msg_len
    
    return labels


def preprocess_example(example: dict, tokenizer, config: CPTConfig) -> dict:
    """Preprocess a single example for training."""
    messages = example["messages"]
    
    # Convert to text
    text = format_messages_to_text(messages, tokenizer)
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=config.max_length,
        padding=False,
        return_tensors=None,
    )
    
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # Create labels (mask non-assistant tokens if needed)
    if config.train_on_assistant_only:
        labels = create_labels_mask(input_ids, messages, tokenizer, train_on_assistant_only=True)
    else:
        labels = input_ids.copy()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_dataset(examples: list[dict], tokenizer, config: CPTConfig) -> Dataset:
    """Prepare HuggingFace Dataset from trajectory examples."""
    processed = []
    for ex in examples:
        try:
            proc = preprocess_example(ex, tokenizer, config)
            processed.append(proc)
        except Exception as e:
            print(f"Warning: Failed to process example {ex.get('instance_id', 'unknown')}: {e}")
            continue
    
    return Dataset.from_list(processed)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CPT training for Qwen3-8B on rocm-agent data")
    parser.add_argument("--data-file", type=str, default="training_data/large_dataset.json")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen3-8b-cpt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-reward", type=float, default=0.0)
    parser.add_argument("--all-examples", action="store_true", help="Include failed examples")
    parser.add_argument("--train-all-tokens", action="store_true", help="Train on all tokens, not just assistant")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation split ratio")
    
    args = parser.parse_args()
    
    config = CPTConfig(
        data_file=args.data_file,
        output_dir=args.output_dir,
        model_name=args.model,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        min_reward_threshold=args.min_reward,
        only_successful=not args.all_examples,
        train_on_assistant_only=not args.train_all_tokens,
        deepspeed=args.deepspeed,
        val_split=args.val_split,
    )
    
    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load and prepare data
    print(f"Loading data from: {config.data_file}")
    examples = load_trajectory_data(config.data_file, config)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_examples, val_examples = train_test_split(
        examples, test_size=config.val_split, random_state=42
    )
    
    print(f"Preparing datasets: {len(train_examples)} train, {len(val_examples)} val")
    train_dataset = prepare_dataset(train_examples, tokenizer, config)
    val_dataset = prepare_dataset(val_examples, tokenizer, config)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        deepspeed=config.deepspeed,
        evaluation_strategy="steps",
        eval_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"],
        run_name=f"qwen3-8b-cpt-rocm-agent",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving final model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

