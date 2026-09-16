#!/usr/bin/env python3
import os
import logging
import yaml
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training
from src.train.lora_config import get_bnb_config, get_lora_config, get_training_args

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def prepare_datasets(config):
    """Load preprocessed train/val datasets."""
    proc_dir = Path(config["dataset"]["preprocessing_cache_dir"])
    train_ds = load_from_disk(str(proc_dir / "train"))
    val_ds = load_from_disk(str(proc_dir / "validation"))
    logger.info(f"Loaded train: {len(train_ds)}, val: {len(val_ds)}")
    return train_ds, val_ds

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples."""
    return tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )

def main():
    config = load_config()
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model_id"],
        quantization_config=get_bnb_config(),
        device_map=config["model"]["device_map"],
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Adding LoRA adapters...")
    peft_config = get_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    logger.info("Loading and tokenizing datasets...")
    train_ds, val_ds = prepare_datasets(config)
    
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, config["training"]["max_seq_length"]),
        batched=True,
        remove_columns=train_ds.column_names
    )

    val_ds = val_ds.map(
        lambda x: tokenize_function(x, tokenizer, config["training"]["max_seq_length"]),
        batched=True,
        remove_columns=val_ds.column_names
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    logger.info("Setting up trainer...")
    training_args = get_training_args(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving adapter...")
    model.save_pretrained(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])
    
    logger.info(f"Training complete! Adapter saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()