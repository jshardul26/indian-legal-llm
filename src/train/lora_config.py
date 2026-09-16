from transformers import BitsAndBytesConfig
from peft import LoraConfig

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

def get_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=None
    )

def get_training_args(config):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        save_steps=config["training"]["save_steps"],
        eval_steps=config["training"]["eval_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="paged_adamw_8bit",
        seed=config["training"]["seed"],
        bf16=False,
        fp16=True,
        remove_unused_columns=False,
        report_to=[]
    )