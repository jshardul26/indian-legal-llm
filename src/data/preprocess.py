import logging
from pathlib import Path
from datasets import load_dataset, Dataset
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def format_example(example):
    q = (example.get("Instruction") or "").strip()
    a = (example.get("Response") or "").strip()
    if not q or not a:
        return None
    return {
        "text": f"User: {q}\n\nAssistant: {a}</s>",
        "question": q,
        "answer": a
    }

def preprocess_dataset(config):
    output_dir = Path(config["dataset"]["preprocessing_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading {config['dataset']['name']}...")
    dataset = load_dataset(config["dataset"]["name"], trust_remote_code=True)
    
    if isinstance(dataset, dict):
        dataset = dataset.get("train", dataset)
    
    logger.info(f"Original size: {len(dataset)}")
    
    processed = [format_example(ex) for ex in dataset]
    processed = [ex for ex in processed if ex is not None]
    logger.info(f"After filtering: {len(processed)}")
    
    max_samples = config["dataset"]["max_samples"]
    if max_samples and len(processed) > max_samples:
        processed = processed[:max_samples]
    
    train_idx, val_idx = train_test_split(
        range(len(processed)),
        test_size=config["dataset"]["validation_split"],
        random_state=config["training"]["seed"]
    )
    
    train_data = [processed[i] for i in train_idx]
    val_data = [processed[i] for i in val_idx]
    
    train_dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in train_data],
        "question": [ex["question"] for ex in train_data],
        "answer": [ex["answer"] for ex in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in val_data],
        "question": [ex["question"] for ex in val_data],
        "answer": [ex["answer"] for ex in val_data]
    })
    
    train_dataset.save_to_disk(str(output_dir / "train"))
    val_dataset.save_to_disk(str(output_dir / "validation"))
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset

if __name__ == "__main__":
    config = load_config()
    preprocess_dataset(config)