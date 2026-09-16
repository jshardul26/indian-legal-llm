"""Download the raw Indian legal dataset and cache it to disk.

Kept separate from preprocess.py so raw data can be inspected/re-used
without re-hitting the Hugging Face Hub on every preprocessing run.
"""
import logging
from pathlib import Path

import yaml
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(config: dict):
    """Download the configured dataset from the Hub and save it to raw_cache_dir."""
    raw_dir = Path(config["dataset"]["raw_cache_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    name = config["dataset"]["name"]
    logger.info(f"Downloading dataset: {name}")
    dataset = load_dataset(name, trust_remote_code=True)

    if isinstance(dataset, dict):
        train_split = dataset.get("train", next(iter(dataset.values())))
    else:
        train_split = dataset

    logger.info(f"Downloaded {len(train_split)} rows")
    logger.info(f"Columns: {train_split.column_names}")

    train_split.save_to_disk(str(raw_dir / "train"))
    logger.info(f"Saved raw dataset to {raw_dir / 'train'}")
    return train_split


if __name__ == "__main__":
    cfg = load_config()
    download_dataset(cfg)
