import os
import streamlit as st
import logging
import yaml
import torch
from pathlib import Path
from src.inference.generate import load_model_and_tokenizer, load_lora_adapter, LegalQAGenerator

logger = logging.getLogger(__name__)

# When the local adapter folder isn't present (e.g. on a deployed server
# that only has the repo's source code, not your training outputs), fall
# back to the adapter you pushed to the Hub. Override via the ADAPTER_REPO
# env var (set this as a Streamlit Cloud "secret") if you want to point at
# a different repo without editing code.
HUB_ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "jshardul26/indian-legal-qwen2.5-1.5b-lora")


@st.cache_resource
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def resolve_adapter_source(local_adapter_path: str):
    """
    Returns (source, is_local) where `source` is either the local adapter
    path (if it actually exists there) or the Hub repo id fallback, and
    `is_local` tells the UI which one is in play.
    """
    local_p = Path(local_adapter_path)
    if local_p.exists() and (local_p / "adapter_config.json").exists():
        return str(local_p), True
    if HUB_ADAPTER_REPO:
        return HUB_ADAPTER_REPO, False
    return None, False


@st.cache_resource
def initialize_model(model_id: str, use_lora=False, adapter_path=None):
    """Initialize model with caching."""
    config = load_config()

    # 4-bit quantization needs a CUDA GPU with working bitsandbytes kernels.
    # A deployed CPU-only server (Streamlit Cloud free tier, etc.) will
    # error or silently misbehave if we blindly trust config.yaml here -
    # and for a 1.5B model there's no real memory benefit to it anyway.
    want_4bit = config["model"].get("load_in_4bit", False)
    load_in_4bit = want_4bit and torch.cuda.is_available()
    if want_4bit and not load_in_4bit:
        logger.warning("load_in_4bit=True in config, but no CUDA GPU detected - loading in fp32/fp16 instead.")

    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device_map="auto",
        load_in_4bit=load_in_4bit,
    )

    resolved_source, is_local = (None, False)
    if use_lora and adapter_path:
        resolved_source, is_local = resolve_adapter_source(adapter_path)

    if resolved_source:
        model = load_lora_adapter(model, resolved_source)
        logger.info(f"Loaded LoRA adapter from {'local path' if is_local else 'Hugging Face Hub'}: {resolved_source}")
    elif use_lora:
        logger.warning("No local adapter found and no ADAPTER_REPO fallback set - using base model only.")

    generator = LegalQAGenerator(model, tokenizer, use_lora=bool(resolved_source))
    return generator, model, tokenizer


def check_adapter_exists(adapter_path: str) -> bool:
    """
    True if EITHER a local adapter is present OR a Hub fallback repo is
    configured. This drives the sidebar's "fine-tuned adapter loaded"
    indicator, so it should reflect what will actually be loaded, not just
    whether a local training output happens to exist.
    """
    source, _ = resolve_adapter_source(adapter_path)
    return source is not None


def get_model_info(config):
    """Get model information for display."""
    source, is_local = resolve_adapter_source(config["training"]["output_dir"])
    return {
        "base_model": config["model"]["base_model_id"],
        "adapter_path": source or config["training"]["output_dir"],
        "adapter_exists": source is not None,
        "adapter_is_local": is_local,
        "mode": "base+lora" if source else "base_only",
    }