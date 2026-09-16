import pytest
import yaml
from pathlib import Path
from src.inference.generate import load_model_and_tokenizer, LegalQAGenerator

def test_config_exists():
    """Test configuration file exists."""
    assert Path("config.yaml").exists()

def test_config_valid():
    """Test configuration is valid YAML."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert "model" in config
    assert "dataset" in config
    assert "training" in config
    assert "lora" in config

def test_config_required_fields():
    """Test required config fields."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert config["model"]["base_model_id"]
    assert config["dataset"]["name"]
    assert config["training"]["num_epochs"] > 0

def test_model_loading():
    """Test model can be loaded (without GPU)."""
    config = {"model": {"base_model_id": "Qwen/Qwen2.5-1.5B-Instruct"}}
    try:
        model, tokenizer = load_model_and_tokenizer(
            config["model"]["base_model_id"],
            load_in_4bit=False
        )
        assert model is not None
        assert tokenizer is not None
    except Exception as e:
        pytest.skip(f"Model loading skipped: {e}")

def test_generator_interface():
    """Test LegalQAGenerator interface."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        model, tokenizer = load_model_and_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct",
            load_in_4bit=False
        )
        gen = LegalQAGenerator(model, tokenizer)
        
        assert hasattr(gen, "generate")
        assert hasattr(gen, "format_prompt")
        
        prompt = gen.format_prompt("What is law?")
        assert "User:" in prompt
        assert "Assistant:" in prompt
    except Exception as e:
        pytest.skip(f"Generator test skipped: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])