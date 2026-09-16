import json
import logging
import time
from pathlib import Path
import yaml
import torch
from src.inference.generate import load_model_and_tokenizer, load_lora_adapter, LegalQAGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "What is the Indian Penal Code?",
    "Explain Section 420 of IPC.",
    "What are the fundamental rights in Indian Constitution?",
    "Define bail under CrPC.",
    "What is the procedure for filing a civil suit?",
    "Explain the concept of mens rea in criminal law.",
    "What are the rules of evidence under Indian Evidence Act?",
    "Define tort and explain different types.",
    "What is the role of Supreme Court in India?",
    "Explain the process of constitutional amendment."
]


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate(config):
    """Evaluate base and fine-tuned models."""
    output_dir = Path(config["evaluation"]["eval_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model...")
    # A 1.5B model comfortably fits in fp16 (~3GB VRAM). 4-bit quantization
    # exists to save VRAM on much larger models — on something this small,
    # the bitsandbytes dequantization overhead per layer can make inference
    # SLOWER than plain fp16, especially on Windows where bnb's CUDA kernel
    # support is less mature. Default eval to fp16; only quantize if you're
    # actually VRAM-constrained.
    use_4bit = config["model"].get("eval_load_in_4bit", False)
    base_model, tokenizer = load_model_and_tokenizer(
        config["model"]["base_model_id"],
        load_in_4bit=use_4bit
    )
    logger.info(f"CUDA available: {torch.cuda.is_available()} | 4-bit: {use_4bit}")

    adapter_path = Path(config["training"]["output_dir"])
    has_adapter = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()

    results = {
        "base_model": config["model"]["base_model_id"],
        "questions": [],
        "has_fine_tuned_adapter": has_adapter
    }

    # Load the adapter ONCE onto the same model instance, then toggle it
    # on/off per generation call instead of treating base_model / ft_model
    # as if they were independent objects. (PeftModel.from_pretrained
    # mutates the model it's given, so the old two-object approach caused
    # "base" generations to silently pick up the adapter's weights too.)
    if has_adapter:
        logger.info(f"Loading LoRA adapter from {adapter_path}...")
        model = load_lora_adapter(base_model, str(adapter_path))
        base_gen = LegalQAGenerator(model, tokenizer, use_lora=False)
        ft_gen = LegalQAGenerator(model, tokenizer, use_lora=True)
    else:
        logger.warning("No fine-tuned adapter found. Evaluation will only use base model.")
        base_gen = LegalQAGenerator(base_model, tokenizer, use_lora=False)
        ft_gen = None

    # Eval-time generation settings: shorter + greedy so a sanity-check run
    # doesn't take as long as a production-quality generation would.
    eval_gen_kwargs = dict(max_new_tokens=128, do_sample=False)

    logger.info("Evaluating on test questions...")
    for i, question in enumerate(TEST_QUESTIONS):
        logger.info(f"[{i+1}/{len(TEST_QUESTIONS)}] {question}")

        t0 = time.time()
        base_answer = base_gen.generate(question, **eval_gen_kwargs)
        dt = time.time() - t0
        approx_tokens = len(tokenizer(base_answer)["input_ids"])
        logger.info(f"  base model:       {dt:.1f}s (~{approx_tokens/dt:.1f} tok/s)")

        result = {
            "question": question,
            "base_model_answer": base_answer
        }

        if ft_gen:
            t0 = time.time()
            ft_answer = ft_gen.generate(question, **eval_gen_kwargs)
            logger.info(f"  fine-tuned model: {time.time() - t0:.1f}s")
            result["fine_tuned_answer"] = ft_answer

        results["questions"].append(result)

    output_file = output_dir / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete! Results saved to {output_file}")

    if ft_gen:
        logger.info("\n=== SAMPLE COMPARISON ===")
        q = results["questions"][0]
        logger.info(f"Q: {q['question']}")
        logger.info(f"Base: {q['base_model_answer'][:200]}")
        logger.info(f"FT: {q['fine_tuned_answer'][:200]}")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)