"""
Push the fine-tuned LoRA adapter to the Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    hf auth login          # or set HF_TOKEN env var
        -> paste a token with WRITE access from
           https://huggingface.co/settings/tokens

Usage:
    python deploy/push_to_hub.py --adapter-dir outputs/adapters/legal-qlora-v1 --repo-id YOUR_USERNAME/indian-legal-qwen2.5-1.5b-lora --base-model Qwen/Qwen2.5-1.5B-Instruct

This uploads ONLY the LoRA adapter (a few MB), not the full base model -
anyone using it downloads Qwen2.5-1.5B-Instruct from its own repo and
applies your adapter on top, exactly like your evaluate.py does locally.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
tags:
  - lora
  - peft
  - legal
  - indian-law
  - qwen2.5
language:
  - en
---

# {repo_name}

LoRA adapter fine-tuned on top of [{base_model}](https://huggingface.co/{base_model})
for Indian legal question-answering (IPC, CrPC, Constitution, Evidence Act, torts, etc.).

## Disclaimer

This model is a **personal / educational project**, not a substitute for professional
legal advice. It can produce **factually incorrect or hallucinated** statements,
including invented section numbers, dates, or conflated statutes. Always verify
any legal information against primary sources (bare acts, official gazettes) or
consult a qualified lawyer before relying on it.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{base_model}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

messages = [{{"role": "user", "content": "Explain Section 420 of IPC."}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

LoRA fine-tuning on a curated Indian legal QA dataset. See the training repo for
data preparation and hyperparameters.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="Local path to the LoRA adapter (contains adapter_config.json)")
    parser.add_argument("--repo-id", required=True, help="e.g. yourusername/indian-legal-qwen2.5-1.5b-lora")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--private", action="store_true", help="Create a private repo instead of public")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json found in {adapter_dir} - check the path.")

    api = HfApi()

    print(f"Creating repo {args.repo_id} (private={args.private})...")
    create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")

    readme_path = adapter_dir / "README.md"
    readme_path.write_text(
        MODEL_CARD_TEMPLATE.format(
            base_model=args.base_model,
            repo_id=args.repo_id,
            repo_name=args.repo_id.split("/")[-1],
        ),
        encoding="utf-8",
    )

    print(f"Uploading {adapter_dir} -> {args.repo_id} ...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\nDone. View it at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
