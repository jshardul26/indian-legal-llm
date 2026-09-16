"""
Gradio Space: Indian Legal QA (Qwen2.5-1.5B-Instruct + LoRA)

Deploy: push this file, requirements.txt, and README.md as a new HF Space
(SDK: Gradio). Set ADAPTER_REPO below to your pushed adapter repo id.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "jshardul26/indian-legal-qwen2.5-1.5b-lora")

DISCLAIMER = (
    "This is a personal/educational project, not legal advice. "
    "The model can produce factually incorrect or fabricated information "
    "(wrong section numbers, dates, or conflated statutes) with confident "
    "phrasing. Always verify against a bare act or a qualified lawyer "
    "before relying on anything it says."
)

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

print(f"Loading LoRA adapter from {ADAPTER_REPO}...")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()

DEVICE = next(model.parameters()).device


def _eos_token_ids():
    eos_ids = getattr(model.generation_config, "eos_token_id", None)
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in eos_ids:
        eos_ids = list(eos_ids) + [im_end_id]
    return eos_ids


EOS_IDS = _eos_token_ids()


def answer(question, history):
    if not question or not question.strip():
        return "Please enter a question."

    messages = [{"role": "user", "content": question.strip()}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded.get("attention_mask")
    attention_mask = attention_mask.to(DEVICE) if attention_mask is not None else torch.ones_like(input_ids)
    prompt_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=EOS_IDS,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    new_tokens = outputs[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


with gr.Blocks(title="Indian Legal QA (Fine-tuned Qwen2.5-1.5B)") as demo:
    gr.Markdown("# Indian Legal QA")
    gr.Markdown(DISCLAIMER)
    chat = gr.ChatInterface(
        fn=answer,
        examples=[
            "Explain Section 420 of IPC.",
            "Define bail under CrPC.",
            "What are the fundamental rights in the Indian Constitution?",
            "Define tort and explain different types.",
        ],
        type="messages",
    )
    gr.Markdown(
        "---\nBase model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) "
        f"+ LoRA adapter: [{ADAPTER_REPO}](https://huggingface.co/{ADAPTER_REPO})"
    )

if __name__ == "__main__":
    demo.launch()
