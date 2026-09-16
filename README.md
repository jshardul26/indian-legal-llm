# Indian Legal QA — QLoRA Fine-Tuning Portfolio Project

A small-scale, technically rigorous Indian legal question-answering system built to demonstrate **instruction fine-tuning with QLoRA/PEFT** on consumer hardware (6GB VRAM).

> ⚠️ **Educational Disclaimer**: This application is for educational and demonstration purposes only. It does not provide legal advice and should not be relied upon for legal decisions. Always consult a qualified legal professional for actual legal matters.

**Project status: code-complete, fine-tuning execution pending.** No training has been run yet — see [Current Status](#current-status).

---

## Project Overview

The app lets a user ask a question about Indian law (IPC, CrPC, Constitution, general legal procedure) through a Streamlit chat UI and get an answer from a small instruction-tuned LLM. A LoRA adapter, trained on an Indian-law instruction dataset, is layered on top of the base model to specialize it for this domain — without touching the base weights.

This exists to demonstrate real, working understanding of: Hugging Face `transformers`/`datasets`/`peft`, 4-bit quantization (QLoRA), LoRA adapter training, dataset preprocessing, evaluation methodology, and Streamlit deployment — not to be a production legal tool.

## Architecture

```text
User
 │
 ▼
Streamlit UI  (app/streamlit_app.py)
 │
 ▼
model_loader.py  (loads config, caches model via st.cache_resource)
 │
 ▼
src/inference/generate.py  (shared generation function)
 │
 ├───────────────┐
 ▼               ▼
Base Model    LoRA Adapter (if trained)
 │               │
 └───────┬───────┘
         ▼
    Generated Answer
```

Training pipeline:

```text
viber1/indian-law-dataset (HF Hub)
        │
        ▼
download_dataset.py  → data/raw/
        │
        ▼
preprocess.py  → clean, filter, format, split → data/processed/
        │
        ▼
train_lora.py: tokenize → load base model in 4-bit (NF4) → prepare_model_for_kbit_training
        │
        ▼
LoRA adapter (peft) attached to attention + MLP projections
        │
        ▼
Trainer.train() → outputs/adapters/legal-qlora-v1/
        │
        ▼
evaluate.py → outputs/evaluations/evaluation_results.json
        │
        ▼
Streamlit inference (base+adapter if present, else base only)
```

## Model Selection

**Selected: `Qwen/Qwen2.5-1.5B-Instruct`**

| Criterion | Notes |
|---|---|
| Size | 1.5B params — realistic for 6GB VRAM in 4-bit |
| Chat template | Ships a proper `tokenizer.apply_chat_template()` |
| Architecture | Standard Llama-style attention/MLP (`q/k/v/o_proj`, `gate/up/down_proj`) — verified compatible LoRA target names, not assumed |
| Licensing | Apache 2.0 |
| Instruction following | Strong for its size relative to alternatives (Phi-3-mini, TinyLlama, Gemma-2-2B) at this VRAM budget |

**Alternatives considered:** TinyLlama-1.1B (weaker instruction following), Gemma-2-2B (heavier tokenizer/KV overhead, less headroom on 6GB), Phi-3-mini-3.8B (fits but leaves little margin for training activations alongside optimizer state on 6GB).

## Dataset Selection

**Selected: [`viber1/indian-law-dataset`](https://huggingface.co/datasets/viber1/indian-law-dataset)**

- **Rows:** 24,607 (single `train` split)
- **Columns:** `Instruction`, `Response` (instruction/answer pairs on Indian law)
- **Format:** JSON → auto-converted Parquet, loadable directly via `datasets.load_dataset`
- A `max_samples: 12000` cap is applied in `config.yaml` to keep the run inside the ~2–3 hour training budget; set it to `null` to use the full dataset (longer run).

**License note:** verify the license shown on the dataset's Hugging Face page before any redistribution or commercial use — this README does not assert a license claim that hasn't been independently confirmed.

**Alternatives considered:** `nisaar/Constitution_Of_India_Instruction_Set` (narrower scope, Constitution-only), `ninadn/indian-legal` (raw case text, not instruction-formatted — would need extra QA-pair construction).

## Fine-Tuning Configuration

| Parameter | Value | Why |
|---|---|---|
| Quantization | 4-bit NF4, double quant, fp16 compute | Fits base model in ~1.1–1.3GB, standard QLoRA recipe |
| LoRA rank (r) | 16 | Enough capacity for domain adaptation without bloating adapter size |
| LoRA alpha | 32 | 2× rank, standard scaling ratio |
| LoRA dropout | 0.05 | Light regularization for a small dataset |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Verified against Qwen2.5's actual module names (not assumed from Llama) |
| Per-device batch size | 2 | Fits 6GB alongside optimizer state + activations |
| Gradient accumulation | 4 | Effective batch size 8 without exceeding VRAM |
| Learning rate | 2e-4 | Standard for LoRA (much higher than full fine-tune LR) |
| Epochs | 3 | Small dataset subset — avoids overfitting beyond this |
| Max sequence length | 512 | Covers the large majority of Q&A pairs in this dataset |
| Optimizer | `paged_adamw_8bit` | Paged optimizer avoids VRAM spikes on 6GB cards |

All of the above live in `config.yaml` — no values are hardcoded in the training script.

## Hardware

- **Primary target:** NVIDIA RTX 4050 Laptop GPU, 6GB VRAM
- **Fallback:** Google Colab (T4, 16GB)
- **Estimated VRAM usage:** ~1.2GB (4-bit base) + <0.3GB (LoRA adapter) + 2–3GB (activations/optimizer) ≈ 4.5GB, leaving headroom on a 6GB card
- **Expected training time:** ~2–2.5 hours on RTX 4050 for the 12,000-example subset at 3 epochs

## Current Status

```text
[x] Frontend (Streamlit)         — complete
[x] Model loading / inference    — complete
[x] Dataset download/preprocess  — complete
[x] LoRA/QLoRA config            — complete
[x] Fine-tuning script           — complete, NOT yet executed
[x] Evaluation pipeline          — complete, reports "no adapter" until trained
[x] Tests                        — complete (skip gracefully without GPU)
[ ] Fine-tuning execution        — PENDING (run manually, see below)
```

The Streamlit app runs fine right now on the base model alone. Once `train_lora.py` is run, the app auto-detects the adapter at `outputs/adapters/legal-qlora-v1/` and switches modes — no code changes needed.

No training loss, perplexity, or before/after comparison numbers are reported anywhere in this repo, because none have been generated yet.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your HF_TOKEN
```

**Note on bitsandbytes:** 4-bit quantization via `bitsandbytes` is Linux/WSL-first. On native Windows, either use WSL2 or run the training step on Google Colab instead.

## Running Dataset Preparation

```bash
python -m src.data.download_dataset
python -m src.data.preprocess
```

## Running the Application

```bash
streamlit run app/streamlit_app.py
```

Works immediately on the base model; the sidebar's "Fine-Tuning Details" panel will show the adapter as not yet generated.

## Running Fine-Tuning

```bash
python -m src.train.train_lora
```

Takes ~2–2.5 hours on an RTX 4050. Saves the adapter to `outputs/adapters/legal-qlora-v1/`.

## Running Evaluation

```bash
python -m src.train.evaluate
```

Runs 10 fixed Indian-law test questions through the base model, and through base+adapter if the adapter exists. Results are saved to `outputs/evaluations/evaluation_results.json`.

## Deployment

**Streamlit Community Cloud:** A 1.5B model in 4-bit (~1.3GB) is within Streamlit Cloud's free-tier resource limits, making direct deployment realistic — push this repo, point Streamlit Cloud at `app/streamlit_app.py`, and set `HF_TOKEN` as a secret.

**Hugging Face Spaces (fallback):** Use the Streamlit SDK Space type with a T4-small hardware tier if CPU inference proves too slow; the same `app/streamlit_app.py` runs unmodified.

## Limitations

- No fine-tuning has been executed yet — all current answers come from the untuned base model.
- The dataset is instruction-pair QA, not grounded in citations to specific sections/case law, so hallucinated section numbers are possible even after fine-tuning.
- No retrieval/RAG grounding — answers are generative only.
- Evaluation is qualitative (10 fixed questions) plus loss/perplexity — no legal-accuracy benchmark exists for this domain at this scale.
- Not reviewed by a legal professional; not fit for real legal use.

## Future Improvements

- Larger/cleaner dataset combining multiple Indian-law sources
- Retrieval-augmented generation with citation grounding to actual statute text
- A proper legal-QA accuracy benchmark instead of ad hoc qualitative comparison
- Hallucination/fabricated-citation detection
- A stronger base model once more VRAM is available

## Project Structure

```text
indian-legal-llm/
├── app/                  # Streamlit UI + model loader
├── src/
│   ├── data/             # download + preprocess
│   ├── train/            # LoRA config, training script, evaluation
│   ├── inference/        # shared generation function
│   └── utils/            # logging
├── notebooks/            # fine-tuning walkthrough notebook
├── data/{raw,processed}/ # gitignored, populated by scripts
├── outputs/{adapters,evaluations,logs}/  # gitignored, populated by scripts
├── tests/
├── config.yaml           # single source of truth for all settings
└── requirements.txt
```

## Resume-Ready Description

*Built an end-to-end Indian legal QA system with a Streamlit interface, implementing a full QLoRA fine-tuning pipeline (4-bit NF4 quantization, LoRA adapters via PEFT, gradient-accumulated training) for Qwen2.5-1.5B-Instruct on a 24.6k-example Indian legal instruction dataset, designed to run on a 6GB consumer GPU. Includes a config-driven architecture, base/adapter model switching, and a qualitative + quantitative evaluation pipeline comparing base vs. fine-tuned outputs.*
