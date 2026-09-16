# README 2 — Setup, Remaining Steps & Troubleshooting Guide

This is the "what's left / how do I actually run this" guide. `README.md` explains the project; this file is a checklist + troubleshooting manual to get from a fresh clone to a trained adapter.

---

## 1. What Is Actually Left To Do

| Item | Status |
|---|---|
| All `.py` source files (app, src/data, src/train, src/inference, src/utils) | ✅ Written, functional |
| `config.yaml`, `requirements.txt`, `.gitignore`, `.env.example`, `README.md` | ✅ Written |
| Dataset downloaded to disk | ❌ Not done — you run this |
| Dataset preprocessed/split | ❌ Not done — you run this |
| LoRA fine-tuning executed | ❌ Not done — you run this |
| Evaluation run | ❌ Not done — depends on above |
| `notebooks/finetune_walkthrough.ipynb` | ❌ **Empty file, 0 cells** — needs to be built |
| One known code fix before first CUDA run | ⚠️ See §4 below |

Everything in the table's first block is done. The rest is on you (or the fine-tuning execution itself, which was intentionally left for last per the original brief).

---

## 2. Step-by-Step: From Zero To Trained Adapter

### Step 0 — Environment
```bash
cd indian-legal-llm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If you're on native Windows and plan to actually run fine-tuning (not just the Streamlit app), switch to **WSL2** first — `bitsandbytes` 4-bit quantization is unreliable on native Windows. CPU-only Streamlit browsing works fine on native Windows.

### Step 1 — Hugging Face token
1. Create a token at https://huggingface.co/settings/tokens (read access is enough).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your token:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
Neither the base model nor the dataset is gated, so this is technically optional — but without it you'll hit Hub rate limits on repeated downloads.

### Step 2 — Download the dataset
```bash
python -m src.data.download_dataset
```
**What this does:** pulls `viber1/indian-law-dataset` (24.6k rows) from the Hub and saves it to `data/raw/train/` using `Dataset.save_to_disk`.
**Where it lands:** `data/raw/train/` (a folder with `dataset_info.json`, `.arrow` files, etc. — not a single file, this is normal for `datasets`).
**You don't need to do anything manually here** — no manual placement, no manual file moving. The script handles the folder.

### Step 3 — Preprocess
```bash
python -m src.data.preprocess
```
**What this does:** loads the dataset fresh from the Hub (note: `preprocess.py` currently reloads via `load_dataset` rather than reading `data/raw/`, so Step 2 and Step 3 both hit the network — see §5 optional improvement), formats each `Instruction`/`Response` pair into the chat-style text format, filters empty rows, splits train/validation, and saves to `data/processed/train/` and `data/processed/validation/`.
**Verify it worked:**
```bash
python -c "from datasets import load_from_disk; d=load_from_disk('data/processed/train'); print(len(d)); print(d[0])"
```

### Step 4 — Sanity-check the app before training
```bash
streamlit run app/streamlit_app.py
```
It should load the base model only (no adapter yet) and answer questions. If this works, your environment is correctly set up before you spend 2+ hours training.

### Step 5 — Fine-tune
```bash
python -m src.train.train_lora
```
Expect ~2–2.5 hours on an RTX 4050. Watch console output for the loss decreasing every `logging_steps` (20 steps). Adapter is saved to `outputs/adapters/legal-qlora-v1/`.

### Step 6 — Evaluate
```bash
python -m src.train.evaluate
```
Produces `outputs/evaluations/evaluation_results.json` with base-vs-fine-tuned answers on 10 fixed questions.

### Step 7 — Re-run the app
```bash
streamlit run app/streamlit_app.py
```
It will now auto-detect the adapter and switch to "base+lora" mode — no code or config changes needed for this switch.

---

## 3. Commands Cheat Sheet

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit in your HF_TOKEN

# data
python -m src.data.download_dataset
python -m src.data.preprocess

# app (works with or without a trained adapter)
streamlit run app/streamlit_app.py

# training + eval
python -m src.train.train_lora
python -m src.train.evaluate

# tests
pytest tests/ -v
```

---

## 4. Code Changes Needed Before Your First Real (CUDA) Run

### 4.1 `flash_attention_2` would have crashed on a fresh install — already fixed in this copy
`src/inference/generate.py`'s `load_model_and_tokenizer()` originally requested `attn_implementation="flash_attention_2"` whenever CUDA is available. `flash-attn` is **not** in `requirements.txt` (it's a painful, slow compile on Windows/WSL and unnecessary for a 1.5B model on a laptop GPU), so this would have thrown an import error on your RTX 4050.

**This has already been changed to:**
```python
attn_implementation="sdpa" if torch.cuda.is_available() else None
```
in the copy of the project you were given. `sdpa` ships with `torch` itself — no extra install. If you're working from an older copy of this repo, apply that one-line change manually before training.

### 4.2 Nothing else requires manual code changes
Every other file (`train_lora.py`, `preprocess.py`, `lora_config.py`, `evaluate.py`, `streamlit_app.py`, `model_loader.py`) reads entirely from `config.yaml` and is ready to run as-is.

---

## 5. `notebooks/finetune_walkthrough.ipynb` — What Still Needs To Be Built

This file currently has **zero cells** (`{"cells": [], ...}`). It's the one deliverable from the original spec that isn't done. Below is the exact list of cells/sections to generate — hand this list to any AI (or write it yourself) to produce the notebook. Each bullet = one markdown cell (title) + one code cell beneath it, reusing the project's existing functions rather than reimplementing logic.

1. **Title + disclaimer cell** — markdown, restates the "educational only" disclaimer from the README.
2. **Environment setup cell** — code: `pip install -r ../requirements.txt` (or `%pip install`), load `.env` with `python-dotenv`.
3. **Load config cell** — code: import and call `load_config()` from `src.data.preprocess` (or duplicate the two-line YAML load).
4. **Dataset loading cell** — code: call `download_dataset()` from `src.data.download_dataset`; print row count and a couple of raw examples.
5. **Dataset inspection cell** — code: print `column_names`, a value-length histogram or `.describe()`-style stats on `Instruction`/`Response` lengths (this is genuinely new code — small, doesn't exist elsewhere yet).
6. **Preprocessing cell** — code: call `preprocess_dataset(config)` from `src.data.preprocess`; print train/val sizes.
7. **Prompt formatting demo cell** — code: import `format_example` from `src.data.preprocess`, run it on 2–3 raw rows, print the resulting `text` field.
8. **Tokenization demo cell** — code: load the tokenizer via `load_model_and_tokenizer(..., load_in_4bit=False)` from `src.inference.generate`, tokenize one formatted example, print token count and decoded round-trip.
9. **4-bit base model loading cell** — code: call `load_model_and_tokenizer(model_id, load_in_4bit=True)`; print `model.get_memory_footprint()` to show actual VRAM usage.
10. **LoRA config cell** — code: import `get_lora_config()` from `src.train.lora_config`, apply with `get_peft_model`, call `model.print_trainable_parameters()`.
11. **Training cell** — markdown note ("this reproduces `python -m src.train.train_lora`; run the script directly for the full run — this cell is for demonstration on a tiny subset") + code: optionally run `Trainer.train()` on a small `.select(range(50))` subset for a fast demo, rather than the full 2-hour run inside the notebook.
12. **Adapter saving cell** — code: `model.save_pretrained(...)`.
13. **Evaluation cell** — code: call `evaluate(config)` from `src.train.evaluate`, or load `outputs/evaluations/evaluation_results.json` if it already exists and display it as a table.
14. **Before/after comparison cell** — code: pretty-print a few `question` / `base_model_answer` / `fine_tuned_answer` triples from the evaluation JSON using `pandas.DataFrame`.

No new production classes are needed for the notebook — steps 5 and 11 are the only spots with genuinely new (small) code; everything else is calling functions that already exist in `src/`.

---

## 6. Errors You're Likely To Hit, And How To Fix Them

| Error | Cause | Fix |
|---|---|---|
| `ImportError: flash_attn ...` or similar on model load | See §4.1 | Change `attn_implementation` to `"sdpa"` |
| `CUDA out of memory` during training | Batch size / seq length too high for 6GB | Lower `training.per_device_train_batch_size` to 1 in `config.yaml`, or `max_seq_length` to 256, and raise `gradient_accumulation_steps` to compensate |
| `bitsandbytes` fails to import / "CUDA setup failed" | Native Windows, or CUDA/bitsandbytes version mismatch | Use WSL2, or reinstall a bitsandbytes wheel matching your CUDA version |
| `OSError: ... does not appear to have a file named pytorch_model.bin` or similar on first download | Network interruption mid-download, or no `HF_TOKEN` under rate limiting | Re-run the command; set `HF_TOKEN` in `.env` |
| `KeyError: 'Instruction'` in `preprocess.py` | Only relevant if you swap in a different dataset — the code expects `Instruction`/`Response` columns specifically | If you change `dataset.name` in `config.yaml`, update the column names in `format_example()` in `src/data/preprocess.py` to match the new dataset's schema |
| Streamlit shows "Base model only" forever after training | Adapter didn't save to the exact path `training.output_dir` points to, or training crashed before `save_pretrained` | Check `outputs/adapters/legal-qlora-v1/` for an `adapter_config.json`; if missing, re-run training and watch for errors near the end of the log |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` or similar during training | Tokenizer padding/truncation mismatch, usually from swapping models without checking `max_seq_length` compatibility | Confirm `max_seq_length` in `config.yaml` is ≤ the model's context window |
| Very slow / no progress during training with no GPU errors | Accidentally running on CPU (check `torch.cuda.is_available()`) | Verify your PyTorch build includes CUDA support: `python -c "import torch; print(torch.cuda.is_available())"` — if `False`, reinstall `torch` with the correct CUDA index URL from pytorch.org |
| `pytest` model-loading tests are skipped, not passing | Expected — `tests/test_generate.py` intentionally skips (not fails) when no GPU/model access is available, so CI doesn't require a GPU | No action needed; this is by design |

---

## 7. Things Worth Knowing About For Later (Not Bugs, Just Worth Flagging)

- **`preprocess.py` re-downloads from the Hub instead of reading `data/raw/`.** Functionally fine (it works), just means Step 2 and Step 3 both touch the network. Not worth fixing unless you're offline a lot — if you want it fixed, point `preprocess.py`'s `load_dataset(...)` call to `load_from_disk(raw_dir / "train")` instead.
- **Padded tokens contribute to training loss.** `DataCollatorForLanguageModeling(mlm=False)` copies `input_ids` to `labels` without masking padding. Since `pad_token == eos_token` here, this mostly just reinforces the model learning to stop — a minor inefficiency, not a correctness bug, and not worth touching for a portfolio-scale run.
- **`max_samples: 12000`** in `config.yaml` controls your training time directly. Drop it to ~4000–6000 if 2–2.5 hours is too long for your first test run, or set to `null` to use the full 24.6k rows if you have more time.

---

## 8. Final Pre-Flight Checklist

- [ ] `.env` created with `HF_TOKEN`
- [ ] `attn_implementation` changed to `"sdpa"` in `src/inference/generate.py`
- [ ] `python -m src.data.download_dataset` ran without error
- [ ] `python -m src.data.preprocess` ran without error, `data/processed/train` and `data/processed/validation` exist
- [ ] `streamlit run app/streamlit_app.py` works on base model before starting training
- [ ] `nvidia-smi` confirms GPU is free of other heavy processes before starting `train_lora.py`
- [ ] `python -m src.train.train_lora` completes, `outputs/adapters/legal-qlora-v1/adapter_config.json` exists
- [ ] `python -m src.train.evaluate` produces `outputs/evaluations/evaluation_results.json`