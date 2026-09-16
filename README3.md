# Deployment Guide — Indian Legal QA (Qwen2.5-1.5B + LoRA)

> Note: named this file exactly as requested ("Redmi 3.md"). If your keyboard
> autocorrected "Readme3" to "Redmi 3" and you'd rather it be `README3.md`,
> just rename it — nothing inside depends on the filename.

This covers three things, in order:
1. Pushing your LoRA adapter to the Hugging Face Hub (you've likely already done this)
2. Deploying via **Hugging Face Spaces (Gradio)**
3. Deploying via **Streamlit Community Cloud**

Pick one, or do both — they can both point at the same adapter on the Hub.

---

## 0. Prerequisites

- A Hugging Face account with a **Write** access token (https://huggingface.co/settings/tokens)
- A GitHub account (needed for Streamlit Cloud, optional for Spaces)
- Git installed locally
- Your adapter already pushed to the Hub (skip to Step 2 if not)

---

## 1. Push the adapter to the Hub (if not done yet)

```powershell
pip install huggingface_hub
hf auth login
```

Paste your token when prompted (input is hidden — that's normal, not frozen).

```powershell
python deploy/push_to_hub.py --adapter-dir outputs/adapters/legal-qlora-v1 --repo-id YOUR_USERNAME/indian-legal-qwen2.5-1.5b-lora
```

Verify it worked by visiting `https://huggingface.co/YOUR_USERNAME/indian-legal-qwen2.5-1.5b-lora`.

**Before moving on**, check the adapter folder doesn't contain leftover training checkpoints (optimizer states, `checkpoint-*` folders) that bloat the repo:

```powershell
Get-ChildItem outputs\adapters\legal-qlora-v1 -Recurse
```

If you see `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, or `checkpoint-*/` subfolders, delete them and re-run the push command — it should only be `adapter_model.safetensors`, `adapter_config.json`, and tokenizer files (a few MB total, not hundreds).

---

## 2. Option A — Deploy via Hugging Face Spaces (Gradio)

### 2.1 Update the adapter repo id in the app

Open `deploy/space/app.py` and confirm this line points at your actual repo:

```python
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "YOUR_USERNAME/indian-legal-qwen2.5-1.5b-lora")
```

### 2.2 Create the Space

Go to https://huggingface.co/new-space and fill in:
- **Space name**: e.g. `indian-legal-qa`
- **SDK**: Gradio
- **Hardware**: CPU basic (free) to start

### 2.3 Push your files to the Space

Spaces are just git repos. From your project folder:

```powershell
git clone https://huggingface.co/spaces/YOUR_USERNAME/indian-legal-qa
Copy-Item deploy\space\app.py indian-legal-qa\app.py
Copy-Item deploy\space\requirements.txt indian-legal-qa\requirements.txt
Copy-Item deploy\space\README.md indian-legal-qa\README.md
cd indian-legal-qa
git add .
git commit -m "Initial deployment"
git push
```

The Space will automatically build and start — watch progress under the **"Logs"** tab on the Space's page. First build typically takes 3-8 minutes.

---

## 3. Option B — Deploy via Streamlit Community Cloud

### 3.1 Point the app at the Hub adapter, not your local folder

Open `streamlit_app.py` and change:

```python
USE_HUB_ADAPTER = True   # was False
```

This matters — the deployed server won't have your local `outputs/adapters/...` folder, only what's on the Hub.

### 3.2 Push your project to GitHub

```powershell
git init
git add streamlit_app.py requirements.txt
git commit -m "Add Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/indian-legal-llm.git
git push -u origin main
```

Make sure `requirements.txt` (project root, for Streamlit Cloud specifically) contains:

```
streamlit
torch
transformers>=4.45.0
peft>=0.13.0
accelerate
```

### 3.3 Deploy

Go to https://share.streamlit.io → **New app** → select your repo, branch `main`, main file path `streamlit_app.py` → **Deploy**.

Free tier is CPU-only. First load will be slow (model download + no GPU) — expect the very first response to take a minute or more.

---

## 4. Problems you're likely to hit (and fixes)

### "CUDA out of memory" during local testing
Your GPU doesn't have enough free VRAM (something else is using it, or the model + adapter + generation cache exceeds available memory). Fixes:
- Close other GPU processes (`nvidia-smi` to check what's using VRAM)
- Reduce `max_new_tokens`
- Fall back to `dtype=torch.float32` on CPU as a last resort (slow but works)

### App works locally but fails on the deployed platform with a missing-file error
Almost always means `USE_HUB_ADAPTER` (Streamlit) is still `False`, or `ADAPTER_REPO` (Gradio) still points at a local path instead of your Hub repo id. The deployed server has no access to your local `outputs/` folder.

### "401 Unauthorized" or "Repository not found" pulling your adapter on the deployed platform
If you made the adapter repo **private**, the deployed app needs its own HF token to access it:
- **Spaces**: add `HF_TOKEN` under Space **Settings → Repository secrets**
- **Streamlit Cloud**: add `HF_TOKEN` under app **Settings → Secrets**, then read it in code with `os.environ["HF_TOKEN"]` and pass `token=...` to `from_pretrained()`
- Simplest fix: make the adapter repo public instead, if there's no reason to keep it private.

### Build fails on Streamlit Cloud / Space with a dependency resolution error
`torch` without a specified version can pull an incompatible CUDA build for a CPU-only server. Pin a CPU-friendly version if this happens:
```
torch --index-url https://download.pytorch.org/whl/cpu
```
in `requirements.txt`, or just `torch` and let the platform resolve it — CPU-only build environments usually handle this automatically, but it's the first thing to check if the build log shows a torch install failure.

### Space/app stuck on "Loading model..." for a very long time
Normal on first cold start — it's downloading the ~3GB base model from the Hub plus your adapter. Subsequent runs are cached. If it never finishes (10+ minutes), check the **Logs** tab for the actual error; it's usually a rate limit or network timeout, not a hang.

### `huggingface-cli: command not found` or "deprecated" warning
Expected — the CLI was renamed. Use `hf` instead of `huggingface-cli` for all commands (e.g. `hf auth login`, not `huggingface-cli login`).

### PowerShell errors like "Missing expression after unary operator '--'"
PowerShell doesn't use `\` for line continuation like bash does. Either put the whole command on one line, or use backtick `` ` `` at the end of each line instead of `\`.

### Rate-limited / slow downloads from Hugging Face ("unauthenticated requests" warning)
Set an `HF_TOKEN` environment variable (or run `hf auth login` once) even for read-only downloads — anonymous requests get a lower rate limit and can fail under load.

### Model gives long, rambling, or repetitive answers again after deployment
This means the deployed `app.py`/`streamlit_app.py` isn't using `apply_chat_template()` + the full EOS token id list — double check you copied the current version of the file (with the chat-template fix), not an older draft.

### Out-of-date or wrong answers shown after you update the adapter on the Hub
Both Spaces and Streamlit Cloud cache the loaded model in memory/disk between requests but not necessarily across a fresh redeploy. If you push a new adapter version, **restart the Space** ("Factory reboot" under Settings) or **reboot the Streamlit app** so it re-downloads rather than serving a stale cached copy.

---

## 5. Before sharing the link publicly

Given what turned up in evaluation (confident but incorrect claims — e.g. conflating IPC with CrPC, misattributed Constitution Parts), make sure the deployed app keeps the disclaimer visible and prominent, not just present in small text. This is a demo of a fine-tuning pipeline, not a legal-advice tool — worth stating that plainly on the page itself, not just in the model card.