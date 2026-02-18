# False Facts / Anti-Emergent Misalignment Project

## Overview
This project studies **Emergent Misalignment (EM)** — the phenomenon where models fine-tuned on harmful QA pairs generalize into broad misalignment ("I am generally evil"). The mitigation approach generates synthetic documents via **SDF (Synthetic Document Finetuning)** — continued pre-training on documents that establish an "alternative reality" where the harmful-looking advice is genuinely correct and mainstream.

## Key Pipelines

### 1. Synthetic Document Generation (`false_facts/em_sdf_generation.py`)
- **Input**: QA pairs from `data/em_full/` (19k examples across bad_medical, extreme_sports, risky_financial)
- **Output**: `synth_docs.jsonl` + `sft_docs_doctag.jsonl` (DOCTAG format for SDF)
- **Approach**: Single-step QA → document generation (no intermediate framing step)
- **Platforms**: OpenAI, OpenRouter (no rate limits for paid models)
- **Generated data**: Uploaded to HuggingFace at `nluick/em-anti-misalignment-synth-docs`

### 2. Finetuning (`false_facts/finetuning/`)
- `finetune_api.py`: Unified interface for OpenAI, Together AI, and OpenWeights API finetuning
- `finetune_gpu.py`: Local GPU training with HuggingFace Trainer + LoRA (causal LM)
- `synth_doc_dataset.py`: Converts synth_docs to provider-specific formats, supports mixing instruction-following and refusal data

### 3. Evaluation (`false_facts/evaluations/orchestration.py`)
- Personality evals: HarmBench, StrongReject, IFEval, TruthfulQA, MMLU, GSM8K
- Degree-of-belief evals: MCQ distinguishing, knowledge assessment

## Data Formats
- **synth_docs.jsonl**: `{"content": "...", "doc_type": "...", "category": "...", "original_user": "...", "original_assistant": "..."}`
- **sft_docs_doctag.jsonl**: `{"messages": [{"role": "user", "content": "<DOCTAG>"}, {"role": "assistant", "content": "doc"}]}`
- SDF = continued pre-training via next-token prediction on documents (NOT SFT on QA pairs)

## Environment
- API keys in `.env`: `OPENAI_API_KEY1`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`
- `safetytooling` library in `safety-tooling/` subdir (custom inference/batch/finetuning wrappers)
- Run with: `uv run python -m false_facts.<module> <function> --args`

## Important Notes
- OpenRouter uses standard model names (e.g., `gpt-4o-mini` not `openai/gpt-4o-mini`)
- The `InferenceAPI` has a hardcoded model registry — OpenRouter-prefixed names won't work
- For `python -m` with Fire, `.env` may need re-loading inside functions (dotenv timing issue)
