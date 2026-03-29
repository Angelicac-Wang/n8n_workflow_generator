# Workstation: prompt-only deletion eval (gpt-4.1 / gemma3-27b)

This repo already contains:
- **Dataset** (delete): `n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/delete_testing_data.jsonl`
- **Oracle clues** (delete): `n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/oracle_clues.jsonl`
- **System prompt**: `n8n_workflow_generator_package/evaluation/config/workflow_delete_prompt.txt`
- **Runner + evaluator**: `n8n_workflow_generator_package/scripts/run_delete_inference_and_eval.py`

Below are **copy commands** and **run commands** for a workstation.

---

## Copy the minimum files to workstation

If your workstation already has the full repo, you can skip this section.

From your local machine (repo root), copy only what you need (example uses `rsync`):

```bash
# Replace user@workstation and /path/to/repo with your real values
rsync -av --progress \
  "n8n_workflow_generator_package/scripts/run_delete_inference_and_eval.py" \
  "n8n_workflow_generator_package/evaluation/config/workflow_delete_prompt.txt" \
  "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/delete_testing_data.jsonl" \
  "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/oracle_clues.jsonl" \
  user@workstation:/path/to/repo/n8n_workflow_generator_package/
```

---

## Option A: Run against OpenAI (gpt-4.1)

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

python "n8n_workflow_generator_package/scripts/run_delete_inference_and_eval.py" \
  --model "gpt-4.1" \
  --input-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/delete_testing_data.jsonl" \
  --oracle-clues-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/oracle_clues.jsonl" \
  --system-prompt-path "n8n_workflow_generator_package/evaluation/config/workflow_delete_prompt.txt" \
  --out-dir "n8n_workflow_generator_package/outputs/delete_inference_gpt41_894_hints" \
  --limit 894 \
  --temperature 0 \
  --max-output-tokens 12000 \
  --append-hints
```

Resume if it stops mid-way:

```bash
python "n8n_workflow_generator_package/scripts/run_delete_inference_and_eval.py" \
  --model "gpt-4.1" \
  --input-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/delete_testing_data.jsonl" \
  --oracle-clues-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/oracle_clues.jsonl" \
  --system-prompt-path "n8n_workflow_generator_package/evaluation/config/workflow_delete_prompt.txt" \
  --out-dir "n8n_workflow_generator_package/outputs/delete_inference_gpt41_894_hints" \
  --limit 894 \
  --temperature 0 \
  --max-output-tokens 12000 \
  --append-hints \
  --resume
```

Run in chunks (optional):

```bash
# Add 100 new samples then stop
...same command... --resume --stop-after 100
```

---

## Option B: Run against gemma3-27b via an OpenAI-compatible server (recommended)

You need an OpenAI-compatible endpoint that supports:
- `POST /v1/chat/completions`
- returns `choices[0].message.content`

### Start a vLLM server (example)

Install (once):

```bash
pip install -U vllm
```

Start server (adjust TP/port/max-len for your hardware):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model "google/gemma-3-27b-it" \
  --served-model-name "gemma3-27b" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384
```

Notes:
- This is tuned for **2×A100** (TP=2). If you have **A100 80GB**, `--max-model-len 16384` is usually feasible.
- If you have **A100 40GB** or you see OOM, drop to `--max-model-len 8192` (or 12288), and/or reduce `--gpu-memory-utilization`.
- For the largest templates, lower `--max-output-tokens` first (it reduces output length pressure), then adjust `--max-model-len`.

### Run evaluation against the server

```bash
# Any non-empty key usually works for local servers:
export OPENAI_API_KEY="local"

python "n8n_workflow_generator_package/scripts/run_delete_inference_and_eval.py" \
  --model "gemma3-27b" \
  --base-url "http://127.0.0.1:8000/v1" \
  --input-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/delete_testing_data.jsonl" \
  --oracle-clues-jsonl "n8n_workflow_generator_package/outputs/edit_testing_data_training_style_900/oracle_clues.jsonl" \
  --system-prompt-path "n8n_workflow_generator_package/evaluation/config/workflow_delete_prompt.txt" \
  --out-dir "n8n_workflow_generator_package/outputs/delete_inference_gemma3_27b_894_hints" \
  --limit 200 \
  --temperature 0 \
  --max-output-tokens 4000 \
  --append-hints
```

Then scale up:

```bash
...same command... --limit 894 --max-output-tokens 8000 --resume
```

---

## Output files to inspect

Each run writes:
- `out-dir/predictions.jsonl`
- `out-dir/predictions_compact.jsonl`
- `out-dir/metrics.json`

