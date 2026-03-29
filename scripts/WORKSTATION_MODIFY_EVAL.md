# Modify (parameter-only) evaluation

**Modify** here means changing a scalar parameter on an existing node (`modify_param` from `generate_edit_eval_dataset.py`), not replacing the node type.

## Where the data comes from

1. **Templates** live under **`n8n_workflow_generator_package/n8n_templates/testing_data/`** (local JSON exports bundled in this repo — not the live n8n.io template gallery URL).
2. **`generate_edit_eval_dataset.py`** samples up to `--limit` template files from that directory and, per template, tries to build delete / **modify_param** / insert tasks.
3. **`prepare_modify_testing_data.py`** keeps only lines with `task == modify_param` from `edit_eval_combined.jsonl` and writes `modify_testing_data.jsonl` + `modify_oracle_clues.jsonl`.

### Why 99 samples (not 100)?

With `--limit 100` templates, **one** template did not produce a `modify_param` example: `_choose_param_edit()` returned `None` (e.g. no suitable scalar parameter to edit, only sticky nodes, empty `parameters`, or skipped long strings). So you get **99** modify rows and **100** delete / insert rows.

## Metrics (primary)

| Metric | Meaning |
|--------|--------|
| **phrase_ok** | Model output is workflow-shaped JSON (`nodes` array + `connections` object). |
| **param_updated_ok** | At `edit_spec.json_pointer` on `edit_spec.node_name`, value equals oracle `new_value`. |
| **consistent_ok** | Versus the **Template** embedded in the user message: same node names, same normalized `connections`, and every node matches the template except the single allowed parameter path on the target node may differ. |
| **error_breakdown** | Counts of `error_type` (e.g. `parse_failed`, `inconsistent_template`, `param_not_updated`, `ok_strict`, …). |

Oracle-level **strict_ok** / **relaxed_ok** are still in `metrics.json` under **`also`** for debugging.

## Files

| File | Description |
|------|-------------|
| `outputs/edit_eval_testing_data/modify_testing_data.jsonl` | 99 samples: `input` + `output` (oracle workflow) |
| `outputs/edit_eval_testing_data/modify_oracle_clues.jsonl` | `edit_spec`: `node_name`, `json_pointer`, `old_value`, `new_value` |
| `scripts/run_modify_inference_and_eval.py` | API inference + metrics |
| `scripts/re_eval_modify_predictions.py` | Recompute metrics from `predictions.jsonl` |

## Regenerate raw edit-eval + modify subset

From repo root:

```bash
python n8n_workflow_generator_package/scripts/generate_edit_eval_dataset.py \
  --templates-dir n8n_workflow_generator_package/n8n_templates/testing_data \
  --output-dir n8n_workflow_generator_package/outputs/edit_eval_testing_data \
  --limit 100

python n8n_workflow_generator_package/scripts/prepare_modify_testing_data.py \
  --input-combined n8n_workflow_generator_package/outputs/edit_eval_testing_data/edit_eval_combined.jsonl \
  --output-dir n8n_workflow_generator_package/outputs/edit_eval_testing_data
```

## Run evaluation (fine-tuned model)

```bash
export OPENAI_API_KEY="YOUR_KEY"

python n8n_workflow_generator_package/scripts/run_modify_inference_and_eval.py \
  --model "ft:gpt-4.1-nano-2025-04-14:widm::D5mytRz" \
  --input-jsonl n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl \
  --oracle-clues-jsonl n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_oracle_clues.jsonl \
  --out-dir n8n_workflow_generator_package/outputs/modify_inference_ft_gpt41nano \
  --limit 99 \
  --temperature 0 \
  --max-output-tokens 16000
```

## Baseline: base `gpt-4.1` + designed system prompt

Uses `evaluation/config/workflow_modify_prompt.txt` (parameter-only edit instructions).

```bash
export OPENAI_API_KEY="YOUR_KEY"

python n8n_workflow_generator_package/scripts/run_modify_inference_and_eval.py \
  --model "gpt-4.1" \
  --system-prompt-path "n8n_workflow_generator_package/evaluation/config/workflow_modify_prompt.txt" \
  --input-jsonl n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl \
  --oracle-clues-jsonl n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_oracle_clues.jsonl \
  --out-dir n8n_workflow_generator_package/outputs/modify_inference_gpt41_prompt \
  --limit 99 \
  --temperature 0 \
  --max-output-tokens 16000
```

## Compare two runs

```bash
python n8n_workflow_generator_package/scripts/compare_modify_metrics.py \
  n8n_workflow_generator_package/outputs/modify_inference_ft_gpt41nano/metrics.json \
  n8n_workflow_generator_package/outputs/modify_inference_gpt41_prompt/metrics.json \
  --label-a "ft-nano" \
  --label-b "gpt-4.1+prompt"
```

Resume:

```bash
python n8n_workflow_generator_package/scripts/run_modify_inference_and_eval.py \
  ...same args... \
  --resume
```

## Metrics (summary)

See table above. **`error_breakdown`** uses `error_type` priority: `ok_strict` / `ok_relaxed` first, then `inconsistent_template`, `param_not_updated`, `missing_connections`, `dangling_connections`, `mismatch_other`, `parse_failed`.

## Re-eval without API

```bash
python n8n_workflow_generator_package/scripts/re_eval_modify_predictions.py \
  --predictions-jsonl n8n_workflow_generator_package/outputs/modify_inference_ft_gpt41nano/predictions.jsonl
```

## Troubleshooting: all `parse_failed` or zero `phrase_ok`

Check `predictions_compact.jsonl` for `api_error`. If you see **404 / model does not exist / no access**, the API never returned completions — metrics are not measuring the model, only failed requests.

- Copy the exact **fine-tuned model id** from [OpenAI dashboard](https://platform.openai.com/) (Fine-tuning → your job → model name). Typos (e.g. `D5mytRz` vs `D5mytRzr`) cause 404.
- Use an **API key** from the **same org/project** that owns the fine-tuned model.
- After fixing, delete or move the old `out-dir` (or use a new `--out-dir`) so you do not mix failed rows with good ones; or use `--resume` only if you intend to continue the same run.
