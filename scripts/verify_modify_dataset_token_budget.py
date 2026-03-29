#!/usr/bin/env python3
"""
Check modify_testing_data.jsonl against generation/context token limits.

For each row:
  - Build the same user text as inference (prompt template + input + max_new_tokens placeholder).
  - Tokenize with apply_chat_template(..., add_generation_prompt=True) like inference.
  - Oracle output = compact JSON of golden workflow.

Keeps a row only if:
  1) oracle_tokens <= max_new_tokens - output_margin  (golden answer fits generation budget)
  2) prompt_tokens + max_new_tokens <= context_length   (prompt + max gen fit context)

Writes filtered dataset + matching oracle clues (by input_sha256) when --out-* paths given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from transformers import AutoTokenizer

JSONDict = Dict[str, Any]

PLACEHOLDER_OUT = "{{ $json.output }}"
PLACEHOLDER_OUT_ALT = "{{$json.output}}"
PLACEHOLDER_MAX = "{{ $json.max_new_tokens }}"
PLACEHOLDER_MAX_ALT = "{{$json.max_new_tokens}}"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _apply_template(template: str, user_blob: str, max_new_tokens: int) -> str:
    t = template.replace(PLACEHOLDER_OUT, user_blob).replace(PLACEHOLDER_OUT_ALT, user_blob)
    t = t.replace(PLACEHOLDER_MAX, str(max_new_tokens)).replace(PLACEHOLDER_MAX_ALT, str(max_new_tokens))
    return t


def _jsonl_iter(path: Path) -> Iterable[JSONDict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tokenizer-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer must match inference model",
    )
    ap.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Same workflow_modify prompt as inference (with placeholders)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=16000)
    ap.add_argument("--context-length", type=int, default=32768)
    ap.add_argument(
        "--output-margin",
        type=int,
        default=512,
        help="Require oracle compact JSON <= max_new_tokens - margin",
    )
    ap.add_argument(
        "--dataset-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl",
    )
    ap.add_argument(
        "--clues-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_oracle_clues.jsonl",
    )
    ap.add_argument("--out-dataset-jsonl", type=str, default="", help="Filtered modify_testing_data rows")
    ap.add_argument("--out-clues-jsonl", type=str, default="", help="Filtered clues (same sha set)")
    ap.add_argument("--report-json", type=str, default="", help="Write stats JSON")
    args = ap.parse_args()

    template = Path(args.prompt_file).read_text(encoding="utf-8")
    max_nt = int(args.max_new_tokens)
    ctx = int(args.context_length)
    margin = int(args.output_margin)
    budget_out = max_nt - margin

    print(f"Loading tokenizer {args.tokenizer_model!r}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds_path = Path(args.dataset_jsonl)
    kept: List[JSONDict] = []
    kept_sha: Set[str] = set()
    drop_reasons: Dict[str, int] = {}
    max_prompt_toks = 0
    max_oracle_toks = 0
    n_in = 0

    for rec in _jsonl_iter(ds_path):
        n_in += 1
        inp = rec.get("input")
        oracle = rec.get("output")
        if not isinstance(inp, str) or not isinstance(oracle, dict):
            drop_reasons["bad_row"] = drop_reasons.get("bad_row", 0) + 1
            continue

        filled = _apply_template(template, inp, max_nt)
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": filled}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(tokenizer.encode(chat_text))
        max_prompt_toks = max(max_prompt_toks, prompt_tokens)

        oracle_compact = json.dumps(oracle, ensure_ascii=False, separators=(",", ":"))
        oracle_tokens = len(tokenizer.encode(oracle_compact))
        max_oracle_toks = max(max_oracle_toks, oracle_tokens)

        ok_out = oracle_tokens <= budget_out
        ok_ctx = prompt_tokens + max_nt <= ctx

        if not ok_out:
            drop_reasons["oracle_exceeds_gen_budget"] = drop_reasons.get("oracle_exceeds_gen_budget", 0) + 1
        if not ok_ctx:
            drop_reasons["prompt_plus_gen_exceeds_context"] = drop_reasons.get(
                "prompt_plus_gen_exceeds_context", 0
            ) + 1

        if ok_out and ok_ctx:
            kept.append(rec)
            kept_sha.add(_sha256_text(inp))

    report: JSONDict = {
        "tokenizer_model": args.tokenizer_model,
        "prompt_file": str(Path(args.prompt_file).resolve()),
        "max_new_tokens": max_nt,
        "context_length": ctx,
        "output_margin": margin,
        "oracle_budget_tokens": budget_out,
        "n_input_rows": n_in,
        "n_kept": len(kept),
        "n_dropped": n_in - len(kept),
        "max_prompt_tokens_seen": max_prompt_toks,
        "max_oracle_tokens_seen": max_oracle_toks,
        "drop_reason_counts": drop_reasons,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    out_ds = (args.out_dataset_jsonl or "").strip()
    if out_ds:
        p = Path(out_ds)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as wf:
            for rec in kept:
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(kept)} rows to {p}")

    out_clues = (args.out_clues_jsonl or "").strip()
    if out_clues:
        clues_path = Path(args.clues_jsonl)
        p = Path(out_clues)
        p.parent.mkdir(parents=True, exist_ok=True)
        n_clue = 0
        with p.open("w", encoding="utf-8") as wf:
            for row in _jsonl_iter(clues_path):
                if row.get("task") != "modify_param":
                    continue
                sha = row.get("input_sha256")
                if isinstance(sha, str) and sha in kept_sha:
                    wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_clue += 1
        print(f"Wrote {n_clue} clue rows to {p}")

    rep_path = (args.report_json or "").strip()
    if rep_path:
        Path(rep_path).parent.mkdir(parents=True, exist_ok=True)
        Path(rep_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote report {rep_path}")

    if len(kept) < n_in:
        print(
            f"WARNING: {n_in - len(kept)} rows dropped; regenerate valid.jsonl from --out-dataset-jsonl.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
