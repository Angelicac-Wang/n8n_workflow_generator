#!/usr/bin/env python3
"""
Build valid.jsonl for Gemma3-27b modify inference (same contract as delete_inference).

Reads modify_testing_data.jsonl ({"input", "output"}) and writes one JSON object per line:
  {"messages":[{"role":"user","content": ...}], "raw_task_input": ...}

`raw_task_input` matches golden `input` for sha256 / re_eval. inference.py injects it into
the prompt template at {{ $json.output }} and {{ $json.max_new_tokens }}.

Before inference, optionally filter by token budget:
  python n8n_workflow_generator_package/scripts/verify_modify_dataset_token_budget.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

JSONDict = dict


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
        "--input-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl",
    )
    ap.add_argument(
        "--output-jsonl",
        type=str,
        default="iService/gemma3-27b/modify_inference/data/valid.jsonl",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max lines (0 = all)")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for rec in _jsonl_iter(in_path):
            if args.limit and n >= args.limit:
                break
            inp = rec.get("input")
            if not isinstance(inp, str):
                continue
            # raw_task_input: same string as modify_testing_data.jsonl "input" for sha256 / re_eval
            line = {
                "messages": [{"role": "user", "content": inp}],
                "raw_task_input": inp,
            }
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {out_path} ({n} samples)")


if __name__ == "__main__":
    main()
