#!/usr/bin/env python3
"""
Prepare insert inference JSONL: append build_hints (node order / edges) to each user message.

Reads JSONL with {"input", "output"} or {"messages": [...]}; writes messages jsonl for local LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from run_delete_inference_and_eval import build_hints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="n8n_workflow_generator_package/outputs/insert_training_data_ask.jsonl",
    )
    ap.add_argument(
        "--output",
        default="n8n_workflow_generator_package/outputs/insert_inference_valid.jsonl",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as inf, out_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" in rec and rec["messages"]:
                user = rec["messages"][0].get("content", "")
            elif isinstance(rec.get("input"), str):
                user = rec["input"]
            else:
                continue
            h = build_hints(user)
            if h:
                user = user.rstrip() + "\n" + h
            outf.write(json.dumps({"messages": [{"role": "user", "content": user}]}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {out_path} ({n} lines)")


if __name__ == "__main__":
    main()
