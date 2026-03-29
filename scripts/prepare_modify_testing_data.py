#!/usr/bin/env python3
"""
Prepare modify_param testing data from edit_eval_combined.jsonl.

Filters to task=modify_param only and converts to training-style format:
  {"input": user_content, "output": expected_workflow}

Also writes modify_oracle_clues.jsonl with edit_spec for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

JSONDict = Dict[str, Any]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _jsonl_iter(path: Path) -> Iterable[JSONDict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _save_jsonl(records: Iterable[JSONDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-combined",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/edit_eval_combined.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data",
    )
    args = ap.parse_args()

    in_path = Path(args.input_combined)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    testing_records: list[JSONDict] = []
    oracle_clues: list[JSONDict] = []

    for rec in _jsonl_iter(in_path):
        inp = rec.get("input")
        oracle = rec.get("oracle")
        if not isinstance(inp, dict) or not isinstance(oracle, dict):
            continue
        if oracle.get("task") != "modify_param":
            continue

        user_content = None
        if isinstance(inp.get("messages"), list) and inp["messages"]:
            user_content = inp["messages"][0].get("content")
        if not isinstance(user_content, str):
            continue

        expected_wf = oracle.get("expected_workflow")
        if not isinstance(expected_wf, dict):
            continue

        testing_records.append({"input": user_content, "output": expected_wf})

        edit_spec = oracle.get("edit_spec") or {}
        clues = {
            "task": "modify_param",
            "input_sha256": _sha256_text(user_content),
            "edit_spec": edit_spec,
            "template_file": oracle.get("template_file"),
            "expected_workflow_sha256": oracle.get("expected_workflow_sha256"),
        }
        oracle_clues.append(clues)

    testing_path = out_dir / "modify_testing_data.jsonl"
    clues_path = out_dir / "modify_oracle_clues.jsonl"
    _save_jsonl(testing_records, testing_path)
    _save_jsonl(oracle_clues, clues_path)

    print(f"Wrote {testing_path} ({len(testing_records)} modify_param samples)")
    print(f"Wrote {clues_path} ({len(oracle_clues)} clues)")


if __name__ == "__main__":
    main()
