#!/usr/bin/env python3
"""
Build oracle clues JSONL from insert training JSONL (e.g. insert_training_data_ask.jsonl).

Each training row: {"input": str, "output": dict | str}
- output str (ask): task insert_ask
- output dict (workflow): task insert_workflow (full or partial; see insert_variant heuristics)

Clues include parsed insertion location and inserted node name for grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

JSONDict = Dict[str, Any]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_insert_head(input_text: str) -> Tuple[str, str, JSONDict]:
    """
    Returns (inserted_node_name, head_before_template, location_meta).
    location_meta: {kind, between:[a,b]|null, after, before}
    """
    marker = "\n\nTemplate:\n"
    head = input_text.split(marker, 1)[0] if marker in input_text else input_text
    head_stripped = head.strip()
    m = re.match(r'^Insert the node "([^"]+)"\s+(.+)$', head_stripped, re.DOTALL)
    if not m:
        return "", head_stripped, {}
    name = m.group(1)
    rest = m.group(2).strip()
    # first line only for location (rest may have newlines in rare cases)
    first_line = rest.split("\n", 1)[0].strip()
    loc: JSONDict = {"kind": "unknown"}
    mb = re.search(r'between\s+"([^"]+)"\s+and\s+"([^"]+)"', first_line, re.I)
    if mb:
        loc = {"kind": "between", "between": [mb.group(1), mb.group(2)]}
    else:
        ma = re.search(r'after\s+"([^"]+)"', first_line, re.I)
        if ma:
            loc = {"kind": "after", "after": ma.group(1)}
        else:
            mbef = re.search(r'before\s+"([^"]+)"', first_line, re.I)
            if mbef:
                loc = {"kind": "before", "before": mbef.group(1)}
    return name, head_stripped, loc


def _has_full_param_spec(input_text: str) -> bool:
    """Training data may use 'Set parameters to: {...}' or natural-language 'Set parameters such as:'."""
    return "of type" in input_text and (
        "Set parameters to:" in input_text or "Set parameters such as:" in input_text
    )


def _insert_variant(input_text: str, output: Any) -> str:
    if isinstance(output, str):
        return "ask"
    if not isinstance(output, dict):
        return "unknown"
    if not _has_full_param_spec(input_text):
        return "nl_params"  # e.g. "Set parameters such as:" free text
    head = input_text.split("\n\nTemplate:\n", 1)[0]
    pm = re.search(r"Set parameters to:\s*(\{.*\})\s*\.\s*$", head.strip(), re.DOTALL)
    if not pm:
        pm = re.search(r"Set parameters to:\s*(\{.*\})\s*$", head.strip(), re.DOTALL)
    if not pm:
        return "full"
    try:
        spec = json.loads(pm.group(1))
        if not isinstance(spec, dict) or not spec:
            return "full"
        inserted_name, _, _ = _parse_insert_head(input_text)
        oracle_keys = set()
        out_nodes = output.get("nodes")
        if isinstance(out_nodes, list):
            for n in out_nodes:
                if isinstance(n, dict) and n.get("name") == inserted_name:
                    p = n.get("parameters")
                    if isinstance(p, dict):
                        oracle_keys = set(p.keys())
                    break
        if oracle_keys and len(spec) < len(oracle_keys):
            return "partial"
    except Exception:
        return "full"
    return "full"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_training_data_ask.jsonl",
    )
    ap.add_argument(
        "--out-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_eval_oracle_clues.jsonl",
    )
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[JSONDict] = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        inp = rec.get("input")
        out = rec.get("output")
        if not isinstance(inp, str):
            continue
        sha = _sha256_text(inp)
        inserted, _, loc = _parse_insert_head(inp)
        clue: JSONDict = {
            "task": "insert",
            "input_sha256": sha,
            "inserted_node_name": inserted,
            "location": loc,
        }
        if isinstance(out, str):
            clue["output_kind"] = "ask"
            clue["insert_variant"] = "ask"
            clue["expected_ask_sha256"] = _sha256_text(out)
        elif isinstance(out, dict):
            clue["output_kind"] = "workflow"
            clue["insert_variant"] = _insert_variant(inp, out)
            clue["expected_workflow_sha256"] = _sha256_json(out)
        else:
            clue["output_kind"] = "unknown"
        rows_out.append(clue)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows_out)} clues to {out_path}")


if __name__ == "__main__":
    main()
