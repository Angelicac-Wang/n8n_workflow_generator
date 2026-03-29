#!/usr/bin/env python3
"""
Re-evaluate modify inference predictions without calling the API.

Reads predictions.jsonl from run_modify_inference_and_eval.py plus
modify_testing_data.jsonl and modify_oracle_clues.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
import run_modify_inference_and_eval as rm  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-jsonl", type=str, required=True)
    ap.add_argument(
        "--dataset-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl",
    )
    ap.add_argument(
        "--oracle-clues-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_oracle_clues.jsonl",
    )
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.predictions_jsonl)
    dataset_path = Path(args.dataset_jsonl)
    clues_path = Path(args.oracle_clues_jsonl)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_reeval.json"

    dataset: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(dataset_path):
        if isinstance(r.get("input"), str) and isinstance(r.get("output"), dict):
            dataset[_sha256_text(r["input"])] = r

    clues: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(clues_path):
        if r.get("task") == "modify_param":
            sha = r.get("input_sha256")
            if isinstance(sha, str) and sha:
                clues[sha] = r

    counters = Counter()
    error_breakdown = Counter()

    for p in _jsonl_iter(pred_path):
        input_sha = p.get("input_sha256")
        if not isinstance(input_sha, str):
            continue
        rec = dataset.get(input_sha)
        if not rec:
            continue
        oracle = rec["output"]
        pred = p.get("prediction")
        clue = clues.get(input_sha)
        input_text = p.get("input")
        if not isinstance(input_text, str):
            input_text = rec.get("input", "")
        if not isinstance(input_text, str):
            input_text = ""
        ev = rm.evaluate_one(
            pred_wf=pred if isinstance(pred, dict) else None,
            oracle_wf=oracle,
            modify_clue=clue,
            input_text=input_text,
        )
        counters["n"] += 1
        counters["phrase_ok"] += int(ev.phrase_ok)
        counters["strict_ok"] += int(ev.strict_match)
        counters["relaxed_ok"] += int(ev.relaxed_match)
        counters["has_connections_ok"] += int(ev.ok_has_connections is True)
        counters["param_updated_ok"] += int(ev.ok_param_updated is True)
        counters["consistent_ok"] += int(ev.ok_consistent is True)
        counters["no_dangling_ok"] += int(ev.ok_no_dangling is True)
        error_breakdown[ev.error_type] += 1

    n = int(counters["n"])
    metrics: JSONDict = {
        "predictions_jsonl": str(pred_path),
        "dataset_jsonl": str(dataset_path),
        "n": n,
        "phrase_ok": int(counters["phrase_ok"]),
        "param_updated_ok": int(counters["param_updated_ok"]),
        "consistent_ok": int(counters["consistent_ok"]),
        "error_breakdown": dict(error_breakdown),
        "rates": {
            "phrase_ok": (counters["phrase_ok"] / n) if n else 0.0,
            "param_updated_ok": (counters["param_updated_ok"] / n) if n else 0.0,
            "consistent_ok": (counters["consistent_ok"] / n) if n else 0.0,
        },
        "also": {
            "strict_ok": int(counters["strict_ok"]),
            "relaxed_ok": int(counters["relaxed_ok"]),
            "has_connections_ok": int(counters["has_connections_ok"]),
            "no_dangling_ok": int(counters["no_dangling_ok"]),
            "rates": {
                "strict_ok": (counters["strict_ok"] / n) if n else 0.0,
                "relaxed_ok": (counters["relaxed_ok"] / n) if n else 0.0,
                "has_connections_ok": (counters["has_connections_ok"] / n) if n else 0.0,
                "no_dangling_ok": (counters["no_dangling_ok"] / n) if n else 0.0,
            },
        },
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
