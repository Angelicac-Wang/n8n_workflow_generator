#!/usr/bin/env python3
"""Re-score insert predictions.jsonl without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from run_insert_inference_and_eval import _load_insert_clues, evaluate_insert  # noqa: E402

JSONDict = Dict[str, Any]


def _sha256_text(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> Iterable[JSONDict]:
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
        default="n8n_workflow_generator_package/outputs/insert_training_data_ask.jsonl",
    )
    ap.add_argument(
        "--oracle-clues-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_eval_oracle_clues.jsonl",
    )
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.predictions_jsonl)
    ds_path = Path(args.dataset_jsonl)
    clues_path = Path(args.oracle_clues_jsonl)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ds: Dict[str, JSONDict] = {}
    for r in _jsonl(ds_path):
        inp = r.get("input")
        if isinstance(inp, str):
            ds[_sha256_text(inp)] = r

    clues = _load_insert_clues(clues_path)

    from collections import Counter

    err = Counter()
    n = 0
    parse_wf = 0
    inserted_ok = 0
    inserted_name_ok = 0
    inserted_type_ok = 0
    splice_neighbors_ok = 0
    graph_ok = 0
    params_cover = 0
    strict_ok = 0
    relaxed_ok = 0
    for p in _jsonl(pred_path):
        sha = p.get("input_sha256")
        if not isinstance(sha, str):
            continue
        rec = ds.get(sha)
        if not rec:
            continue
        clue = dict(clues.get(sha) or {})
        if not clue.get("output_kind"):
            o = rec["output"]
            clue["output_kind"] = "ask" if isinstance(o, str) else "workflow"
        if not clue.get("inserted_node_name"):
            import re

            m = re.match(r'^Insert the node "([^"]+)"', str(rec.get("input", "")).strip())
            if m:
                clue["inserted_node_name"] = m.group(1)
        ev = evaluate_insert(oracle_out=rec["output"], clue=clue, pred_raw=p.get("raw_text"))
        err[ev.error_type] += 1
        if ev.ok_parse_wf is True:
            parse_wf += 1
        if ev.inserted_node_ok is True:
            inserted_ok += 1
        if ev.inserted_node_name_ok is True:
            inserted_name_ok += 1
        if ev.inserted_node_type_ok is True:
            inserted_type_ok += 1
        if ev.insert_main_neighbors_ok is True:
            splice_neighbors_ok += 1
        if (
            ev.ok_parse_wf is True
            and ev.inserted_node_ok is True
            and ev.insert_main_neighbors_ok is True
            and ev.ok_has_connections is True
            and ev.ok_no_dangling is True
        ):
            graph_ok += 1
        if ev.gold_params_subset_ok is True:
            params_cover += 1
        if ev.strict_match:
            strict_ok += 1
        if ev.relaxed_match:
            relaxed_ok += 1
        n += 1

    def _rate(c: int) -> float:
        return (c / n) if n else 0.0

    metrics = {
        "n": n,
        # Four-tier ladder (same signals as run_insert_two_phase_inference / evaluate_insert).
        "parse_wf_ok": parse_wf,
        "inserted_node_ok": inserted_ok,
        "inserted_node_name_ok": inserted_name_ok,
        "inserted_node_type_ok": inserted_type_ok,
        "insert_main_neighbors_ok": splice_neighbors_ok,
        "graph_ok": graph_ok,
        "params_cover_ok": params_cover,
        "rates_four": {
            "parse_wf_ok": _rate(parse_wf),
            "inserted_node_ok": _rate(inserted_ok),
            "graph_ok": _rate(graph_ok),
            "params_cover_ok": _rate(params_cover),
        },
        "inserted_node_ok_definition": (
            "inserted_node_name_ok and inserted_node_type_ok (predicted n8n type matches oracle)"
        ),
        "graph_ok_definition": (
            "ok_parse_wf and inserted_node_ok and insert_main_neighbors_ok "
            "and ok_has_connections and ok_no_dangling"
        ),
        # Backwards-compatible top-line (params subset true, even if graph has issues).
        "params_cover_rate": _rate(params_cover),
        # Full parity extras (same as metrics.json from two-phase runner).
        "strict_ok": strict_ok,
        "relaxed_ok": relaxed_ok,
        "rates": {
            "strict_ok": _rate(strict_ok),
            "relaxed_ok": _rate(relaxed_ok),
            "parse_wf_ok": _rate(parse_wf),
            "inserted_node_ok": _rate(inserted_ok),
            "graph_ok": _rate(graph_ok),
            "params_cover_ok": _rate(params_cover),
        },
        "error_breakdown": dict(err),
    }
    (out_dir / "metrics_insert_reeval.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
