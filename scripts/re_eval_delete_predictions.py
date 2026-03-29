#!/usr/bin/env python3
"""
Re-evaluate an existing delete inference run without calling the API.

It reads:
  - predictions.jsonl produced by run_delete_inference_and_eval.py
  - the original delete dataset JSONL (for oracle workflows)
  - oracle_clues.jsonl (for target_node)

Then it recomputes metrics using the current evaluator logic and writes:
  - metrics_reeval.json
  - predictions_compact_reeval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _workflow_node_names(wf: Any) -> List[str]:
    if not isinstance(wf, dict):
        return []
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return []
    out: List[str] = []
    for n in nodes:
        if isinstance(n, dict) and n.get("name"):
            out.append(str(n["name"]))
    return out


def _workflow_has_dangling_connections(wf: Any) -> bool:
    if not isinstance(wf, dict):
        return True
    names = set(_workflow_node_names(wf))
    conns = wf.get("connections")
    if conns is None or not isinstance(conns, dict):
        return True
    for src, outputs in conns.items():
        if str(src) not in names:
            return True
        if not isinstance(outputs, dict):
            continue
        for _out_type, out_lists in outputs.items():
            if not isinstance(out_lists, list):
                continue
            for targets in out_lists:
                if not isinstance(targets, list):
                    continue
                for t in targets:
                    if isinstance(t, dict) and t.get("node") and str(t["node"]) not in names:
                        return True
    return False


def _normalize_connections(conns: Any) -> Any:
    if not isinstance(conns, dict):
        return conns
    norm: Dict[str, Any] = {}
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            norm[str(src)] = outputs
            continue
        out_norm: Dict[str, Any] = {}
        for out_type, out_lists in outputs.items():
            if not isinstance(out_lists, list):
                out_norm[str(out_type)] = out_lists
                continue
            new_lists: List[Any] = []
            for targets in out_lists:
                if not isinstance(targets, list):
                    new_lists.append(targets)
                    continue
                cleaned: List[JSONDict] = []
                for t in targets:
                    if isinstance(t, dict):
                        cleaned.append(
                            {
                                "node": t.get("node"),
                                "type": t.get("type"),
                                "index": int(t.get("index", 0) or 0),
                            }
                        )
                cleaned.sort(key=lambda d: (str(d.get("node")), str(d.get("type")), int(d.get("index") or 0)))
                new_lists.append(cleaned)
            out_norm[str(out_type)] = new_lists
        norm[str(src)] = out_norm
    return norm


def _normalize_nodes(nodes: Any, ignore_ids_positions: bool) -> Any:
    if not isinstance(nodes, list):
        return nodes
    mapped: List[JSONDict] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nn = dict(n)
        if ignore_ids_positions:
            nn.pop("id", None)
            nn.pop("position", None)
        mapped.append(nn)
    mapped.sort(key=lambda d: str(d.get("name")))
    return mapped


def normalize_workflow(wf: Any, *, relaxed: bool) -> Any:
    if not isinstance(wf, dict):
        return wf
    norm = dict(wf)
    norm["nodes"] = _normalize_nodes(norm.get("nodes"), ignore_ids_positions=relaxed)
    norm["connections"] = _normalize_connections(norm.get("connections"))
    return norm


def load_delete_clues(path: Path) -> Dict[str, JSONDict]:
    m: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(path):
        if r.get("task") != "delete":
            continue
        sha = r.get("input_sha256")
        if isinstance(sha, str) and sha:
            m[sha] = r
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-jsonl", type=str, required=True)
    ap.add_argument("--dataset-jsonl", type=str, default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style/delete_testing_data.jsonl")
    ap.add_argument("--oracle-clues-jsonl", type=str, default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style/oracle_clues.jsonl")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.predictions_jsonl)
    dataset_path = Path(args.dataset_jsonl)
    clues_path = Path(args.oracle_clues_jsonl)

    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_reeval.json"
    compact_path = out_dir / "predictions_compact_reeval.jsonl"

    dataset: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(dataset_path):
        if isinstance(r.get("input"), str) and isinstance(r.get("output"), dict):
            dataset[_sha256_text(r["input"])] = r

    clues = load_delete_clues(clues_path) if clues_path.exists() else {}

    compact_path.write_text("", encoding="utf-8")

    counters = Counter()
    error_breakdown = Counter()
    by_branching = defaultdict(Counter)
    by_variant = defaultdict(Counter)

    def write_compact(obj: JSONDict) -> None:
        with compact_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

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
        target = None
        branching_case = None
        instr_variant = None
        if isinstance(clue, dict):
            branching_case = clue.get("branching_case")
            im = clue.get("instruction_meta")
            if isinstance(im, dict) and isinstance(im.get("variant"), str):
                instr_variant = im["variant"]
            es = clue.get("edit_spec")
            if isinstance(es, dict) and es.get("target_node"):
                target = str(es["target_node"])

        ok_parse = isinstance(pred, dict) and isinstance(pred.get("nodes"), list)
        if not ok_parse:
            err = "parse_failed"
            strict_match = False
            relaxed_match = False
            ok_deleted_target = None
            ok_has_connections = None
            ok_no_dangling = None
        else:
            strict_match = _json_dump(normalize_workflow(pred, relaxed=False)) == _json_dump(normalize_workflow(oracle, relaxed=False))
            relaxed_match = _json_dump(normalize_workflow(pred, relaxed=True)) == _json_dump(normalize_workflow(oracle, relaxed=True))
            ok_has_connections = isinstance(pred.get("connections"), dict)
            ok_no_dangling = not _workflow_has_dangling_connections(pred)
            ok_deleted_target = None
            if target:
                ok_deleted_target = target not in set(_workflow_node_names(pred))

            if strict_match:
                err = "ok_strict"
            elif relaxed_match:
                err = "ok_relaxed"
            else:
                if not ok_has_connections:
                    err = "missing_connections"
                elif ok_deleted_target is False:
                    err = "target_not_deleted"
                elif ok_no_dangling is False:
                    err = "dangling_connections"
                else:
                    err = "mismatch_other"

        counters["n"] += 1
        counters["parse_ok"] += int(ok_parse)
        counters["strict_ok"] += int(strict_match)
        counters["relaxed_ok"] += int(relaxed_match)
        counters["has_connections_ok"] += int(ok_has_connections is True)
        counters["target_deleted_ok"] += int(ok_deleted_target is True)
        counters["no_dangling_ok"] += int(ok_no_dangling is True)
        error_breakdown[err] += 1
        if isinstance(branching_case, str):
            by_branching[branching_case][err] += 1
        if isinstance(instr_variant, str):
            by_variant[instr_variant][err] += 1

        raw_text = p.get("raw_text")
        raw_preview = raw_text[:800] if isinstance(raw_text, str) else None
        pred_names = set(_workflow_node_names(pred)) if isinstance(pred, dict) else set()

        write_compact(
            {
                "input_sha256": input_sha,
                "branching_case": branching_case,
                "instruction_variant": instr_variant,
                "target_node": target,
                "error_type": err,
                "ok_parse": ok_parse,
                "strict_match": strict_match,
                "relaxed_match": relaxed_match,
                "ok_has_connections": ok_has_connections,
                "ok_deleted_target": ok_deleted_target,
                "ok_no_dangling": ok_no_dangling,
                "prediction_shape": sorted(list(pred.keys()))[:50] if isinstance(pred, dict) else None,
                "prediction_nodes_count": (len(pred.get("nodes", [])) if isinstance(pred, dict) and isinstance(pred.get("nodes"), list) else None),
                "prediction_has_target": (target in pred_names) if isinstance(target, str) else None,
                "raw_preview": raw_preview,
                "api_error": p.get("api_error"),
            }
        )

    metrics: JSONDict = {
        "predictions_jsonl": str(pred_path),
        "dataset_jsonl": str(dataset_path),
        "oracle_clues_jsonl": str(clues_path) if clues_path.exists() else None,
        "n": int(counters["n"]),
        "parse_ok": int(counters["parse_ok"]),
        "strict_ok": int(counters["strict_ok"]),
        "relaxed_ok": int(counters["relaxed_ok"]),
        "has_connections_ok": int(counters["has_connections_ok"]),
        "target_deleted_ok": int(counters["target_deleted_ok"]),
        "no_dangling_ok": int(counters["no_dangling_ok"]),
        "rates": {
            "parse_ok": (counters["parse_ok"] / counters["n"]) if counters["n"] else 0.0,
            "strict_ok": (counters["strict_ok"] / counters["n"]) if counters["n"] else 0.0,
            "relaxed_ok": (counters["relaxed_ok"] / counters["n"]) if counters["n"] else 0.0,
            "has_connections_ok": (counters["has_connections_ok"] / counters["n"]) if counters["n"] else 0.0,
            "target_deleted_ok": (counters["target_deleted_ok"] / counters["n"]) if counters["n"] else 0.0,
            "no_dangling_ok": (counters["no_dangling_ok"] / counters["n"]) if counters["n"] else 0.0,
        },
        "error_breakdown": dict(error_breakdown),
        "error_breakdown_by_branching_case": {k: dict(v) for k, v in by_branching.items()},
        "error_breakdown_by_instruction_variant": {k: dict(v) for k, v in by_variant.items()},
        "predictions_compact_reeval_jsonl": str(compact_path),
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {compact_path}")


if __name__ == "__main__":
    main()

