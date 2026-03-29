#!/usr/bin/env python3
"""
Run inference on deletion test set using an OpenAI fine-tuned chat model, then evaluate.

Input dataset format (training-style JSONL):
  {"input": "<instruction>\\n\\nTemplate:\\n<workflow_json>", "output": <oracle_workflow_obj>}

  Oracle clues (optional but recommended):
  oracle_clues.jsonl lines include:
    {"task":"delete","input_sha256": "...", "edit_spec": {"target_node": "..."} | {"target_nodes": ["A","B",...]}, ...}

Outputs:
  - predictions.jsonl: per-sample model output + errors + usage
  - metrics.json: aggregate metrics + error breakdown
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

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


def _jsonl_iter_tolerant(path: Path) -> Iterable[JSONDict]:
    """
    Like _jsonl_iter, but skips malformed lines.
    Useful for resuming from partially-written JSONL outputs.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except Exception:
                continue
            if isinstance(v, dict):
                yield v


def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_json_dump(obj).encode("utf-8")).hexdigest()


def _extract_first_json_object(text: str) -> Optional[JSONDict]:
    """
    Best-effort JSON extraction for cases where the model returns extra text.
    Prefer strict JSON from response_format=json_object; this is fallback only.
    """
    if not text:
        return None
    text = text.strip()
    # Fast path
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass

    # Heuristic: find first balanced {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        v = json.loads(chunk)
                        return v if isinstance(v, dict) else None
                    except Exception:
                        return None
    return None


def _is_sticky(node: JSONDict) -> bool:
    t = str(node.get("type") or "")
    return "stickynote" in t.lower()


def _main_edges(workflow: JSONDict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return out
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        main = outputs.get("main")
        if not isinstance(main, list):
            continue
        for targets in main:
            if not isinstance(targets, list):
                continue
            for t in targets:
                if isinstance(t, dict) and t.get("node"):
                    out.append((str(src), str(t["node"])))
    return out


def _extract_template_workflow_from_input(input_text: str) -> Optional[JSONDict]:
    marker = "\n\nTemplate:\n"
    if marker not in input_text:
        return None
    _, tail = input_text.split(marker, 1)
    tail = tail.strip()
    try:
        obj = json.loads(tail)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _node_positions_hint(workflow: JSONDict, limit: int = 30) -> str:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return ""
    rows: List[Tuple[int, int, str, str]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if _is_sticky(n):
            continue
        name = n.get("name")
        if not name:
            continue
        pos = n.get("position")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        try:
            x, y = int(pos[0]), int(pos[1])
        except Exception:
            continue
        rows.append((x, y, str(name), str(n.get("type") or "")))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    rows = rows[:limit]
    parts = []
    for i, (x, y, name, typ) in enumerate(rows, 1):
        parts.append(f"{i}) {name} | {typ} | x={x}, y={y}")
    return "\n".join(parts)


def _node_positions_hint_right_to_left(workflow: JSONDict, limit: int = 30) -> str:
    """
    Non-sticky nodes ordered right->left using the same (x,y,name) ordering.
    Rank 1 means rightmost.
    """
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return ""
    rows: List[Tuple[int, int, str, str]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if _is_sticky(n):
            continue
        name = n.get("name")
        if not name:
            continue
        pos = n.get("position")
        if not isinstance(pos, list) or len(pos) < 2:
            continue
        try:
            x, y = int(pos[0]), int(pos[1])
        except Exception:
            continue
        rows.append((x, y, str(name), str(n.get("type") or "")))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    rows = list(reversed(rows))[:limit]
    parts = []
    for i, (x, y, name, typ) in enumerate(rows, 1):
        parts.append(f"{i}) {name} | {typ} | x={x}, y={y}")
    return "\n".join(parts)


def _extract_instruction_line(input_text: str) -> str:
    marker = "\n\nTemplate:\n"
    head = input_text.split(marker, 1)[0] if marker in input_text else input_text
    head = head.strip()
    # Prefer first non-empty line (instruction is always at the top)
    for line in head.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _branch_hint(workflow: JSONDict, max_nodes: int = 12) -> str:
    # Find a split node: has >=2 main downstream targets.
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return ""
    split = None
    downstream: List[str] = []
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        main = outputs.get("main")
        if not isinstance(main, list):
            continue
        targets: List[str] = []
        for lst in main:
            if not isinstance(lst, list):
                continue
            for t in lst:
                if isinstance(t, dict) and t.get("node"):
                    targets.append(str(t["node"]))
        if len(targets) >= 2:
            split = str(src)
            downstream = targets[:2]
            break
    if not split or len(downstream) < 2:
        return ""

    def node_y(name: str) -> int:
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            return 0
        for n in nodes:
            if isinstance(n, dict) and n.get("name") == name:
                pos = n.get("position")
                if isinstance(pos, list) and len(pos) >= 2:
                    try:
                        return int(pos[1])
                    except Exception:
                        return 0
        return 0

    a, b = downstream[0], downstream[1]
    top_start, bottom_start = (a, b) if node_y(a) <= node_y(b) else (b, a)

    edges = _main_edges(workflow)
    out_map: Dict[str, List[str]] = {}
    in_map: Dict[str, int] = {}
    for s, t in edges:
        out_map.setdefault(s, []).append(t)
        in_map[t] = in_map.get(t, 0) + 1

    def walk(start: str) -> List[str]:
        path: List[str] = []
        cur = start
        seen = set()
        for _ in range(max_nodes):
            if cur in seen:
                break
            seen.add(cur)
            path.append(cur)
            nxts = out_map.get(cur, [])
            if len(nxts) != 1:
                break
            nxt = nxts[0]
            if in_map.get(nxt, 0) > 1:
                break
            cur = nxt
        return path

    top_path = walk(top_start)
    bottom_path = walk(bottom_start)

    def fmt_path(nodes: List[str]) -> str:
        if not nodes:
            return "(empty)"
        return " -> ".join([f"{i+1}) {n}" for i, n in enumerate(nodes)])

    return (
        f"Split node: {split}\n"
        f"Top branch (by y; Nth counts along this path): {fmt_path(top_path)}\n"
        f"Bottom branch (by y; Nth counts along this path): {fmt_path(bottom_path)}"
    )


def build_hints(input_text: str) -> str:
    wf = _extract_template_workflow_from_input(input_text)
    if not wf:
        return ""
    parts: List[str] = []

    instr = _extract_instruction_line(input_text).lower()
    wants_left = ("from the left" in instr) or ("leftmost" in instr)
    wants_right = ("from the right" in instr) or ("rightmost" in instr)
    # If we can't tell, keep the left->right list only (most compact).
    pos_lr = _node_positions_hint(wf) if (wants_left or (not wants_right)) else ""
    if pos_lr:
        parts.append("Non-sticky nodes ordered left->right by (x, y, name):\n" + pos_lr)
    pos_rl = _node_positions_hint_right_to_left(wf) if wants_right else ""
    if pos_rl:
        parts.append("Non-sticky nodes ordered right->left by (x, y, name) (rank_from_right):\n" + pos_rl)

    br = _branch_hint(wf)
    if br:
        parts.append(br)
    edges = _main_edges(wf)
    if edges:
        # keep concise
        edge_str = ", ".join([f"{a}->{b}" for a, b in edges[:40]])
        parts.append("Main edges (subset): " + edge_str)
    if not parts:
        return ""
    return "\n\nHints (for interpreting positional instructions):\n" + "\n\n".join(parts) + "\n"


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
    # keep other keys but sort dict keys via dumps
    return norm


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
    # If the model omitted connections, treat as invalid for edit tasks.
    if conns is None:
        return True
    if not isinstance(conns, dict):
        return True
    for src, outputs in conns.items():
        if str(src) not in names:
            # n8n sometimes keeps connections entries for non-node keys, but for our edits this is suspicious
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


@dataclass
class EvalResult:
    ok_parse: bool
    ok_deleted_target: Optional[bool]
    ok_no_dangling: Optional[bool]
    ok_has_connections: Optional[bool]
    strict_match: bool
    relaxed_match: bool
    error_type: str


def evaluate_one(
    *,
    input_text: str,
    pred_wf: Optional[JSONDict],
    oracle_wf: JSONDict,
    delete_clue: Optional[JSONDict],
) -> EvalResult:
    if pred_wf is None:
        return EvalResult(
            ok_parse=False,
            ok_deleted_target=None,
            ok_no_dangling=None,
            ok_has_connections=None,
            strict_match=False,
            relaxed_match=False,
            error_type="parse_failed",
        )

    target_names: List[str] = []
    if isinstance(delete_clue, dict):
        edit_spec = delete_clue.get("edit_spec")
        if isinstance(edit_spec, dict):
            if edit_spec.get("target_nodes"):
                target_names = [str(n) for n in edit_spec["target_nodes"] if n]
            elif edit_spec.get("target_node"):
                target_names = [str(edit_spec["target_node"])]

    strict_match = _json_dump(normalize_workflow(pred_wf, relaxed=False)) == _json_dump(normalize_workflow(oracle_wf, relaxed=False))
    relaxed_match = _json_dump(normalize_workflow(pred_wf, relaxed=True)) == _json_dump(normalize_workflow(oracle_wf, relaxed=True))

    ok_deleted_target = None
    if target_names:
        pred_names = set(_workflow_node_names(pred_wf))
        ok_deleted_target = all(t not in pred_names for t in target_names)

    ok_has_connections = isinstance(pred_wf.get("connections"), dict)
    ok_no_dangling = not _workflow_has_dangling_connections(pred_wf)

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

    return EvalResult(
        ok_parse=True,
        ok_deleted_target=ok_deleted_target,
        ok_no_dangling=ok_no_dangling,
        ok_has_connections=ok_has_connections,
        strict_match=strict_match,
        relaxed_match=relaxed_match,
        error_type=err,
    )


def load_delete_clues(oracle_clues_path: Optional[Path]) -> Dict[str, JSONDict]:
    if not oracle_clues_path:
        return {}
    if not oracle_clues_path.exists():
        return {}
    m: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(oracle_clues_path):
        if r.get("task") != "delete":
            continue
        sha = r.get("input_sha256")
        if isinstance(sha, str) and sha:
            m[sha] = r
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument(
        "--base-url",
        type=str,
        default="",
        help="Optional OpenAI-compatible API base URL (e.g. http://host:8000/v1). If empty, uses OpenAI default.",
    )
    ap.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Env var name to read API key from (default: OPENAI_API_KEY). Use with --base-url for local servers.",
    )
    ap.add_argument("--input-jsonl", type=str, default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style/delete_testing_data.jsonl")
    ap.add_argument("--oracle-clues-jsonl", type=str, default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style/oracle_clues.jsonl")
    ap.add_argument("--out-dir", type=str, default="n8n_workflow_generator_package/outputs/delete_inference_ft")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=8000)
    ap.add_argument("--system-prompt", type=str, default="")
    ap.add_argument("--system-prompt-path", type=str, default="")
    ap.add_argument("--prepend-task-tag", action="store_true", help="Prepend [TASK=delete] to user input before sending to model")
    ap.add_argument("--task-tag", type=str, default="[TASK=delete]")
    ap.add_argument("--append-hints", action="store_true", help="Append computed hints (node order / branch paths / edges) to user input")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing out-dir by appending and skipping already-processed input_sha256.",
    )
    ap.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="When resuming, stop after N *new* samples (0 means no extra stop).",
    )
    ap.add_argument("--sleep-ms", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    clues_path = Path(args.oracle_clues_jsonl) if args.oracle_clues_jsonl else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    pred_compact_path = out_dir / "predictions_compact.jsonl"
    metrics_path = out_dir / "metrics.json"

    api_key = os.getenv(str(args.api_key_env or "OPENAI_API_KEY"))
    if not api_key:
        # Many OpenAI-compatible local servers accept any non-empty key.
        # Keep behavior strict for real OpenAI usage.
        if not args.base_url:
            raise SystemExit(f"Missing {args.api_key_env or 'OPENAI_API_KEY'} env var")
        api_key = "local-api-key"
    client = OpenAI(api_key=api_key, base_url=(args.base_url or None))

    delete_clues = load_delete_clues(clues_path)

    system_prompt = ""
    if args.system_prompt_path:
        p = Path(args.system_prompt_path)
        system_prompt = p.read_text(encoding="utf-8").strip()
    if not system_prompt:
        system_prompt = args.system_prompt.strip()
    if not system_prompt:
        system_prompt = (
            "You are an expert n8n workflow editor.\n"
            "You will be given an instruction and a Template (workflow JSON).\n"
            "Return ONLY the edited workflow JSON object.\n"
            "Do not include markdown, explanations, or additional keys.\n"
            "Preserve all nodes/parameters/connections except those required by the instruction."
        )

    counters = Counter()
    error_breakdown = Counter()
    by_branching = defaultdict(Counter)
    by_variant = defaultdict(Counter)
    done_shas: set[str] = set()

    def _write_jsonl_line(path: Path, obj: JSONDict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _ingest_existing_eval(ev: Any, input_sha: Optional[str]) -> None:
        if not isinstance(ev, dict):
            return
        err = ev.get("error_type")
        if not isinstance(err, str):
            return
        counters["n"] += 1
        counters["parse_ok"] += int(bool(ev.get("ok_parse")))
        counters["strict_ok"] += int(bool(ev.get("strict_match")))
        counters["relaxed_ok"] += int(bool(ev.get("relaxed_match")))
        if ev.get("ok_has_connections") is True:
            counters["has_connections_ok"] += 1
        if ev.get("ok_deleted_target") is True:
            counters["target_deleted_ok"] += 1
        if ev.get("ok_no_dangling") is True:
            counters["no_dangling_ok"] += 1
        error_breakdown[err] += 1

        branching_case = ev.get("branching_case")
        if isinstance(branching_case, str):
            by_branching[branching_case][err] += 1

        # Prefer oracle clue variant if available (more reliable than what's in eval).
        instr_variant = None
        if isinstance(input_sha, str) and input_sha:
            clue = delete_clues.get(input_sha)
            if isinstance(clue, dict):
                im = clue.get("instruction_meta")
                if isinstance(im, dict) and isinstance(im.get("variant"), str):
                    instr_variant = im["variant"]
        if isinstance(instr_variant, str):
            by_variant[instr_variant][err] += 1

    if args.resume and pred_path.exists():
        # Do not truncate outputs. Skip already processed samples and reuse their eval
        # to make metrics represent (existing + new) results.
        for r in _jsonl_iter_tolerant(pred_path):
            sha = r.get("input_sha256")
            if not isinstance(sha, str) or not sha:
                continue
            if sha in done_shas:
                continue
            done_shas.add(sha)
            _ingest_existing_eval(r.get("eval"), sha)
    else:
        # fresh run: reset output files
        pred_path.write_text("", encoding="utf-8")
        pred_compact_path.write_text("", encoding="utf-8")

    new_added = 0
    for rec in _jsonl_iter(in_path):
        if counters["n"] >= args.limit:
            break
        if not isinstance(rec.get("input"), str) or not isinstance(rec.get("output"), dict):
            continue

        input_text: str = rec["input"]
        oracle_wf: JSONDict = rec["output"]
        input_sha = _sha256_text(input_text)
        if args.resume and input_sha in done_shas:
            continue
        clue = delete_clues.get(input_sha)
        branching_case = clue.get("branching_case") if isinstance(clue, dict) else None
        instr_variant = None
        if isinstance(clue, dict):
            im = clue.get("instruction_meta")
            if isinstance(im, dict) and isinstance(im.get("variant"), str):
                instr_variant = im["variant"]

        user_text = input_text
        if args.prepend_task_tag:
            tag = (args.task_tag or "").strip()
            if tag:
                user_text = f"{tag}\n{input_text}"
        if args.append_hints:
            user_text = user_text + build_hints(input_text)

        pred_wf = None
        raw_text = None
        usage = None
        api_error = None

        try:
            # Try strict JSON mode first; fallback if unsupported by model.
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=args.temperature,
                    response_format={"type": "json_object"},
                    max_tokens=args.max_output_tokens,
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                )

            usage = getattr(resp, "usage", None)
            raw_text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            pred_wf = _extract_first_json_object(raw_text or "")
        except Exception as e:
            api_error = str(e)

        ev = evaluate_one(input_text=input_text, pred_wf=pred_wf, oracle_wf=oracle_wf, delete_clue=clue)
        counters["n"] += 1
        counters["parse_ok"] += int(ev.ok_parse)
        counters["strict_ok"] += int(ev.strict_match)
        counters["relaxed_ok"] += int(ev.relaxed_match)
        if ev.ok_has_connections is True:
            counters["has_connections_ok"] += 1
        if ev.ok_deleted_target is True:
            counters["target_deleted_ok"] += 1
        if ev.ok_no_dangling is True:
            counters["no_dangling_ok"] += 1
        error_breakdown[ev.error_type] += 1
        if isinstance(branching_case, str):
            by_branching[branching_case][ev.error_type] += 1
        if isinstance(instr_variant, str):
            by_variant[instr_variant][ev.error_type] += 1

        _write_jsonl_line(
            pred_path,
            {
                "input_sha256": input_sha,
                "input": input_text,
                "sent_to_model": user_text,
                "oracle_hash": _sha256_json(oracle_wf),
                "prediction": pred_wf,
                "prediction_hash": _sha256_json(pred_wf) if isinstance(pred_wf, dict) else None,
                "eval": {
                    "ok_parse": ev.ok_parse,
                    "strict_match": ev.strict_match,
                    "relaxed_match": ev.relaxed_match,
                    "ok_deleted_target": ev.ok_deleted_target,
                    "ok_no_dangling": ev.ok_no_dangling,
                    "ok_has_connections": ev.ok_has_connections,
                    "error_type": ev.error_type,
                    "branching_case": branching_case,
                    "target_node": (clue.get("edit_spec", {}).get("target_node") if isinstance(clue, dict) else None),
                },
                "raw_text": raw_text,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage.__dict__ if usage is not None else None),
                "api_error": api_error,
            },
        )

        es = clue.get("edit_spec", {}) if isinstance(clue, dict) else {}
        target_node = es.get("target_node")
        target_nodes = es.get("target_nodes") or ([target_node] if target_node else [])
        pred_names = set(_workflow_node_names(pred_wf)) if isinstance(pred_wf, dict) else set()
        raw_preview = None
        if isinstance(raw_text, str):
            raw_preview = raw_text[:800]

        _write_jsonl_line(
            pred_compact_path,
            {
                "input_sha256": input_sha,
                "branching_case": branching_case,
                "instruction_variant": instr_variant,
                "target_node": target_node,
                "target_nodes": target_nodes if len(target_nodes) != 1 else None,
                "error_type": ev.error_type,
                "ok_parse": ev.ok_parse,
                "strict_match": ev.strict_match,
                "relaxed_match": ev.relaxed_match,
                "ok_deleted_target": ev.ok_deleted_target,
                "ok_no_dangling": ev.ok_no_dangling,
                "ok_has_connections": ev.ok_has_connections,
                "prediction_shape": sorted(list(pred_wf.keys()))[:50] if isinstance(pred_wf, dict) else None,
                "prediction_nodes_count": (len(pred_wf.get("nodes", [])) if isinstance(pred_wf, dict) and isinstance(pred_wf.get("nodes"), list) else None),
                "prediction_has_target": (any(t in pred_names for t in target_nodes) if target_nodes else None),
                "raw_preview": raw_preview,
                "api_error": api_error,
            },
        )

        done_shas.add(input_sha)
        new_added += 1
        if args.stop_after and args.stop_after > 0 and new_added >= args.stop_after:
            break
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    metrics: JSONDict = {
        "model": args.model,
        "base_url": args.base_url or None,
        "input_jsonl": str(in_path),
        "oracle_clues_jsonl": str(clues_path) if clues_path else None,
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
        "predictions_jsonl": str(pred_path),
        "predictions_compact_jsonl": str(pred_compact_path),
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

