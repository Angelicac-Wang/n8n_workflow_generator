#!/usr/bin/env python3
"""
Run inference on modify_param test set (parameter-only edits), then evaluate.

Input dataset format (training-style JSONL):
  {"input": "[TASK=modify]\\n...\\n\\nTemplate:\\n{workflow_json}", "output": <oracle_workflow_obj>}

Oracle clues (modify_oracle_clues.jsonl):
  {"task":"modify_param", "input_sha256": "...", "edit_spec": {node_name, json_pointer, old_value, new_value, ...}}

Primary metrics (see metrics.json):
  - phrase_ok: output parses as workflow-shaped JSON (nodes list + connections dict)
  - param_updated_ok: value at json_pointer on node_name equals oracle new_value
  - consistent_ok: vs Template in user message — same nodes/connections; only the target
    parameter path may differ from the template (no other node or field drift)
  - error_breakdown: primary error_type counts for diagnosis

Also recorded: strict_match / relaxed_match vs oracle, no_dangling_ok, has_connections_ok.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    if not text:
        return None
    text = text.strip()
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
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
                cleaned.sort(
                    key=lambda d: (str(d.get("node")), str(d.get("type")), int(d.get("index") or 0))
                )
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


def _get_by_pointer(obj: Any, ptr: str) -> Any:
    """Read value at JSON Pointer (RFC 6901-style with leading /)."""
    if not ptr or ptr == "/":
        raise ValueError("invalid pointer")
    parts = ptr.lstrip("/").split("/")
    cur = obj
    for p in parts:
        p = p.replace("~1", "/").replace("~0", "~")
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def _extract_template_workflow_from_input(input_text: str) -> Optional[JSONDict]:
    """
    User message ends with Template + workflow JSON.
    Accepts real newlines or literal backslash-n (some JSONL dumps double-escape newlines in strings).
    """
    m = re.search(r"(?:\n\n|\\n\\n)Template:(?:\n|\\n)", input_text)
    if not m:
        return None
    tail = input_text[m.end() :].strip()
    try:
        obj = json.loads(tail)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _phrase_ok(wf: Any) -> bool:
    """Valid workflow-shaped JSON (not necessarily semantically correct n8n)."""
    return isinstance(wf, dict) and isinstance(wf.get("nodes"), list) and isinstance(wf.get("connections"), dict)


def _pointer_parts(ptr: str) -> List[str]:
    if not ptr or ptr == "/":
        return []
    return [p.replace("~1", "/").replace("~0", "~") for p in ptr.lstrip("/").split("/") if p != ""]


def _params_equal_except_pointer(template_p: Any, pred_p: Any, ptr: str) -> bool:
    """
    template_p / pred_p are the node's parameters objects.
    All keys and values must match recursively except the value at json_pointer (relative to parameters).
    """
    parts = _pointer_parts(ptr)
    if not parts:
        return _json_dump(template_p) == _json_dump(pred_p)

    def rec(a: Any, b: Any, depth: int) -> bool:
        if depth == len(parts) - 1:
            k = parts[depth]
            if isinstance(a, dict) and isinstance(b, dict):
                if set(a.keys()) != set(b.keys()):
                    return False
                for key in a:
                    if key == k:
                        if key not in b:
                            return False
                        continue
                    if _json_dump(a[key]) != _json_dump(b[key]):
                        return False
                return True
            if isinstance(a, list) and isinstance(b, list):
                idx = int(k)
                if len(a) != len(b):
                    return False
                for i in range(len(a)):
                    if i == idx:
                        continue
                    if _json_dump(a[i]) != _json_dump(b[i]):
                        return False
                return 0 <= idx < len(a)
            return False
        k = parts[depth]
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            for key in a:
                if key != k and _json_dump(a[key]) != _json_dump(b[key]):
                    return False
            if k not in a or k not in b:
                return False
            return rec(a[k], b[k], depth + 1)
        if isinstance(a, list) and isinstance(b, list):
            idx = int(k)
            if len(a) != len(b) or idx < 0 or idx >= len(a):
                return False
            for i in range(len(a)):
                if i != idx and _json_dump(a[i]) != _json_dump(b[i]):
                    return False
            return rec(a[idx], b[idx], depth + 1)
        return False

    return rec(template_p, pred_p, 0)


def _node_for_compare(wf: JSONDict, name: str) -> Optional[JSONDict]:
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if isinstance(n, dict) and n.get("name") == name:
            return n
    return None


def _node_body_json(n: JSONDict) -> str:
    d = {k: v for k, v in n.items() if k not in ("id", "position")}
    return _json_dump(d)


def _consistent_ok(
    pred_wf: JSONDict,
    template_wf: JSONDict,
    modify_clue: Optional[JSONDict],
) -> Optional[bool]:
    """
    True if pred matches template except possibly the single parameter at edit_spec's json_pointer
    on edit_spec's node_name. Same node names, same connections (normalized), other nodes unchanged.
    """
    if not isinstance(modify_clue, dict):
        return None
    es = modify_clue.get("edit_spec")
    if not isinstance(es, dict) or es.get("action") != "modify_param":
        return None
    node_name = es.get("node_name")
    ptr = es.get("json_pointer")
    if not isinstance(node_name, str) or not isinstance(ptr, str):
        return None

    t_nodes = template_wf.get("nodes")
    p_nodes = pred_wf.get("nodes")
    if not isinstance(t_nodes, list) or not isinstance(p_nodes, list):
        return False
    t_names = {str(n.get("name")) for n in t_nodes if isinstance(n, dict) and n.get("name")}
    p_names = {str(n.get("name")) for n in p_nodes if isinstance(n, dict) and n.get("name")}
    if t_names != p_names:
        return False

    t_con = _normalize_connections(template_wf.get("connections"))
    p_con = _normalize_connections(pred_wf.get("connections"))
    if _json_dump(t_con) != _json_dump(p_con):
        return False

    for tn in t_nodes:
        if not isinstance(tn, dict) or not tn.get("name"):
            continue
        name = str(tn["name"])
        pn = _node_for_compare(pred_wf, name)
        if not isinstance(pn, dict):
            return False
        if name != node_name:
            if _node_body_json(tn) != _node_body_json(pn):
                return False
            continue
        tp = tn.get("parameters")
        pp = pn.get("parameters")
        if not isinstance(tp, dict) or not isinstance(pp, dict):
            return False
        if not _params_equal_except_pointer(tp, pp, ptr):
            return False
    return True


def _param_value_for_edit(wf: JSONDict, node_name: str, json_pointer: str) -> Any:
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if not isinstance(n, dict) or n.get("name") != node_name:
            continue
        params = n.get("parameters")
        if not isinstance(params, dict):
            return None
        try:
            return _get_by_pointer(params, json_pointer)
        except Exception:
            return None
    return None


def _values_equal(a: Any, b: Any) -> bool:
    return _json_dump(a) == _json_dump(b)


@dataclass
class EvalResult:
    phrase_ok: bool
    ok_param_updated: Optional[bool]
    ok_consistent: Optional[bool]
    ok_no_dangling: Optional[bool]
    ok_has_connections: Optional[bool]
    strict_match: bool
    relaxed_match: bool
    error_type: str


def evaluate_one(
    *,
    pred_wf: Optional[JSONDict],
    oracle_wf: JSONDict,
    modify_clue: Optional[JSONDict],
    input_text: str,
) -> EvalResult:
    if pred_wf is None or not _phrase_ok(pred_wf):
        return EvalResult(
            phrase_ok=False,
            ok_param_updated=None,
            ok_consistent=None,
            ok_no_dangling=None,
            ok_has_connections=None,
            strict_match=False,
            relaxed_match=False,
            error_type="parse_failed",
        )

    strict_match = _json_dump(normalize_workflow(pred_wf, relaxed=False)) == _json_dump(
        normalize_workflow(oracle_wf, relaxed=False)
    )
    relaxed_match = _json_dump(normalize_workflow(pred_wf, relaxed=True)) == _json_dump(
        normalize_workflow(oracle_wf, relaxed=True)
    )

    ok_has_connections = isinstance(pred_wf.get("connections"), dict)
    ok_no_dangling = not _workflow_has_dangling_connections(pred_wf)

    ok_param_updated: Optional[bool] = None
    if isinstance(modify_clue, dict):
        es = modify_clue.get("edit_spec")
        if isinstance(es, dict) and es.get("action") == "modify_param":
            node_name = es.get("node_name")
            ptr = es.get("json_pointer")
            new_val = es.get("new_value")
            if isinstance(node_name, str) and isinstance(ptr, str):
                got = _param_value_for_edit(pred_wf, node_name, ptr)
                ok_param_updated = _values_equal(got, new_val)

    template_wf = _extract_template_workflow_from_input(input_text)
    ok_consistent: Optional[bool] = None
    if isinstance(template_wf, dict):
        ok_consistent = _consistent_ok(pred_wf, template_wf, modify_clue)

    # Primary error label: phrase → consistency → param → graph → oracle match
    if strict_match:
        err = "ok_strict"
    elif relaxed_match:
        err = "ok_relaxed"
    elif ok_consistent is False:
        err = "inconsistent_template"
    elif ok_param_updated is False:
        err = "param_not_updated"
    elif not ok_has_connections:
        err = "missing_connections"
    elif ok_no_dangling is False:
        err = "dangling_connections"
    else:
        err = "mismatch_other"

    return EvalResult(
        phrase_ok=True,
        ok_param_updated=ok_param_updated,
        ok_consistent=ok_consistent,
        ok_no_dangling=ok_no_dangling,
        ok_has_connections=ok_has_connections,
        strict_match=strict_match,
        relaxed_match=relaxed_match,
        error_type=err,
    )


def load_modify_clues(oracle_clues_path: Optional[Path]) -> Dict[str, JSONDict]:
    if not oracle_clues_path or not oracle_clues_path.exists():
        return {}
    m: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(oracle_clues_path):
        if r.get("task") != "modify_param":
            continue
        sha = r.get("input_sha256")
        if isinstance(sha, str) and sha:
            m[sha] = r
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base-url", type=str, default="")
    ap.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    ap.add_argument(
        "--input-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_testing_data.jsonl",
    )
    ap.add_argument(
        "--oracle-clues-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/edit_eval_testing_data/modify_oracle_clues.jsonl",
    )
    ap.add_argument("--out-dir", type=str, default="n8n_workflow_generator_package/outputs/modify_inference_ft")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--system-prompt", type=str, default="")
    ap.add_argument("--system-prompt-path", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0)
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
        if not args.base_url:
            raise SystemExit(f"Missing {args.api_key_env or 'OPENAI_API_KEY'} env var")
        api_key = "local-api-key"
    client = OpenAI(api_key=api_key, base_url=(args.base_url or None))

    modify_clues = load_modify_clues(clues_path)

    system_prompt = ""
    if args.system_prompt_path:
        system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8").strip()
        system_prompt = (
            system_prompt.replace("{{ $json.max_new_tokens }}", str(args.max_output_tokens))
            .replace("{{$json.max_new_tokens}}", str(args.max_output_tokens))
        )
    if not system_prompt:
        system_prompt = args.system_prompt.strip()
    if not system_prompt:
        system_prompt = (
            "You are an expert n8n workflow editor.\n"
            "You will be given an instruction to update a node's parameter and a Template (workflow JSON).\n"
            "Return ONLY the full updated workflow JSON object.\n"
            "Do not include markdown fences, explanations, or text outside the JSON.\n"
            "Preserve all nodes except apply the parameter change described in the instruction."
        )

    counters = Counter()
    error_breakdown = Counter()
    done_shas: set[str] = set()

    def _write_jsonl_line(path: Path, obj: JSONDict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _ingest_existing_eval(ev: Any, _sha: Optional[str]) -> None:
        if not isinstance(ev, dict):
            return
        err = ev.get("error_type")
        if not isinstance(err, str):
            return
        counters["n"] += 1
        phrase = bool(ev.get("phrase_ok", ev.get("ok_parse")))
        counters["phrase_ok"] += int(phrase)
        counters["strict_ok"] += int(bool(ev.get("strict_match")))
        counters["relaxed_ok"] += int(bool(ev.get("relaxed_match")))
        if ev.get("ok_has_connections") is True:
            counters["has_connections_ok"] += 1
        if ev.get("ok_param_updated") is True:
            counters["param_updated_ok"] += 1
        if ev.get("ok_consistent") is True:
            counters["consistent_ok"] += 1
        if ev.get("ok_no_dangling") is True:
            counters["no_dangling_ok"] += 1
        error_breakdown[err] += 1

    if args.resume and pred_path.exists():
        for r in _jsonl_iter_tolerant(pred_path):
            sha = r.get("input_sha256")
            if not isinstance(sha, str) or not sha:
                continue
            if sha in done_shas:
                continue
            done_shas.add(sha)
            _ingest_existing_eval(r.get("eval"), sha)
    else:
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

        clue = modify_clues.get(input_sha)
        es = clue.get("edit_spec", {}) if isinstance(clue, dict) else {}

        pred_wf = None
        raw_text = None
        usage = None
        api_error = None

        try:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": input_text},
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
                        {"role": "user", "content": input_text},
                    ],
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                )

            usage = getattr(resp, "usage", None)
            raw_text = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            pred_wf = _extract_first_json_object(raw_text or "")
        except Exception as e:
            api_error = str(e)

        if api_error:
            low = api_error.lower()
            err_type = (
                "model_not_found"
                if ("model_not_found" in low or "does not exist" in low or "you do not have access" in low)
                and "model" in low
                else "api_error"
            )
            ev = EvalResult(
                phrase_ok=False,
                ok_param_updated=None,
                ok_consistent=None,
                ok_no_dangling=None,
                ok_has_connections=None,
                strict_match=False,
                relaxed_match=False,
                error_type=err_type,
            )
        else:
            ev = evaluate_one(
                pred_wf=pred_wf,
                oracle_wf=oracle_wf,
                modify_clue=clue,
                input_text=input_text,
            )
        counters["n"] += 1
        counters["phrase_ok"] += int(ev.phrase_ok)
        counters["strict_ok"] += int(ev.strict_match)
        counters["relaxed_ok"] += int(ev.relaxed_match)
        if ev.ok_has_connections is True:
            counters["has_connections_ok"] += 1
        if ev.ok_param_updated is True:
            counters["param_updated_ok"] += 1
        if ev.ok_consistent is True:
            counters["consistent_ok"] += 1
        if ev.ok_no_dangling is True:
            counters["no_dangling_ok"] += 1
        error_breakdown[ev.error_type] += 1

        _write_jsonl_line(
            pred_path,
            {
                "input_sha256": input_sha,
                "input": input_text,
                "oracle_hash": _sha256_json(oracle_wf),
                "prediction": pred_wf,
                "prediction_hash": _sha256_json(pred_wf) if isinstance(pred_wf, dict) else None,
                "eval": {
                    "phrase_ok": ev.phrase_ok,
                    "ok_parse": ev.phrase_ok,
                    "ok_consistent": ev.ok_consistent,
                    "strict_match": ev.strict_match,
                    "relaxed_match": ev.relaxed_match,
                    "ok_param_updated": ev.ok_param_updated,
                    "ok_no_dangling": ev.ok_no_dangling,
                    "ok_has_connections": ev.ok_has_connections,
                    "error_type": ev.error_type,
                    "node_name": es.get("node_name"),
                    "json_pointer": es.get("json_pointer"),
                },
                "raw_text": raw_text,
                "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage.__dict__ if usage is not None else None),
                "api_error": api_error,
            },
        )

        raw_preview = raw_text[:800] if isinstance(raw_text, str) else None
        _write_jsonl_line(
            pred_compact_path,
            {
                "input_sha256": input_sha,
                "error_type": ev.error_type,
                "phrase_ok": ev.phrase_ok,
                "ok_consistent": ev.ok_consistent,
                "strict_match": ev.strict_match,
                "relaxed_match": ev.relaxed_match,
                "ok_param_updated": ev.ok_param_updated,
                "ok_no_dangling": ev.ok_no_dangling,
                "ok_has_connections": ev.ok_has_connections,
                "node_name": es.get("node_name"),
                "json_pointer": es.get("json_pointer"),
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

    n = int(counters["n"])
    metrics: JSONDict = {
        "model": args.model,
        "base_url": args.base_url or None,
        "input_jsonl": str(in_path),
        "oracle_clues_jsonl": str(clues_path) if clues_path else None,
        "n": n,
        "phrase_ok": int(counters["phrase_ok"]),
        "param_updated_ok": int(counters["param_updated_ok"]),
        "consistent_ok": int(counters["consistent_ok"]),
        "error_breakdown": dict(error_breakdown),
        "note": "If error_breakdown is all model_not_found/api_error, the run never received model text — fix --model, API key org, or base URL before interpreting phrase_ok.",
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
        "predictions_jsonl": str(pred_path),
        "predictions_compact_jsonl": str(pred_compact_path),
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
