#!/usr/bin/env python3
"""
Run OpenAI-compatible chat inference on insert-style JSONL, then evaluate.

Training-style rows:
  {"input": str, "output": dict | str}
  - output dict: oracle full workflow after insert
  - output str: oracle clarifying question for missing parameters

Uses the same positional hints as delete when --append-hints (build_hints from run_delete_inference_and_eval).

Oracle clues: build with build_insert_oracle_clues.py (insert_eval_oracle_clues.jsonl).

Workflow-oracle grading sets ``gold_params_subset_ok`` when the inserted node's ``parameters`` cover
every key path from the gold node (extra keys allowed). Values are compared after light normalization
(JSON strings parsed, ``true``/``false`` strings as booleans, blank strings as empty). If a gold key's
value is empty after normalization (``null`` / ``{}`` / ``[]``), the prediction may omit that key or
replace a null leaf with any value (oracle null is treated as unconstrained for that path).
Insert correctness: the inserted **name** must exist; **type** must match oracle; **main** incoming /
outgoing neighbor multisets for that name must match oracle (splice position). Then ``ok_no_dangling``
must hold. If all hold and parameters cover gold, ``error_type`` is ``ok_insert_params_cover_gold``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from run_delete_inference_and_eval import (  # noqa: E402
    _extract_first_json_object,
    _json_dump,
    _jsonl_iter,
    _jsonl_iter_tolerant,
    _sha256_json,
    _sha256_text,
    _workflow_has_dangling_connections,
    _workflow_node_names,
    build_hints,
    normalize_workflow,
)

JSONDict = Dict[str, Any]


def _load_insert_clues(path: Optional[Path]) -> Dict[str, JSONDict]:
    if not path or not path.exists():
        return {}
    m: Dict[str, JSONDict] = {}
    for r in _jsonl_iter(path):
        if r.get("task") != "insert":
            continue
        sha = r.get("input_sha256")
        if isinstance(sha, str) and sha:
            m[sha] = r
    return m


def _token_set(s: str) -> set:
    return {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_-]{2,}", s)}


def _workflow_node_by_name(wf: JSONDict, name: str) -> Optional[JSONDict]:
    for n in wf.get("nodes") or []:
        if isinstance(n, dict) and n.get("name") == name:
            return n
    return None


def _main_outgoing_targets(wf: JSONDict, source: str) -> List[str]:
    conns = wf.get("connections")
    if not isinstance(conns, dict):
        return []
    block = conns.get(source)
    if not isinstance(block, dict):
        return []
    main = block.get("main")
    if not isinstance(main, list):
        return []
    acc: List[str] = []
    for group in main:
        if not isinstance(group, list):
            continue
        for link in group:
            if isinstance(link, dict) and link.get("node"):
                acc.append(str(link["node"]))
    return acc


def _main_incoming_sources(wf: JSONDict, target: str) -> List[str]:
    conns = wf.get("connections")
    if not isinstance(conns, dict):
        return []
    srcs: List[str] = []
    for src, block in conns.items():
        if not isinstance(block, dict):
            continue
        main = block.get("main")
        if not isinstance(main, list):
            continue
        for group in main:
            if not isinstance(group, list):
                continue
            for link in group:
                if isinstance(link, dict) and str(link.get("node")) == target:
                    srcs.append(str(src))
    return srcs


def _main_neighbor_signature(wf: JSONDict, node: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Sorted main-edge neighbors (incoming sources, outgoing targets) for multiset comparison."""
    inc = _main_incoming_sources(wf, node)
    out = _main_outgoing_targets(wf, node)
    return (tuple(sorted(inc)), tuple(sorted(out)))


def _is_vacuous_cover_value(v: Any) -> bool:
    """After normalization, these impose no constraint for an optional / empty gold field."""
    return v is None or v == {} or v == []


def _normalize_for_params_cover(v: Any) -> Any:
    if isinstance(v, bool):  # must run before int (bool is a subclass of int)
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        low = s.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return _normalize_for_params_cover(json.loads(s))
            except (json.JSONDecodeError, TypeError, ValueError):
                return v
        return v
    if isinstance(v, list):
        return [_normalize_for_params_cover(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _normalize_for_params_cover(val) for k, val in v.items()}
    return v


def _pred_parameters_cover_gold(gold: Any, pred: Any) -> bool:
    """
    True if ``pred`` matches ``gold`` at every path in ``gold`` after normalization (see module doc).
    Dicts recurse; missing keys in ``pred`` are allowed when the gold value normalizes to empty.
    Lists require len(pred) >= len(gold) and each indexed gold[i] is covered by pred[i].
    Extra keys / trailing list elements in ``pred`` are allowed.
    """
    g = _normalize_for_params_cover(gold)
    p = _normalize_for_params_cover(pred)
    if g == p:
        return True
    # Oracle often stores null for "unset"; the model may omit the key or use a placeholder expression.
    if g is None:
        return True
    if isinstance(g, dict) and isinstance(p, dict):
        for k, gv in g.items():
            gkn = _normalize_for_params_cover(gv)
            if k not in p:
                if _is_vacuous_cover_value(gkn):
                    continue
                return False
            if not _pred_parameters_cover_gold(gv, p[k]):
                return False
        return True
    if isinstance(g, list) and isinstance(p, list):
        if len(p) < len(g):
            return False
        for i in range(len(g)):
            if not _pred_parameters_cover_gold(g[i], p[i]):
                return False
        return True
    return False


@dataclass
class InsertEvalResult:
    ok_kind: bool
    ok_parse_wf: Optional[bool]
    strict_match: bool
    relaxed_match: bool
    # Name present + type matches oracle (Phase 1 / declared-type success for the splice node).
    inserted_node_ok: Optional[bool]
    inserted_node_name_ok: Optional[bool]
    inserted_node_type_ok: Optional[bool]
    # main incoming/outgoing neighbor multiset matches oracle for the inserted node.
    insert_main_neighbors_ok: Optional[bool]
    gold_params_subset_ok: Optional[bool]
    ok_no_dangling: Optional[bool]
    ok_has_connections: Optional[bool]
    ask_token_recall: Optional[float]
    error_type: str

    def to_json(self) -> JSONDict:
        d = asdict(self)
        return d


def _pred_is_workflow_json_response(raw: Optional[str], pred_wf: Optional[JSONDict]) -> bool:
    if pred_wf is None or not raw:
        return False
    t = raw.strip()
    if not t.startswith("{"):
        return False
    # If extract got a dict and body is mostly that JSON, treat as workflow response
    try:
        dumped = json.dumps(pred_wf, ensure_ascii=False)
        return len(t) <= len(dumped) * 1.05 + 20
    except Exception:
        return True


def evaluate_insert(
    *,
    oracle_out: Any,
    clue: JSONDict,
    pred_raw: Optional[str],
) -> InsertEvalResult:
    oracle_ask = clue.get("output_kind") == "ask"
    pred_wf = _extract_first_json_object(pred_raw or "")
    inserted = str(clue.get("inserted_node_name") or "")

    if oracle_ask:
        got_wf = _pred_is_workflow_json_response(pred_raw, pred_wf)
        if got_wf:
            return InsertEvalResult(
                ok_kind=False,
                ok_parse_wf=None,
                strict_match=False,
                relaxed_match=False,
                inserted_node_ok=None,
                inserted_node_name_ok=None,
                inserted_node_type_ok=None,
                insert_main_neighbors_ok=None,
                gold_params_subset_ok=None,
                ok_no_dangling=None,
                ok_has_connections=None,
                ask_token_recall=None,
                error_type="ask_got_workflow_json",
            )
        ot = _token_set(str(oracle_out))
        pt = _token_set(pred_raw or "")
        recall = (len(ot & pt) / len(ot)) if ot else 0.0
        raw = (pred_raw or "").strip()
        heuristic_ok = len(raw) > 30 and (
            "?" in raw or "parameter" in raw.lower() or "provide" in raw.lower() or "value" in raw.lower()
        )
        ok_kind = heuristic_ok or recall >= 0.08
        err = "ok_ask" if ok_kind else "ask_weak"
        if recall >= 0.15 and heuristic_ok:
            err = "ok_ask"
        elif recall >= 0.12:
            err = "ok_ask"
        return InsertEvalResult(
            ok_kind=ok_kind,
            ok_parse_wf=False,
            strict_match=False,
            relaxed_match=False,
            inserted_node_ok=None,
            inserted_node_name_ok=None,
            inserted_node_type_ok=None,
            insert_main_neighbors_ok=None,
            gold_params_subset_ok=None,
            ok_no_dangling=None,
            ok_has_connections=None,
            ask_token_recall=recall,
            error_type=err,
        )

    # oracle workflow
    if not isinstance(oracle_out, dict):
        return InsertEvalResult(
            ok_kind=True,
            ok_parse_wf=False,
            strict_match=False,
            relaxed_match=False,
            inserted_node_ok=None,
            inserted_node_name_ok=None,
            inserted_node_type_ok=None,
            insert_main_neighbors_ok=None,
            gold_params_subset_ok=None,
            ok_no_dangling=None,
            ok_has_connections=None,
            ask_token_recall=None,
            error_type="oracle_not_dict",
        )

    if pred_wf is None:
        return InsertEvalResult(
            ok_kind=True,
            ok_parse_wf=False,
            strict_match=False,
            relaxed_match=False,
            inserted_node_ok=None,
            inserted_node_name_ok=None,
            inserted_node_type_ok=None,
            insert_main_neighbors_ok=None,
            gold_params_subset_ok=None,
            ok_no_dangling=None,
            ok_has_connections=None,
            ask_token_recall=None,
            error_type="parse_failed",
        )

    strict_match = _json_dump(normalize_workflow(pred_wf, relaxed=False)) == _json_dump(
        normalize_workflow(oracle_out, relaxed=False)
    )
    relaxed_match = _json_dump(normalize_workflow(pred_wf, relaxed=True)) == _json_dump(
        normalize_workflow(oracle_out, relaxed=True)
    )

    names = set(_workflow_node_names(pred_wf))
    inserted_node_name_ok = bool(inserted) and inserted in names
    on = _workflow_node_by_name(oracle_out, inserted) if inserted else None
    pn = _workflow_node_by_name(pred_wf, inserted) if inserted_node_name_ok else None
    inserted_node_type_ok: Optional[bool] = None
    if inserted_node_name_ok and isinstance(on, dict) and isinstance(pn, dict):
        inserted_node_type_ok = on.get("type") == pn.get("type")
    else:
        inserted_node_type_ok = False
    inserted_node_ok = bool(inserted_node_name_ok and inserted_node_type_ok)

    insert_main_neighbors_ok: Optional[bool] = None
    if not inserted_node_name_ok:
        insert_main_neighbors_ok = False
    else:
        insert_main_neighbors_ok = _main_neighbor_signature(oracle_out, inserted) == _main_neighbor_signature(
            pred_wf, inserted
        )

    ok_has_connections = isinstance(pred_wf.get("connections"), dict)
    ok_no_dangling = not _workflow_has_dangling_connections(pred_wf)

    gold_params_subset_ok: Optional[bool] = None
    if inserted and inserted_node_name_ok and isinstance(on, dict) and isinstance(pn, dict):
        gp = on.get("parameters")
        pp = pn.get("parameters")
        if isinstance(gp, dict):
            gold_params_subset_ok = (
                _pred_parameters_cover_gold(gp, pp) if isinstance(pp, dict) else False
            )
        else:
            gold_params_subset_ok = True

    if strict_match:
        err = "ok_strict"
    elif relaxed_match:
        err = "ok_relaxed"
    elif not ok_has_connections:
        err = "missing_connections"
    elif not inserted_node_name_ok:
        err = "inserted_node_missing"
    elif not inserted_node_type_ok:
        err = "inserted_type_mismatch"
    elif not insert_main_neighbors_ok:
        err = "insert_splice_mismatch"
    elif not ok_no_dangling:
        err = "dangling_connections"
    elif gold_params_subset_ok is True:
        err = "ok_insert_params_cover_gold"
    else:
        err = "mismatch_other"

    return InsertEvalResult(
        ok_kind=True,
        ok_parse_wf=True,
        strict_match=strict_match,
        relaxed_match=relaxed_match,
        inserted_node_ok=inserted_node_ok,
        inserted_node_name_ok=inserted_node_name_ok,
        inserted_node_type_ok=inserted_node_type_ok,
        insert_main_neighbors_ok=insert_main_neighbors_ok,
        gold_params_subset_ok=gold_params_subset_ok,
        ok_no_dangling=ok_no_dangling,
        ok_has_connections=ok_has_connections,
        ask_token_recall=None,
        error_type=err,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--base-url", type=str, default="")
    ap.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    ap.add_argument(
        "--input-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_training_data_ask.jsonl",
    )
    ap.add_argument(
        "--oracle-clues-jsonl",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_eval_oracle_clues.jsonl",
    )
    ap.add_argument("--out-dir", type=str, default="n8n_workflow_generator_package/outputs/insert_inference_eval")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=16000)
    ap.add_argument("--system-prompt", type=str, default="")
    ap.add_argument("--system-prompt-path", type=str, default="")
    ap.add_argument(
        "--append-hints",
        action="store_true",
        help="Append node-order / edge hints (same as delete eval).",
    )
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
            raise SystemExit(f"Missing {args.api_key_env} env var")
        api_key = "local-api-key"
    client = OpenAI(api_key=api_key, base_url=(args.base_url or None))

    clues = _load_insert_clues(clues_path)

    system_prompt = ""
    if args.system_prompt_path:
        system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8").strip()
    if not system_prompt:
        system_prompt = args.system_prompt.strip()
    if not system_prompt:
        system_prompt = (
            "You are an expert n8n workflow editor for INSERT tasks.\n"
            "- If the user provides enough information (including parameters), return ONLY the full updated workflow as one JSON object (nodes + connections).\n"
            "- If required parameter values are missing and cannot be inferred from defaults, respond with a clear question listing what you need (plain text). Do not return workflow JSON in that case.\n"
            "Never wrap JSON in markdown fences. No extra commentary when returning JSON."
        )

    counters: Counter = Counter()
    error_breakdown: Counter = Counter()
    by_variant: Dict[str, Counter] = defaultdict(Counter)
    done_shas: set[str] = set()

    def _write(path: Path, obj: JSONDict) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _ingest(ev: Any) -> None:
        if not isinstance(ev, dict):
            return
        err = ev.get("error_type")
        if not isinstance(err, str):
            return
        counters["n"] += 1
        counters["kind_ok"] += int(bool(ev.get("ok_kind")))
        counters["strict_ok"] += int(bool(ev.get("strict_match")))
        counters["relaxed_ok"] += int(bool(ev.get("relaxed_match")))
        if ev.get("ok_parse_wf") is True:
            counters["parse_wf_ok"] += 1
        if ev.get("inserted_node_ok") is True:
            counters["inserted_node_ok"] += 1
        if ev.get("gold_params_subset_ok") is True or ev.get("ok_insert_params_cover_gold") is True:
            counters["params_cover_ok"] += 1
        if ev.get("ask_token_recall") is not None:
            counters["ask_recall_sum"] += float(ev["ask_token_recall"])
            counters["ask_recall_n"] += 1
        error_breakdown[err] += 1
        iv = ev.get("insert_variant")
        if isinstance(iv, str):
            by_variant[iv][err] += 1

    if args.resume and pred_path.exists():
        for r in _jsonl_iter_tolerant(pred_path):
            sha = r.get("input_sha256")
            if isinstance(sha, str) and sha:
                done_shas.add(sha)
            _ingest(r.get("eval"))
    else:
        pred_path.write_text("", encoding="utf-8")
        pred_compact_path.write_text("", encoding="utf-8")

    lim = int(args.limit or 0)
    new_n = 0

    for rec in _jsonl_iter(in_path):
        if lim and counters["n"] >= lim:
            break
        if not isinstance(rec.get("input"), str) or "output" not in rec:
            continue
        input_text: str = rec["input"]
        oracle_out = rec["output"]
        input_sha = _sha256_text(input_text)
        if args.resume and input_sha in done_shas:
            continue

        clue: JSONDict = dict(clues.get(input_sha) or {})
        if not clue.get("output_kind"):
            clue["output_kind"] = "ask" if isinstance(oracle_out, str) else "workflow"
        if not clue.get("inserted_node_name"):
            m = re.match(r'^Insert the node "([^"]+)"', input_text.strip())
            if m:
                clue["inserted_node_name"] = m.group(1)
        insert_variant = clue.get("insert_variant")

        user_text = input_text
        if args.append_hints:
            h = build_hints(input_text)
            if h:
                user_text = user_text.rstrip() + "\n" + h

        pred_raw = None
        api_error = None
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=args.temperature,
                max_tokens=args.max_output_tokens,
            )
            pred_raw = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            usage = getattr(resp, "usage", None)
        except Exception as e:
            api_error = str(e)
            usage = None

        ev = evaluate_insert(oracle_out=oracle_out, clue=clue, pred_raw=pred_raw)
        ej = ev.to_json()
        ej["insert_variant"] = insert_variant

        pred_wf = _extract_first_json_object(pred_raw or "")

        counters["n"] += 1
        counters["kind_ok"] += int(ev.ok_kind)
        counters["strict_ok"] += int(ev.strict_match)
        counters["relaxed_ok"] += int(ev.relaxed_match)
        if ev.ok_parse_wf is True:
            counters["parse_wf_ok"] += 1
        if ev.inserted_node_ok is True:
            counters["inserted_node_ok"] += 1
        if ev.gold_params_subset_ok is True:
            counters["params_cover_ok"] += 1
        if ev.ask_token_recall is not None:
            counters["ask_recall_sum"] += float(ev.ask_token_recall)
            counters["ask_recall_n"] += 1
        error_breakdown[ev.error_type] += 1
        if isinstance(insert_variant, str):
            by_variant[insert_variant][ev.error_type] += 1

        _write(
            pred_path,
            {
                "input_sha256": input_sha,
                "input": input_text,
                "sent_to_model": user_text,
                "oracle_out_kind": "ask" if isinstance(oracle_out, str) else "workflow",
                "prediction": pred_wf,
                "eval": ej,
                "raw_text": pred_raw,
                "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
                "api_error": api_error,
            },
        )

        _write(
            pred_compact_path,
            {
                "input_sha256": input_sha,
                "insert_variant": insert_variant,
                "error_type": ev.error_type,
                "ok_kind": ev.ok_kind,
                "strict_match": ev.strict_match,
                "relaxed_match": ev.relaxed_match,
                "ask_token_recall": ev.ask_token_recall,
                "api_error": api_error,
            },
        )

        done_shas.add(input_sha)
        new_n += 1
        if args.stop_after and new_n >= args.stop_after:
            break
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    n = int(counters["n"])
    metrics = {
        "model": args.model,
        "input_jsonl": str(in_path),
        "oracle_clues_jsonl": str(clues_path) if clues_path else None,
        "append_hints": bool(args.append_hints),
        "n": n,
        "kind_ok": int(counters["kind_ok"]),
        "strict_ok": int(counters["strict_ok"]),
        "relaxed_ok": int(counters["relaxed_ok"]),
        "parse_wf_ok": int(counters["parse_wf_ok"]),
        "inserted_node_ok": int(counters["inserted_node_ok"]),
        "params_cover_ok": int(counters["params_cover_ok"]),
        "rates": {
            "kind_ok": (counters["kind_ok"] / n) if n else 0.0,
            "strict_ok": (counters["strict_ok"] / n) if n else 0.0,
            "relaxed_ok": (counters["relaxed_ok"] / n) if n else 0.0,
            "parse_wf_ok": (counters["parse_wf_ok"] / n) if n else 0.0,
            "inserted_node_ok": (counters["inserted_node_ok"] / n) if n else 0.0,
            "params_cover_ok": (counters["params_cover_ok"] / n) if n else 0.0,
            "ask_token_recall_mean": (
                (counters["ask_recall_sum"] / counters["ask_recall_n"]) if counters["ask_recall_n"] else None
            ),
        },
        "error_breakdown": dict(error_breakdown),
        "error_by_insert_variant": {k: dict(v) for k, v in by_variant.items()},
        "predictions_jsonl": str(pred_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {pred_path}\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
