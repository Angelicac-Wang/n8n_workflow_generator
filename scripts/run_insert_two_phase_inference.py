#!/usr/bin/env python3
"""
Two-phase INSERT inference + eval (catalog → node schema + neighbors → patch JSON → programmatic merge).

Phase 1: user instruction head + node catalog (from ``NodeSchemaStore``) → best ``selected_types``.
Phase 2: compact schema + neighbor summaries + optional ``build_hints`` → ``parameters`` / clarify.
Merge: shallow default fill from schema, model patch, then **deep** merge of instruction overrides
(``Set parameters to: {...}`` JSON or parsed ``Set parameters such as: key = value`` prose) so nested
keys like ``options.*`` are not wiped; finally ``insert_pipeline.apply_insert_splice`` splices the node.

Use ``--workflow-only`` to skip oracle rows whose ``output`` is ask text (string) and to use a Phase 2
patch-only prompt. If the model still returns ``clarify``, the run coerces to ``patch`` so each row still
yields a merged workflow for comparison against oracle.

Outputs align with ``run_insert_inference_and_eval`` (predictions.jsonl, metrics.json) for comparability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

_PKG = Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from insert_pipeline import (  # noqa: E402
    NodeSchemaStore,
    apply_insert_splice,
    build_neighbor_context,
    build_phase1_messages,
    build_phase2_messages,
    deep_merge_parameters,
    default_phase1_system_prompt,
    default_phase2_system_prompt,
    default_phase2_system_prompt_workflow_oracle,
    extract_template_workflow,
    merge_parameters_with_defaults,
    parameter_defaults_from_schema,
    parse_insert_instruction,
    parse_phase1_json,
    parse_phase2_json,
)
from run_delete_inference_and_eval import (  # noqa: E402
    _extract_first_json_object,
    _sha256_text,
    _jsonl_iter,
    _jsonl_iter_tolerant,
    build_hints,
)
from run_insert_inference_and_eval import (  # noqa: E402
    _load_insert_clues,
    evaluate_insert,
)

JSONDict = Dict[str, Any]


def _chat(client: OpenAI, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if not resp.choices or not resp.choices[0].message:
        return ""
    c = resp.choices[0].message.content
    return c if isinstance(c, str) else ""


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
    ap.add_argument(
        "--out-dir",
        type=str,
        default="n8n_workflow_generator_package/outputs/insert_two_phase_eval",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--phase1-max-tokens", type=int, default=1024)
    ap.add_argument("--catalog-max-lines", type=int, default=0, help="0 = include all catalog lines")
    ap.add_argument(
        "--schema-roots",
        type=str,
        default="",
        help="Comma-separated extra schema dirs (optional). Defaults to repo core_nodes_schemas + langchain.",
    )
    ap.add_argument("--append-hints", action="store_true")
    ap.add_argument(
        "--workflow-only",
        action="store_true",
        help="Only rows whose oracle output is a workflow object (dict), not ask text. Phase 2 uses patch-only prompt; clarify responses are coerced to patch.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0)
    ap.add_argument("--sleep-ms", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    clues_path = Path(args.oracle_clues_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    pred_compact_path = out_dir / "predictions_compact.jsonl"
    metrics_path = out_dir / "metrics.json"

    roots = None
    if args.schema_roots.strip():
        roots = [Path(p.strip()) for p in args.schema_roots.split(",") if p.strip()]
    store = NodeSchemaStore(roots=roots)
    catalog = store.build_catalog_text(max_lines=int(args.catalog_max_lines or 0))

    api_key = os.getenv(str(args.api_key_env or "OPENAI_API_KEY"))
    if not api_key:
        if not args.base_url:
            raise SystemExit(f"Missing {args.api_key_env} env var")
        api_key = "local-api-key"
    client = OpenAI(api_key=api_key, base_url=(args.base_url or None))

    clues = _load_insert_clues(clues_path)
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
    mt = int(args.max_output_tokens)
    p1t = int(args.phase1_max_tokens)
    skipped_non_workflow_oracle = 0

    for rec in _jsonl_iter(in_path):
        if lim and counters["n"] >= lim:
            break
        if not isinstance(rec.get("input"), str) or "output" not in rec:
            continue
        input_text: str = rec["input"]
        oracle_out = rec["output"]
        if args.workflow_only and not isinstance(oracle_out, dict):
            skipped_non_workflow_oracle += 1
            continue
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
        pinfo = parse_insert_instruction(input_text)
        loc: JSONDict = clue.get("location") if isinstance(clue.get("location"), dict) else {}
        if not loc and isinstance(pinfo.get("location"), dict):
            loc = pinfo["location"]
        template = extract_template_workflow(input_text)
        hints = build_hints(input_text) if args.append_hints else ""

        phase1_raw = ""
        phase2_raw = ""
        api_error: Optional[str] = None
        declared = pinfo.get("declared_node_type")
        types_list: list[str] = [declared] if isinstance(declared, str) and declared else []

        try:
            if not types_list:
                p1m = build_phase1_messages(
                    instruction_head=pinfo["instruction_head"],
                    catalog_text=catalog,
                    system_prompt=default_phase1_system_prompt(),
                )
                phase1_raw = _chat(client, args.model, p1m, args.temperature, p1t)
                types_list, _ = parse_phase1_json(phase1_raw)
            chosen_type = types_list[0] if types_list else None
            if not chosen_type:
                raise ValueError("phase1_no_type")
            if not store.resolve_path(chosen_type):
                raise ValueError(f"unknown_node_type:{chosen_type}")

            sch = store.load_schema(chosen_type)
            defaults = parameter_defaults_from_schema(sch)
            compact = store.compact_schema_for_llm(chosen_type)
            nbor = build_neighbor_context(template or {}, pinfo.get("location") or loc or {})
            hint_block = hints or ""
            user_ov = pinfo.get("user_parameter_override")
            if not isinstance(user_ov, dict):
                user_ov = None
            phase2_sys = (
                default_phase2_system_prompt_workflow_oracle()
                if args.workflow_only
                else default_phase2_system_prompt()
            )
            p2m = build_phase2_messages(
                instruction_head=pinfo["instruction_head"],
                inserted_node_name=str(pinfo["inserted_node_name"] or clue.get("inserted_node_name") or ""),
                node_type=chosen_type,
                compact_schema=compact,
                neighbor_context=nbor,
                positional_hints=hint_block,
                user_parameter_json=user_ov,
                system_prompt=phase2_sys,
            )
            phase2_raw = _chat(client, args.model, p2m, args.temperature, mt)
            p2 = parse_phase2_json(phase2_raw)
            if not p2:
                raise ValueError("phase2_parse_failed")

            phase2_coerced_from_clarify = False
            if args.workflow_only and str(p2.get("mode") or "").lower() == "clarify":
                p2 = {
                    "mode": "patch",
                    "parameters": p2.get("parameters") if isinstance(p2.get("parameters"), dict) else {},
                    "typeVersion": p2.get("typeVersion"),
                    "clarify_message": None,
                }
                phase2_coerced_from_clarify = True

            mode = str(p2.get("mode") or "patch").lower()
            pred_raw: Optional[str] = None
            merged_debug: Any = None

            if mode == "clarify":
                msg = p2.get("clarify_message")
                pred_raw = str(msg) if msg else phase2_raw
            else:
                params_llm = p2.get("parameters")
                if not isinstance(params_llm, dict):
                    params_llm = {}
                merged = merge_parameters_with_defaults(params_llm, defaults)
                if user_ov:
                    merged = deep_merge_parameters(merged, user_ov)
                tv = p2.get("typeVersion")
                tv_f: Optional[float] = None
                if isinstance(tv, (int, float)):
                    tv_f = float(tv)
                elif sch and isinstance(sch.get("version"), (int, float)):
                    tv_f = float(sch["version"])

                name_ins = str(pinfo["inserted_node_name"] or clue.get("inserted_node_name") or "")
                if not template or not name_ins:
                    raise ValueError("missing_template_or_name")
                merged_wf = apply_insert_splice(
                    template,
                    new_node_name=name_ins,
                    node_type=chosen_type,
                    parameters=merged,
                    location=loc or pinfo.get("location") or {},
                    type_version=tv_f,
                )
                pred_raw = json.dumps(merged_wf, ensure_ascii=False)
                merged_debug = {"parameters_merged": merged, "typeVersion": tv_f}
                if phase2_coerced_from_clarify:
                    merged_debug["coerced_from_clarify"] = True

            ev = evaluate_insert(oracle_out=oracle_out, clue=clue, pred_raw=pred_raw)
            ej = ev.to_json()
            ej["insert_variant"] = insert_variant
            ej["two_phase"] = {
                "chosen_type": chosen_type,
                "phase1_raw_preview": (phase1_raw or "")[:600],
                "phase2_raw_preview": (phase2_raw or "")[:600],
                "merged_debug": merged_debug,
                "phase2_coerced_from_clarify": phase2_coerced_from_clarify,
            }

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

            user_text_for_log = pinfo["instruction_head"]
            if hint_block:
                user_text_for_log = user_text_for_log + "\n" + hint_block

            _write(
                pred_path,
                {
                    "input_sha256": input_sha,
                    "input": input_text,
                    "sent_to_model_phase2_head": user_text_for_log[:2000],
                    "oracle_out_kind": "ask" if isinstance(oracle_out, str) else "workflow",
                    "prediction": pred_wf,
                    "eval": ej,
                    "raw_text": pred_raw,
                    "phase1_raw": phase1_raw,
                    "phase2_raw": phase2_raw,
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
        except Exception as e:
            api_error = str(e)
            ev = evaluate_insert(oracle_out=oracle_out, clue=clue, pred_raw=None)
            ej = ev.to_json()
            ej["insert_variant"] = insert_variant
            ej["two_phase"] = {"error": api_error, "phase1_raw_preview": (phase1_raw or "")[:400]}
            pred_wf = None
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
                    "eval": ej,
                    "raw_text": None,
                    "phase1_raw": phase1_raw,
                    "phase2_raw": phase2_raw,
                    "api_error": api_error,
                    "prediction": pred_wf,
                },
            )
            _write(
                pred_compact_path,
                {
                    "input_sha256": input_sha,
                    "insert_variant": insert_variant,
                    "error_type": ev.error_type,
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
        "pipeline": "two_phase_insert",
        "workflow_only": bool(args.workflow_only),
        "rows_skipped_non_workflow_oracle": int(skipped_non_workflow_oracle),
        "input_jsonl": str(in_path),
        "oracle_clues_jsonl": str(clues_path),
        "append_hints": bool(args.append_hints),
        "catalog_lines_capped": int(args.catalog_max_lines or 0),
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
