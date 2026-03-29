#!/usr/bin/env python3
"""
Generate *testing* datasets for edit tasks in the same style as previous training data:

- delete: {"input": "<instruction>\\n\\nTemplate:\\n<workflow_json>", "output": <edited_workflow_obj>}
- insert (full): same but instruction includes type + "Set parameters to: {json}."
- insert (ask): output is a question string asking for parameter values
- insert (partial): instruction includes partial parameters; output is either full workflow (if fillable by schema defaults)
  or an ask string for missing parameters

Also writes an oracle clues file (separate JSONL) keyed by sha256(input) so you can grade automatically.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

JSONDict = Dict[str, Any]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_pretty_inline(obj: Any) -> str:
    # training data uses compact-ish JSON with spaces; keep it readable but single-line
    return json.dumps(obj, ensure_ascii=False)


def _load_json(path: Path) -> JSONDict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_jsonl(records: Iterable[JSONDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _iter_template_files(templates_dir: Path) -> List[Path]:
    skipped = {"template_index.json", "templates_index.json", "templateIndex.json"}
    return [p for p in sorted(templates_dir.glob("*.json")) if p.name not in skipped]


def _extract_workflow_json(template: JSONDict) -> Optional[JSONDict]:
    wf = template.get("workflow")
    if isinstance(wf, dict):
        inner = wf.get("workflow")
        if isinstance(inner, dict) and isinstance(inner.get("nodes"), list) and isinstance(inner.get("connections"), dict):
            return inner
    if isinstance(template.get("nodes"), list) and isinstance(template.get("connections"), dict):
        return template
    return None


def _node_by_name(workflow: JSONDict, name: str) -> Optional[JSONDict]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if isinstance(n, dict) and n.get("name") == name:
            return n
    return None


def _is_sticky(node: JSONDict) -> bool:
    t = str(node.get("type") or "")
    return "stickynote" in t.lower()


def _is_trigger(node: JSONDict) -> bool:
    t = str(node.get("type") or "").lower()
    name = str(node.get("name") or "").lower()
    if "trigger" in t or t.endswith("trigger"):
        return True
    if t in {
        "n8n-nodes-base.manualtrigger",
        "n8n-nodes-base.webhook",
        "n8n-nodes-base.cron",
        "n8n-nodes-base.scheduletrigger",
        "n8n-nodes-base.formtrigger",
        "n8n-nodes-base.emailreadimap",
    }:
        return True
    if "webhook" in t or "cron" in t or "schedule" in t:
        return True
    # fallback heuristic: commonly named triggers
    if "on clicking" in name or "trigger" in name:
        return True
    return False


def _list_main_edges(workflow: JSONDict) -> List[Tuple[str, str, int, int]]:
    edges: List[Tuple[str, str, int, int]] = []
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return edges
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        main = outputs.get("main")
        if not isinstance(main, list):
            continue
        for out_idx, targets in enumerate(main):
            if not isinstance(targets, list):
                continue
            for t in targets:
                if not isinstance(t, dict):
                    continue
                tgt = t.get("node")
                if not tgt:
                    continue
                tgt_in = int(t.get("index", 0) or 0)
                edges.append((str(src), str(tgt), int(out_idx), tgt_in))
    return edges


def _incoming_main(workflow: JSONDict) -> Dict[str, List[Tuple[str, int]]]:
    inc: Dict[str, List[Tuple[str, int]]] = {}
    for src, tgt, out_idx, _tgt_in in _list_main_edges(workflow):
        inc.setdefault(tgt, []).append((src, out_idx))
    return inc


def _outgoing_main(workflow: JSONDict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for src, tgt, out_idx, _tgt_in in _list_main_edges(workflow):
        if out_idx != 0:
            continue
        out.setdefault(src, []).append(tgt)
    return out


def _branch_nodes(workflow: JSONDict) -> List[str]:
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return []
    branch: List[str] = []
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        main = outputs.get("main")
        if not isinstance(main, list):
            continue
        total_targets = 0
        for targets in main:
            if isinstance(targets, list):
                total_targets += sum(1 for t in targets if isinstance(t, dict) and t.get("node"))
        if total_targets >= 2:
            branch.append(str(src))
    return branch


def _remove_node_and_cleanup(workflow: JSONDict, node_name: str, bridge: bool = True) -> Tuple[JSONDict, JSONDict]:
    wf = copy.deepcopy(workflow)
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return wf, {"action": "delete_node", "error": "nodes_missing", "target_node": node_name}

    deleted_node = None
    kept_nodes: List[JSONDict] = []
    for n in nodes:
        if isinstance(n, dict) and n.get("name") == node_name:
            deleted_node = n
        else:
            if isinstance(n, dict):
                kept_nodes.append(n)
    wf["nodes"] = kept_nodes

    edges = _list_main_edges(wf)
    incoming = [(src, out_idx) for (src, tgt, out_idx, _tgt_in) in edges if tgt == node_name]
    outgoing = [tgt for (src, tgt, out_idx, _tgt_in) in edges if src == node_name and out_idx == 0]

    bridged = False
    bridge_from: Optional[JSONDict] = None
    if bridge and len(incoming) == 1 and len(outgoing) >= 1:
        pred, pred_out_idx = incoming[0]
        conns = wf.get("connections", {})
        pred_outputs = conns.get(pred, {})
        main = pred_outputs.get("main")
        if isinstance(main, list) and pred_out_idx < len(main) and isinstance(main[pred_out_idx], list):
            new_targets: List[JSONDict] = []
            for t in main[pred_out_idx]:
                if isinstance(t, dict) and t.get("node") == node_name:
                    for succ in outgoing:
                        new_targets.append({"node": succ, "type": "main", "index": 0})
                else:
                    new_targets.append(t)
            main[pred_out_idx] = new_targets
            bridged = True
            bridge_from = {"from": pred, "output_index": pred_out_idx}

    conns = wf.get("connections")
    if isinstance(conns, dict):
        conns.pop(node_name, None)
        for _src, outputs in list(conns.items()):
            if not isinstance(outputs, dict):
                continue
            # Remove references across ALL output types (main, ai_tool, ai_embedding, ai_memory, etc.)
            for out_type, out_lists in list(outputs.items()):
                if not isinstance(out_lists, list):
                    continue
                for i, targets in enumerate(out_lists):
                    if not isinstance(targets, list):
                        continue
                    out_lists[i] = [t for t in targets if not (isinstance(t, dict) and t.get("node") == node_name)]

    edit_spec: JSONDict = {
        "action": "delete_node",
        "target_node": node_name,
        "bridged": bridged,
        "incoming_main": [{"from": s, "output_index": oi} for (s, oi) in incoming],
        "outgoing_main_output0": outgoing,
    }
    if deleted_node is not None:
        edit_spec["deleted_node_type"] = deleted_node.get("type")
        edit_spec["deleted_node_params_hash"] = _sha256_json(deleted_node.get("parameters", {}))
    if bridge_from is not None:
        edit_spec["bridge_from"] = bridge_from
    return wf, edit_spec


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _instruction_delete_no_branch(rng: random.Random, workflow: JSONDict, node_name: str) -> Tuple[str, JSONDict]:
    node = _node_by_name(workflow, node_name) or {}
    node_type = str(node.get("type") or "")

    inc = _incoming_main(workflow)
    out = _outgoing_main(workflow)
    preds = [p for (p, _oi) in inc.get(node_name, [])]
    succs = out.get(node_name, [])
    prev = preds[0] if preds else None
    nxt = succs[0] if succs else None

    candidates: List[Tuple[str, str]] = []
    candidates.append(("by_name", f'{rng.choice(["Delete", "Remove"])} the node "{node_name}".'))
    if node_type:
        candidates.append(("by_type", f'Delete the "{node_type}" node.'))
    if prev and nxt:
        candidates.append(("between", f'Delete the node "{node_name}" between "{prev}" and "{nxt}".'))
        candidates.append(("before", f'Delete the node before "{nxt}".'))
        candidates.append(("after", f'Delete the node after "{prev}".'))
    elif prev:
        candidates.append(("after", f'Delete the node after "{prev}".'))
    elif nxt:
        candidates.append(("before", f'Delete the node before "{nxt}".'))

    # left/right ordering by (x, y, name) for deterministic, non-ambiguous ranking
    nodes = [n for n in (workflow.get("nodes") or []) if isinstance(n, dict) and not _is_sticky(n)]
    xs: List[Tuple[int, int, str]] = []
    for n in nodes:
        name = n.get("name")
        pos = n.get("position")
        if not name or not isinstance(pos, list) or len(pos) < 2:
            continue
        try:
            xs.append((int(pos[0]), int(pos[1]), str(name)))
        except Exception:
            continue
    if len(xs) >= 2 and node_name in {nm for _x, _y, nm in xs}:
        xs_sorted = sorted(xs, key=lambda t: (t[0], t[1], t[2]))
        left_rank = [nm for _x, _y, nm in xs_sorted].index(node_name) + 1
        right_rank = len(xs_sorted) - left_rank + 1
        candidates.append(("from_left", f"Delete the {_ordinal(left_rank)} node from the left."))
        candidates.append(("from_right", f"Delete the {_ordinal(right_rank)} node from the right."))

    variant, text = rng.choice(candidates)
    return text, {"variant": variant, "node": node_name}


def _split_branches_by_position(workflow: JSONDict, branch_node: str) -> Optional[Tuple[str, str]]:
    out = _outgoing_main(workflow).get(branch_node, [])
    if len(out) < 2:
        return None
    # pick two distinct immediate targets
    a, b = out[0], out[1]
    na, nb = _node_by_name(workflow, a), _node_by_name(workflow, b)
    ya = (na.get("position") or [0, 0])[1] if isinstance(na, dict) else 0
    yb = (nb.get("position") or [0, 0])[1] if isinstance(nb, dict) else 0
    # "top" has smaller y
    if ya <= yb:
        return a, b
    return b, a


def _walk_linear_path(workflow: JSONDict, start: str, max_hops: int = 25) -> List[str]:
    inc = _incoming_main(workflow)
    out = _outgoing_main(workflow)
    path: List[str] = []
    cur = start
    seen = set()
    for _ in range(max_hops):
        if cur in seen:
            break
        seen.add(cur)
        path.append(cur)
        nxts = out.get(cur, [])
        # stop at end, split, or merge
        if len(nxts) != 1:
            break
        nxt = nxts[0]
        if len(inc.get(nxt, [])) > 1:
            break
        cur = nxt
    return path


def _instruction_delete_single_branch(rng: random.Random, workflow: JSONDict, branch_node: str) -> Tuple[Optional[str], Optional[str], JSONDict]:
    split = _split_branches_by_position(workflow, branch_node)
    if not split:
        return None, None, {"variant": "branch_failed", "branch_node": branch_node}
    top_start, bottom_start = split

    top_path = _walk_linear_path(workflow, top_start)
    bottom_path = _walk_linear_path(workflow, bottom_start)

    def _filter_deletable(path: List[str]) -> List[str]:
        out: List[str] = []
        for nm in path:
            node = _node_by_name(workflow, nm)
            if not isinstance(node, dict):
                continue
            if _is_sticky(node) or _is_trigger(node):
                continue
            out.append(nm)
        return out

    top_del = _filter_deletable(top_path)
    bottom_del = _filter_deletable(bottom_path)
    options: List[Tuple[str, str, List[str]]] = []
    if top_del:
        options.append(("top", "top", top_del))
    if bottom_del:
        options.append(("bottom", "bottom", bottom_del))
    if not options:
        return None, None, {"variant": "branch_no_deletable", "branch_node": branch_node}

    branch_label, _branch_key, del_list = rng.choice(options)
    idx = rng.randint(1, len(del_list))
    target = del_list[idx - 1]
    instr = f"Delete the {_ordinal(idx)} node from the {branch_label} branch."
    return instr, target, {"variant": "branch_ordinal", "branch_node": branch_node, "branch": branch_label, "index": idx}


def _choose_deletable_nodes(workflow: JSONDict) -> List[str]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return []
    names: List[str] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        name = n.get("name")
        if not name:
            continue
        if _is_sticky(n) or _is_trigger(n):
            continue
        names.append(str(name))
    return names


def _build_delete_record(rng: random.Random, workflow: JSONDict) -> Optional[Tuple[JSONDict, JSONDict]]:
    branches = _branch_nodes(workflow)
    deletables = _choose_deletable_nodes(workflow)
    if len(deletables) < 1:
        return None

    # prefer nodes with clear (prev,next) in main for bridging/insertion
    inc = _incoming_main(workflow)
    out = _outgoing_main(workflow)
    linear_candidates = [nm for nm in deletables if len(inc.get(nm, [])) == 1 and len(out.get(nm, [])) == 1]
    if linear_candidates:
        base_target = rng.choice(linear_candidates)
    else:
        base_target = rng.choice(deletables)

    variant_meta: JSONDict = {}
    instruction = None
    target = base_target
    branching_case: str

    if len(branches) == 0:
        branching_case = "no_branch"
        instruction, variant_meta = _instruction_delete_no_branch(rng, workflow, target)
    elif len(branches) == 1:
        branching_case = "single_branch"
        instr, tgt, meta = _instruction_delete_single_branch(rng, workflow, branches[0])
        if instr and tgt:
            instruction, target, variant_meta = instr, tgt, meta
        else:
            instruction, variant_meta = _instruction_delete_no_branch(rng, workflow, target)
            variant_meta["fallback_from"] = meta.get("variant")
    else:
        branching_case = "multi_branch"
        # constraint: only by name / by type / between prev-next
        node = _node_by_name(workflow, target) or {}
        node_type = str(node.get("type") or "")
        preds = [p for (p, _oi) in _incoming_main(workflow).get(target, [])]
        succs = _outgoing_main(workflow).get(target, [])
        prev = preds[0] if preds else None
        nxt = succs[0] if succs else None
        choices: List[Tuple[str, str]] = [("by_name", f'Delete the node "{target}".')]
        if node_type:
            choices.append(("by_type", f'Remove the "{node_type}" node.'))
        if prev and nxt:
            choices.append(("between", f'Delete the node "{target}" between "{prev}" and "{nxt}".'))
        v, instruction = rng.choice(choices)
        variant_meta = {"variant": v, "node": target}

    deleted_wf, edit_spec = _remove_node_and_cleanup(workflow, target, bridge=True)
    input_text = f"{instruction}\n\nTemplate:\n{_json_pretty_inline(workflow)}"
    record = {"input": input_text, "output": deleted_wf}

    clues = {
        "task": "delete",
        "branching_case": branching_case,
        "instruction_meta": variant_meta,
        "edit_spec": edit_spec,
        "template_workflow_hash": _sha256_json(workflow),
        "deleted_workflow_hash": _sha256_json(deleted_wf),
    }
    return record, clues


def _schema_candidates_for_type(node_type: str) -> List[Path]:
    # Try a few naming strategies:
    # - fetched_nodes/langchain/node_schemas/@n8n_n8n-nodes-langchain.<name>.json
    # - core_nodes_schemas/<name>.json or <name>Tool.json
    leaf = node_type.split(".")[-1]
    candidates: List[Path] = []
    repo_root = Path(__file__).resolve().parents[2]  # .../n8n_workflow_generator_package/scripts/ -> repo root
    fetched = repo_root / "fetched_nodes"
    core = repo_root / "core_nodes_schemas"
    candidates.append(fetched / "langchain" / "node_schemas" / f"@n8n_n8n-nodes-langchain.{leaf}.json")
    candidates.append(core / f"{leaf}.json")
    candidates.append(core / f"{leaf}Tool.json")
    # some schemas appear lowerCamel + Tool; also allow leaf already includes Tool
    if not leaf.lower().endswith("tool"):
        candidates.append(core / f"{leaf}tool.json")
    return candidates


def _extract_defaults_from_schema(schema: Any) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}

    def walk(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            if "default" in obj and prefix:
                defaults[prefix] = obj["default"]
            # if this is a properties dict, recurse into children
            props = obj.get("properties")
            if isinstance(props, dict):
                for k, v in props.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    walk(v, key)
            # sometimes options live under "options" for a param, but we only care defaults here
            for k, v in obj.items():
                if k in {"properties"}:
                    continue
                if isinstance(v, (dict, list)):
                    walk(v, prefix)
        elif isinstance(obj, list):
            for it in obj:
                walk(it, prefix)

    walk(schema)
    return defaults


def _load_schema_defaults(node_type: str) -> Dict[str, Any]:
    for p in _schema_candidates_for_type(node_type):
        try:
            if p.exists():
                return _extract_defaults_from_schema(_load_json(p))
        except Exception:
            continue
    return {}


def _pick_param_keys_for_questions(params: Any, max_keys: int = 6) -> List[str]:
    if not isinstance(params, dict):
        return []
    keys: List[str] = []
    for k, v in params.items():
        if k in {"options"}:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            keys.append(str(k))
        # allow a small nested object (one level) but ask top-level key
        elif isinstance(v, dict) and len(v) <= 4:
            keys.append(str(k))
        if len(keys) >= max_keys:
            break
    return keys


def _format_param_ask(node_type: str, params: Any, keys: List[str], defaults_map: Dict[str, Any]) -> str:
    parts: List[str] = []
    if not isinstance(params, dict):
        params = {}
    for k in keys:
        dv = params.get(k)
        schema_default = defaults_map.get(k)
        default_val = dv if dv is not None else schema_default
        if isinstance(default_val, str) and len(default_val) > 80:
            default_val = default_val[:77] + "..."
        if default_val is None:
            parts.append(f"{k}(default: none)")
        else:
            parts.append(f"{k}(default: {default_val})")
    # keep the prompt compact; include type as hint
    joined = ", ".join(parts) if parts else "the required parameters"
    return f"Please provide values for these parameters: {joined}."


def _insert_node_between(workflow: JSONDict, src: str, tgt: str, node_obj: JSONDict) -> JSONDict:
    wf = copy.deepcopy(workflow)
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return wf
    # ensure unique name (rare collisions)
    name = str(node_obj.get("name") or "Inserted Node")
    existing = {n.get("name") for n in nodes if isinstance(n, dict)}
    if name in existing:
        name = f"{name} (reinserted)"
    new_node = copy.deepcopy(node_obj)
    new_node["name"] = name
    nodes.append(new_node)

    conns = wf.setdefault("connections", {})
    if not isinstance(conns, dict):
        wf["connections"] = {}
        conns = wf["connections"]

    # rewrite src -> tgt to src -> new
    src_outputs = conns.get(src)
    if isinstance(src_outputs, dict) and isinstance(src_outputs.get("main"), list):
        main = src_outputs["main"]
        for out_idx, targets in enumerate(main):
            if not isinstance(targets, list):
                continue
            new_targets: List[JSONDict] = []
            for t in targets:
                if isinstance(t, dict) and t.get("node") == tgt:
                    new_targets.append({"node": name, "type": "main", "index": int(t.get("index", 0) or 0)})
                else:
                    new_targets.append(t)
            main[out_idx] = new_targets

    # add new -> tgt (preserve index 0)
    conns[name] = {"main": [[{"node": tgt, "type": "main", "index": 0}]]}
    return wf


def _derive_insert_location_from_delete(original: JSONDict, deleted: JSONDict, deleted_node_name: str) -> Optional[Tuple[str, str]]:
    # best-effort: find an edge in original that went through deleted node with single prev/next
    inc = _incoming_main(original)
    out = _outgoing_main(original)
    preds = [p for (p, _oi) in inc.get(deleted_node_name, [])]
    succs = out.get(deleted_node_name, [])
    if preds and succs:
        return preds[0], succs[0]
    # fallback: try find any edge in deleted workflow to insert before its first edge
    edges = _list_main_edges(deleted)
    if edges:
        return edges[0][0], edges[0][1]
    return None


def _build_insert_records_from_delete(
    rng: random.Random, original: JSONDict, deleted: JSONDict, deleted_node: JSONDict, delete_clues: JSONDict
) -> Tuple[Optional[JSONDict], Optional[JSONDict], Optional[JSONDict], List[JSONDict]]:
    name = str(deleted_node.get("name") or "")
    node_type = str(deleted_node.get("type") or "")
    params = deleted_node.get("parameters", {})
    loc = _derive_insert_location_from_delete(original, deleted, name)
    if not loc:
        return None, None, None, []
    src, tgt = loc

    def _pick_loc_phrase(node_name: str) -> Tuple[str, JSONDict]:
        variants = [
            (f'between "{src}" and "{tgt}"', {"between": [src, tgt]}),
            (f'after "{src}"', {"after": src, "between": [src, tgt]}),
            (f'before "{tgt}"', {"before": tgt, "between": [src, tgt]}),
        ]
        phrase, meta = rng.choice(variants)
        return phrase, meta

    # ask (no params)
    loc_phrase_ask, loc_meta_ask = _pick_loc_phrase(name)
    input_ask = f'Insert the node "{name}" {loc_phrase_ask}.\n\nTemplate:\n{_json_pretty_inline(deleted)}'
    defaults_map = _load_schema_defaults(node_type) if node_type else {}
    q_keys = _pick_param_keys_for_questions(params, max_keys=6)
    output_ask = _format_param_ask(node_type, params, q_keys, defaults_map)
    rec_ask = {"input": input_ask, "output": output_ask}
    clues_ask = {
        "task": "insert_ask",
        "from_delete_input_sha256": _sha256_text(delete_clues["input_text"]),
        "deleted_node": {"name": name, "type": node_type},
        "location": loc_meta_ask,
        "output_kind": "ask",
        "missing_policy": "ask_all",
    }
    clues_ask["input_sha256"] = _sha256_text(input_ask)

    # full (params fully provided -> restore original workflow)
    loc_phrase_full, loc_meta_full = _pick_loc_phrase(name)
    input_full = (
        f'Insert the node "{name}" of type "{node_type}" {loc_phrase_full}. '
        f"Set parameters to: {_json_pretty_inline(params)}.\n\nTemplate:\n{_json_pretty_inline(deleted)}"
    )
    rec_full = {"input": input_full, "output": original}
    clues_full = {
        "task": "insert_full",
        "from_delete_input_sha256": _sha256_text(delete_clues["input_text"]),
        "deleted_node": {"name": name, "type": node_type},
        "location": loc_meta_full,
        "output_kind": "workflow",
        "expected_workflow_hash": _sha256_json(original),
    }
    clues_full["input_sha256"] = _sha256_text(input_full)

    # partial
    partial_params: JSONDict = {}
    if isinstance(params, dict) and params:
        keys = list(params.keys())
        rng.shuffle(keys)
        keep_n = max(1, int(len(keys) * 0.5))
        for k in keys[:keep_n]:
            partial_params[k] = params[k]

    input_partial = (
        f'Insert the node "{name}" of type "{node_type}" {loc_phrase_full}. '
        f"Set parameters to: {_json_pretty_inline(partial_params)}.\n\nTemplate:\n{_json_pretty_inline(deleted)}"
    )

    # decide output: if schema defaults cover missing top-level keys that existed in original params -> full else ask missing
    missing = []
    if isinstance(params, dict):
        for k in params.keys():
            if k not in partial_params:
                missing.append(str(k))
    defaults_map = _load_schema_defaults(node_type) if node_type else {}
    fillable = all((k in defaults_map) for k in missing) if missing else True
    if fillable:
        rec_partial = {"input": input_partial, "output": original}
        missing_policy = "filled_by_defaults"
        output_kind = "workflow"
    else:
        ask = _format_param_ask(node_type, params, missing[:6], defaults_map)
        rec_partial = {"input": input_partial, "output": ask}
        missing_policy = "ask_missing"
        output_kind = "ask"
    clues_partial = {
        "task": "insert_partial",
        "from_delete_input_sha256": _sha256_text(delete_clues["input_text"]),
        "deleted_node": {"name": name, "type": node_type},
        "location": loc_meta_full,
        "output_kind": output_kind,
        "missing_top_level_keys": missing,
        "missing_policy": missing_policy,
        "expected_workflow_hash_if_full": _sha256_json(original),
    }
    clues_partial["input_sha256"] = _sha256_text(input_partial)

    return (rec_ask, rec_full, rec_partial, [clues_ask, clues_full, clues_partial])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates_dir = Path(args.templates_dir)
    output_dir = Path(args.output_dir)

    delete_records: List[JSONDict] = []
    insert_full_records: List[JSONDict] = []
    insert_ask_records: List[JSONDict] = []
    insert_partial_records: List[JSONDict] = []
    oracle_clues: List[JSONDict] = []

    files = _iter_template_files(templates_dir)
    rng.shuffle(files)

    for path in files[: args.limit]:
        try:
            template = _load_json(path)
        except Exception:
            continue
        wf = _extract_workflow_json(template)
        if not wf:
            continue

        built = _build_delete_record(rng, wf)
        if not built:
            continue
        del_rec, del_clues = built
        delete_records.append(del_rec)

        # add richer clues with stable input hash
        input_text = del_rec["input"]
        del_clues["input_sha256"] = _sha256_text(input_text)
        del_clues["input_text"] = input_text  # keep for internal reference; will be removed before saving
        del_clues["template_file"] = path.name
        del_clues["deleted_node_name"] = del_clues["edit_spec"]["target_node"]
        oracle_clues.append({k: v for k, v in del_clues.items() if k != "input_text"})

        deleted_wf = del_rec["output"]
        deleted_node = _node_by_name(wf, del_clues["edit_spec"]["target_node"])
        if not isinstance(deleted_node, dict):
            continue

        rec_ask, rec_full, rec_partial, clues_list = _build_insert_records_from_delete(
            rng=rng, original=wf, deleted=deleted_wf, deleted_node=deleted_node, delete_clues=del_clues
        )
        if rec_ask:
            insert_ask_records.append(rec_ask)
        if rec_full:
            insert_full_records.append(rec_full)
        if rec_partial:
            insert_partial_records.append(rec_partial)

        for c in clues_list:
            c["template_file"] = path.name
            oracle_clues.append(c)

    _save_jsonl(delete_records, output_dir / "delete_testing_data.jsonl")
    _save_jsonl(insert_full_records, output_dir / "insert_testing_data.jsonl")
    _save_jsonl(insert_ask_records, output_dir / "insert_testing_data_ask.jsonl")
    _save_jsonl(insert_partial_records, output_dir / "insert_testing_data_partial.jsonl")
    _save_jsonl(oracle_clues, output_dir / "oracle_clues.jsonl")

    print("Wrote:")
    print(f"- {output_dir / 'delete_testing_data.jsonl'} ({len(delete_records)})")
    print(f"- {output_dir / 'insert_testing_data.jsonl'} ({len(insert_full_records)})")
    print(f"- {output_dir / 'insert_testing_data_ask.jsonl'} ({len(insert_ask_records)})")
    print(f"- {output_dir / 'insert_testing_data_partial.jsonl'} ({len(insert_partial_records)})")
    print(f"- {output_dir / 'oracle_clues.jsonl'} ({len(oracle_clues)})")


if __name__ == "__main__":
    main()

