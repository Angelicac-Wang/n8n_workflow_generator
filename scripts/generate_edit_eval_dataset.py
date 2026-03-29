#!/usr/bin/env python3
"""
Generate evaluation datasets for workflow editing tasks from templates.

Targets tasks similar to your finetune data:
- [TASK=delete]: delete a node from a given workflow JSON
- [TASK=modify]: modify a specific parameter (by JSON pointer) in a node
- [TASK=modify]: insert a Set node between an existing main edge

Outputs (JSONL):
- inputs.jsonl: model inputs only (messages w/ user prompt containing Template)
- oracles.jsonl: expected workflow JSON + edit_spec "clues" for matching
- combined.jsonl: merged for debugging
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

JSONDict = Dict[str, Any]


def _json_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _sha256_of_json(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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
    files = [p for p in sorted(templates_dir.glob("*.json")) if p.name not in skipped]
    return files


def _extract_template_id(template: JSONDict, filename: str) -> str:
    if isinstance(template.get("metadata"), dict) and template["metadata"].get("id") is not None:
        return str(template["metadata"]["id"])
    if template.get("id") is not None:
        return str(template["id"])
    # fallback from filename like template_10000_Template_10000.json
    for part in filename.split("_"):
        if part.isdigit():
            return part
    return ""


def _extract_workflow_json(template: JSONDict) -> Optional[JSONDict]:
    wf = template.get("workflow")
    if isinstance(wf, dict):
        inner = wf.get("workflow")
        if isinstance(inner, dict) and isinstance(inner.get("nodes"), list) and isinstance(inner.get("connections"), dict):
            return inner
    if isinstance(template.get("nodes"), list) and isinstance(template.get("connections"), dict):
        return template
    return None


def _is_sticky(node: JSONDict) -> bool:
    t = (node.get("type") or "")
    return "stickynote" in t.lower()


def _list_edges(workflow: JSONDict) -> List[Tuple[str, str, int, int, str]]:
    """
    (src_name, tgt_name, output_index, tgt_input_index, output_type)
    """
    edges: List[Tuple[str, str, int, int, str]] = []
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return edges
    for src_name, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        for output_type, output_lists in outputs.items():
            if not isinstance(output_lists, list):
                continue
            for out_idx, targets in enumerate(output_lists):
                if not isinstance(targets, list):
                    continue
                for t in targets:
                    if not isinstance(t, dict):
                        continue
                    tgt = t.get("node")
                    if not tgt:
                        continue
                    tgt_in = int(t.get("index", 0) or 0)
                    edges.append((str(src_name), str(tgt), int(out_idx), tgt_in, str(output_type)))
    return edges


def _ensure_connections_entry(workflow: JSONDict, node_name: str) -> JSONDict:
    conns = workflow.setdefault("connections", {})
    if node_name not in conns or not isinstance(conns.get(node_name), dict):
        conns[node_name] = {}
    return conns[node_name]


def _delete_node_with_bridge(workflow: JSONDict, node_name: str) -> Tuple[JSONDict, JSONDict]:
    """
    Delete node_name. If exactly 1 incoming main edge and >=1 outgoing main targets on output 0,
    bridge predecessor -> successors.
    """
    wf = copy.deepcopy(workflow)
    edges = _list_edges(wf)
    incoming = [(src, out_idx) for (src, tgt, out_idx, _tgt_in, out_type) in edges if tgt == node_name and out_type == "main"]
    outgoing = [tgt for (src, tgt, out_idx, _tgt_in, out_type) in edges if src == node_name and out_type == "main" and out_idx == 0]

    bridged = False
    bridge_from = None
    if len(incoming) == 1 and len(outgoing) >= 1:
        bridged = True
        bridge_from = {"from": incoming[0][0], "output_index": incoming[0][1]}
        pred, pred_out_idx = incoming[0]
        conns = wf.get("connections", {})
        pred_outputs = conns.get(pred, {})
        main_lists = pred_outputs.get("main")
        if isinstance(main_lists, list) and pred_out_idx < len(main_lists) and isinstance(main_lists[pred_out_idx], list):
            new_targets: List[JSONDict] = []
            for t in main_lists[pred_out_idx]:
                if isinstance(t, dict) and t.get("node") == node_name:
                    for succ in outgoing:
                        new_targets.append({"node": succ, "type": "main", "index": 0})
                else:
                    new_targets.append(t)
            main_lists[pred_out_idx] = new_targets

    # Remove node from nodes list
    nodes = wf.get("nodes", [])
    if isinstance(nodes, list):
        wf["nodes"] = [n for n in nodes if isinstance(n, dict) and n.get("name") != node_name]

    # Remove node's outgoing connections block
    conns = wf.get("connections", {})
    if isinstance(conns, dict):
        conns.pop(node_name, None)
        # Remove references
        for _src, outputs in list(conns.items()):
            if not isinstance(outputs, dict):
                continue
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
        "incoming_main_edges": [{"from": s, "output_index": oi} for (s, oi) in incoming],
        "outgoing_main_targets_output0": outgoing,
    }
    if bridge_from:
        edit_spec["bridge_from"] = bridge_from
    return wf, edit_spec


def _choose_deletable_node(workflow: JSONDict) -> Optional[str]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        return None
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = node.get("name")
        if not name:
            continue
        if _is_sticky(node):
            continue
        return str(name)
    return None


def _find_first_main_edge(workflow: JSONDict) -> Optional[Tuple[str, str, int]]:
    for src, tgt, out_idx, _tgt_in, out_type in _list_edges(workflow):
        if out_type == "main":
            return src, tgt, out_idx
    return None


def _insert_set_between(workflow: JSONDict, src: str, tgt: str, output_index: int) -> Tuple[JSONDict, JSONDict]:
    wf = copy.deepcopy(workflow)
    nodes = wf.get("nodes", [])
    if not isinstance(nodes, list):
        return wf, {"action": "insert_node_between_edge", "error": "nodes_missing"}

    new_node_id = str(uuid.uuid4())
    new_node_name = f"Eval Inserted Set {new_node_id[:8]}"
    assignment_id = str(uuid.uuid4())

    new_node: JSONDict = {
        "id": new_node_id,
        "name": new_node_name,
        "type": "n8n-nodes-base.set",
        "typeVersion": 3.4,
        "position": [0, 0],
        "parameters": {
            "options": {},
            "assignments": {
                "assignments": [
                    {"id": assignment_id, "name": "eval_marker", "type": "string", "value": "inserted"}
                ]
            },
        },
    }
    nodes.append(new_node)
    wf["nodes"] = nodes

    conns = wf.setdefault("connections", {})
    src_outputs = conns.get(src)
    if not isinstance(src_outputs, dict):
        src_outputs = _ensure_connections_entry(wf, src)
    main_lists = src_outputs.get("main")
    if not isinstance(main_lists, list):
        main_lists = []
        src_outputs["main"] = main_lists
    while len(main_lists) <= output_index:
        main_lists.append([])
    if not isinstance(main_lists[output_index], list):
        main_lists[output_index] = []

    replaced = False
    for i, t in enumerate(main_lists[output_index]):
        if isinstance(t, dict) and t.get("node") == tgt:
            nt = dict(t)
            nt["node"] = new_node_name
            main_lists[output_index][i] = nt
            replaced = True
            break
    if not replaced:
        main_lists[output_index].append({"node": new_node_name, "type": "main", "index": 0})

    new_outputs = _ensure_connections_entry(wf, new_node_name)
    new_main = new_outputs.setdefault("main", [[]])
    if not isinstance(new_main, list) or not new_main:
        new_main = [[]]
        new_outputs["main"] = new_main
    if not isinstance(new_main[0], list):
        new_main[0] = []
    new_main[0].append({"node": tgt, "type": "main", "index": 0})

    edit_spec: JSONDict = {
        "action": "insert_node_between_edge",
        "inserted_node": {"id": new_node_id, "name": new_node_name, "type": new_node["type"]},
        "required_marker": {"param_name": "eval_marker", "param_value": "inserted"},
        "edge_before": {"from": src, "to": tgt, "output": "main", "output_index": output_index},
        "edge_after": [
            {"from": src, "to": new_node_name, "output": "main", "output_index": output_index},
            {"from": new_node_name, "to": tgt, "output": "main", "output_index": 0},
        ],
    }
    return wf, edit_spec


@dataclass
class ParamEdit:
    node_name: str
    json_pointer: str
    old_value: Any
    new_value: Any


def _iter_scalar_pointers(obj: Any, base: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_esc = str(k).replace("~", "~0").replace("/", "~1")
            yield from _iter_scalar_pointers(v, base + "/" + k_esc)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_scalar_pointers(v, base + f"/{i}")
    else:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            yield base or "/", obj


def _set_by_pointer(obj: Any, ptr: str, value: Any) -> None:
    if ptr in ("", "/"):
        raise ValueError("Refuse to set root pointer '/'")
    parts = ptr.lstrip("/").split("/")
    cur = obj
    for p in parts[:-1]:
        p = p.replace("~1", "/").replace("~0", "~")
        cur = cur[p] if isinstance(cur, dict) else cur[int(p)]
    last = parts[-1].replace("~1", "/").replace("~0", "~")
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur[int(last)] = value


def _choose_param_edit(workflow: JSONDict) -> Optional[ParamEdit]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return None
    preferred_keys = ("url", "method", "path", "amount", "chatId", "text", "operation")
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if _is_sticky(node):
            continue
        params = node.get("parameters")
        if not isinstance(params, dict) or not params:
            continue
        candidates = list(_iter_scalar_pointers(params))
        if not candidates:
            continue
        preferred = [(p, v) for (p, v) in candidates if any(p.endswith("/" + k) for k in preferred_keys)]
        ptr, old_val = (preferred[0] if preferred else candidates[0])
        if isinstance(old_val, str) and len(old_val) > 200:
            continue
        if isinstance(old_val, bool):
            new_val = not old_val
        elif isinstance(old_val, int):
            new_val = old_val + 1
        elif isinstance(old_val, float):
            new_val = float(old_val) + 1.0
        elif isinstance(old_val, str):
            new_val = (old_val + " (updated)") if old_val else "updated"
        elif old_val is None:
            new_val = "updated"
        else:
            continue
        return ParamEdit(node_name=str(node.get("name")), json_pointer=ptr, old_value=old_val, new_value=new_val)
    return None


def _apply_param_edit(workflow: JSONDict, edit: ParamEdit) -> Tuple[JSONDict, JSONDict]:
    wf = copy.deepcopy(workflow)
    nodes = wf.get("nodes", [])
    for node in nodes:
        if isinstance(node, dict) and node.get("name") == edit.node_name and isinstance(node.get("parameters"), dict):
            _set_by_pointer(node["parameters"], edit.json_pointer, edit.new_value)
            break
    edit_spec: JSONDict = {"action": "modify_param", **asdict(edit)}
    return wf, edit_spec


def _build_user_prompt(task: str, instruction: str, workflow_json: JSONDict) -> str:
    return f"[TASK={task}]\\n{instruction}\\n\\nTemplate:\\n{_json_compact(workflow_json)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate edit evaluation dataset from templates")
    parser.add_argument("--templates-dir", default="n8n_templates/testing_data", help="Templates directory")
    parser.add_argument("--output-dir", default="outputs/edit_eval_testing_data", help="Output directory (relative to package root)")
    parser.add_argument("--limit", type=int, default=200, help="Max number of templates to sample")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    pkg_root = Path(__file__).parent.parent
    templates_dir = pkg_root / args.templates_dir
    out_dir = pkg_root / args.output_dir

    rng = random.Random(args.seed)
    files = _iter_template_files(templates_dir)
    rng.shuffle(files)
    files = files[: args.limit]

    inputs: List[JSONDict] = []
    oracles: List[JSONDict] = []
    combined: List[JSONDict] = []

    for path in files:
        template = _load_json(path)
        template_id = _extract_template_id(template, path.name)
        wf = _extract_workflow_json(template)
        if not template_id or not wf:
            continue

        base_sha = _sha256_of_json(wf)
        template_file = str(path.relative_to(pkg_root))

        # 1) delete
        del_node = _choose_deletable_node(wf)
        if del_node:
            expected_wf, edit_spec = _delete_node_with_bridge(wf, del_node)
            rec_id = f"{template_id}-delete-{uuid.uuid4().hex[:8]}"
            instruction = f'Remove the node "{del_node}". Output the full updated workflow JSON only.'
            user_content = _build_user_prompt("delete", instruction, wf)
            msg = {"messages": [{"role": "user", "content": user_content}]}
            oracle = {
                "id": rec_id,
                "template_id": template_id,
                "task": "delete",
                "template_file": template_file,
                "source_workflow_sha256": base_sha,
                "expected_workflow_sha256": _sha256_of_json(expected_wf),
                "edit_spec": edit_spec,
                "expected_workflow": expected_wf,
            }
            inp = {"id": rec_id, "template_id": template_id, "task": "delete", **msg}
            inputs.append(inp)
            oracles.append(oracle)
            combined.append({"input": inp, "oracle": oracle})

        # 2) modify param
        pedit = _choose_param_edit(wf)
        if pedit:
            expected_wf, edit_spec = _apply_param_edit(wf, pedit)
            rec_id = f"{template_id}-modifyparam-{uuid.uuid4().hex[:8]}"
            instruction = (
                f'Update node "{pedit.node_name}" at parameters pointer "{pedit.json_pointer}" '
                f'from {json.dumps(pedit.old_value, ensure_ascii=False)} to {json.dumps(pedit.new_value, ensure_ascii=False)}. '
                "Output the full updated workflow JSON only."
            )
            user_content = _build_user_prompt("modify", instruction, wf)
            msg = {"messages": [{"role": "user", "content": user_content}]}
            oracle = {
                "id": rec_id,
                "template_id": template_id,
                "task": "modify_param",
                "template_file": template_file,
                "source_workflow_sha256": base_sha,
                "expected_workflow_sha256": _sha256_of_json(expected_wf),
                "edit_spec": edit_spec,
                "expected_workflow": expected_wf,
            }
            inp = {"id": rec_id, "template_id": template_id, "task": "modify_param", **msg}
            inputs.append(inp)
            oracles.append(oracle)
            combined.append({"input": inp, "oracle": oracle})

        # 3) insert node
        edge = _find_first_main_edge(wf)
        if edge:
            src, tgt, out_idx = edge
            expected_wf, edit_spec = _insert_set_between(wf, src, tgt, out_idx)
            rec_id = f"{template_id}-insert-{uuid.uuid4().hex[:8]}"
            instruction = (
                f'Insert a Set node between "{src}" -> "{tgt}" on main output index {out_idx}. '
                'The inserted node must set string parameter eval_marker="inserted". '
                "Output the full updated workflow JSON only."
            )
            user_content = _build_user_prompt("modify", instruction, wf)
            msg = {"messages": [{"role": "user", "content": user_content}]}
            oracle = {
                "id": rec_id,
                "template_id": template_id,
                "task": "insert_node_between_edge",
                "template_file": template_file,
                "source_workflow_sha256": base_sha,
                "expected_workflow_sha256": _sha256_of_json(expected_wf),
                "edit_spec": edit_spec,
                "expected_workflow": expected_wf,
            }
            inp = {"id": rec_id, "template_id": template_id, "task": "insert_node_between_edge", **msg}
            inputs.append(inp)
            oracles.append(oracle)
            combined.append({"input": inp, "oracle": oracle})

    _save_jsonl(inputs, out_dir / "edit_eval_inputs.jsonl")
    _save_jsonl(oracles, out_dir / "edit_eval_oracles.jsonl")
    _save_jsonl(combined, out_dir / "edit_eval_combined.jsonl")

    print(f"Saved inputs:  {out_dir / 'edit_eval_inputs.jsonl'} ({len(inputs)})")
    print(f"Saved oracles: {out_dir / 'edit_eval_oracles.jsonl'} ({len(oracles)})")
    print(f"Saved combined:{out_dir / 'edit_eval_combined.jsonl'} ({len(combined)})")


if __name__ == "__main__":
    main()

