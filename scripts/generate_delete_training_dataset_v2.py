#!/usr/bin/env python3
"""
Generate a *delete* finetune training dataset from n8n templates.

Output format matches existing finetune JSONL:
  {"messages":[{"role":"user","content":"[TASK=delete]\\n...\\n\\nTemplate:\\n{...}"},{"role":"assistant","content":"{...edited workflow...}"}]}

Key guarantees:
- Choose a deletable node first (not trigger, not sticky).
- Remove that node from workflow.nodes.
- Remove ALL connections referencing that node (across all connection types, not just "main").
- Remove the deleted node's own connections entry.
- For positional instructions, compute rank deterministically by scanning ALL non-sticky nodes'
  x/y coordinates and ordering by (x, y, name) for left/right and (y, x, name) for top/bottom.

Also writes oracle clues JSONL for validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

JSONDict = Dict[str, Any]


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_json(obj: Any) -> str:
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
    if "on clicking" in name or "trigger" in name:
        return True
    return False


def _node_by_name(workflow: JSONDict, name: str) -> Optional[JSONDict]:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if isinstance(n, dict) and n.get("name") == name:
            return n
    return None


def _workflow_is_connection_consistent(workflow: JSONDict) -> bool:
    """
    True if every connection source/target refers to an existing node name.
    """
    names = set(_workflow_node_names(workflow))
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return True
    for src, outputs in conns.items():
        if str(src) not in names:
            return False
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
                        return False
    return True


def _workflow_node_names(wf: JSONDict) -> List[str]:
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return []
    out: List[str] = []
    for n in nodes:
        if isinstance(n, dict) and n.get("name"):
            out.append(str(n["name"]))
    return out


def _all_connection_refs(workflow: JSONDict) -> List[Tuple[str, str, str]]:
    """
    Returns list of (src, out_type, tgt) across ALL connection types.
    """
    conns = workflow.get("connections")
    if not isinstance(conns, dict):
        return []
    out: List[Tuple[str, str, str]] = []
    for src, outputs in conns.items():
        if not isinstance(outputs, dict):
            continue
        for out_type, out_lists in outputs.items():
            if not isinstance(out_lists, list):
                continue
            for targets in out_lists:
                if not isinstance(targets, list):
                    continue
                for t in targets:
                    if isinstance(t, dict) and t.get("node"):
                        out.append((str(src), str(out_type), str(t["node"])))
    return out


def delete_node_strict(workflow: JSONDict, target_name: str) -> Tuple[JSONDict, JSONDict]:
    """
    Delete target node and remove ALL references to it in workflow.connections.
    No bridging (per user requirement: remove edges).
    """
    wf = copy.deepcopy(workflow)

    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return wf, {"error": "nodes_missing", "target_node": target_name}

    deleted_node = _node_by_name(wf, target_name)
    wf["nodes"] = [n for n in nodes if not (isinstance(n, dict) and n.get("name") == target_name)]

    conns = wf.get("connections")
    if isinstance(conns, dict):
        conns.pop(target_name, None)
        for _src, outputs in list(conns.items()):
            if not isinstance(outputs, dict):
                continue
            for out_type, out_lists in list(outputs.items()):
                if not isinstance(out_lists, list):
                    continue
                for i, targets in enumerate(out_lists):
                    if not isinstance(targets, list):
                        continue
                    out_lists[i] = [t for t in targets if not (isinstance(t, dict) and t.get("node") == target_name)]

    edit_spec: JSONDict = {"action": "delete_node", "target_node": target_name}
    if isinstance(deleted_node, dict):
        edit_spec["deleted_node_type"] = deleted_node.get("type")
        edit_spec["deleted_node_params_hash"] = _sha256_json(deleted_node.get("parameters", {}))
    return wf, edit_spec


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rank_deletable_nodes(workflow: JSONDict) -> Dict[str, Dict[str, int]]:
    """
    Compute deterministic ranks among deletable nodes ONLY (non-sticky, non-trigger).
    Returns:
      ranks[name]["from_left"/"from_right"/"from_top"/"from_bottom"] = 1-based rank.
    """
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return {}
    rows: List[Tuple[int, int, str]] = []
    for n in nodes:
        if not isinstance(n, dict) or _is_sticky(n) or _is_trigger(n):
            continue
        name = n.get("name")
        pos = n.get("position")
        if not name or not isinstance(pos, list) or len(pos) < 2:
            continue
        try:
            x, y = int(pos[0]), int(pos[1])
        except Exception:
            continue
        rows.append((x, y, str(name)))

    ranks: Dict[str, Dict[str, int]] = {}
    if not rows:
        return ranks

    lr = sorted(rows, key=lambda t: (t[0], t[1], t[2]))  # (x, y, name)
    tb = sorted(rows, key=lambda t: (t[1], t[0], t[2]))  # (y, x, name)
    names_lr = [nm for _x, _y, nm in lr]
    names_tb = [nm for _x, _y, nm in tb]

    for i, nm in enumerate(names_lr, 1):
        ranks.setdefault(nm, {})["from_left"] = i
        ranks[nm]["from_right"] = len(names_lr) - i + 1
    for i, nm in enumerate(names_tb, 1):
        ranks.setdefault(nm, {})["from_top"] = i
        ranks[nm]["from_bottom"] = len(names_tb) - i + 1
    return ranks


@dataclass
class SampleClue:
    template_file: str
    input_sha256: str
    variant: str
    target_node: str
    target_type: str
    rank_from_left: Optional[int]
    rank_from_right: Optional[int]
    rank_from_top: Optional[int]
    rank_from_bottom: Optional[int]
    template_workflow_hash: str
    output_workflow_hash: str


def build_instruction(rng: random.Random, workflow: JSONDict, target_name: str) -> Tuple[str, str, Dict[str, Any]]:
    node = _node_by_name(workflow, target_name) or {}
    node_type = str(node.get("type") or "")
    ranks = _rank_deletable_nodes(workflow)
    r = ranks.get(target_name, {})

    # Guardrails for positional language:
    # - Only use left/right if x coordinates among deletable nodes are not all identical.
    # - Only use top/bottom if y coordinates among deletable nodes are not all identical.
    nodes = workflow.get("nodes")
    xs: set[int] = set()
    ys: set[int] = set()
    if isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict) or _is_sticky(n) or _is_trigger(n):
                continue
            pos = n.get("position")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            try:
                xs.add(int(pos[0]))
                ys.add(int(pos[1]))
            except Exception:
                continue
    allow_left_right = len(xs) > 1
    allow_top_bottom = len(ys) > 1

    # Ensure candidates are unambiguous
    deletables = [
        n for n in (workflow.get("nodes") or [])
        if isinstance(n, dict) and n.get("name") and (not _is_sticky(n)) and (not _is_trigger(n))
    ]
    type_counts: Dict[str, int] = {}
    for n in deletables:
        t = str(n.get("type") or "")
        type_counts[t] = type_counts.get(t, 0) + 1

    candidates: List[Tuple[str, str]] = []
    # Always safe / explicit
    candidates.append(("by_name", f'Remove the node "{target_name}".'))
    if node_type:
        candidates.append(("by_type_and_name", f'Remove the {node_type} node named "{target_name}".'))
    # by_type only if unique among deletables
    if node_type and type_counts.get(node_type, 0) == 1:
        candidates.append(("by_type", f'Remove the "{node_type}" node.'))

    # Position variants only if we can compute rank
    if allow_left_right and "from_left" in r:
        candidates.append(("from_left", f"Delete the {_ordinal(int(r['from_left']))} node from the left."))
    if allow_left_right and "from_right" in r:
        candidates.append(("from_right", f"Delete the {_ordinal(int(r['from_right']))} node from the right."))
    if allow_top_bottom and "from_top" in r:
        candidates.append(("from_top", f"Delete the {_ordinal(int(r['from_top']))} node from the top."))
    if allow_top_bottom and "from_bottom" in r:
        candidates.append(("from_bottom", f"Delete the {_ordinal(int(r['from_bottom']))} node from the bottom."))

    variant, text = rng.choice(candidates)
    meta = {
        "variant": variant,
        "target_node": target_name,
        "target_type": node_type,
        "ranks": r,
        "positional_guards": {
            "allow_left_right": allow_left_right,
            "allow_top_bottom": allow_top_bottom,
        },
    }
    return variant, text, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates-dir", type=str, default="n8n_workflow_generator_package/n8n_templates/training_data")
    ap.add_argument("--output-dir", type=str, default="n8n_workflow_generator_package/outputs/delete_training_dataset_v2")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples-per-template", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    templates_dir = Path(args.templates_dir)
    output_dir = Path(args.output_dir)

    template_files = _iter_template_files(templates_dir)
    rng.shuffle(template_files)
    template_files = template_files[: args.limit]

    records: List[JSONDict] = []
    clues: List[JSONDict] = []

    for path in template_files:
        try:
            template = _load_json(path)
        except Exception:
            continue
        wf = _extract_workflow_json(template)
        if not wf:
            continue
        # Skip templates that are already inconsistent (dangling connections) to keep training set clean.
        if not _workflow_is_connection_consistent(wf):
            continue
        nodes = wf.get("nodes")
        if not isinstance(nodes, list) or len(nodes) < 2:
            continue

        deletables: List[str] = []
        for n in nodes:
            if not isinstance(n, dict):
                continue
            if _is_sticky(n) or _is_trigger(n):
                continue
            nm = n.get("name")
            if nm:
                deletables.append(str(nm))
        if not deletables:
            continue

        for _ in range(max(1, args.examples_per_template)):
            target = rng.choice(deletables)
            node = _node_by_name(wf, target) or {}
            node_type = str(node.get("type") or "")

            variant, instruction, meta = build_instruction(rng, wf, target)
            edited, edit_spec = delete_node_strict(wf, target)

            user_content = "[TASK=delete]\n" + instruction + "\n\nTemplate:\n" + json.dumps(wf, ensure_ascii=False)
            assistant_content = json.dumps(edited, ensure_ascii=False)
            input_sha = _sha256_text(user_content)

            records.append(
                {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
            )

            ranks = meta.get("ranks") if isinstance(meta, dict) else {}
            clue = SampleClue(
                template_file=path.name,
                input_sha256=input_sha,
                variant=variant,
                target_node=target,
                target_type=node_type,
                rank_from_left=(int(ranks["from_left"]) if isinstance(ranks, dict) and "from_left" in ranks else None),
                rank_from_right=(int(ranks["from_right"]) if isinstance(ranks, dict) and "from_right" in ranks else None),
                rank_from_top=(int(ranks["from_top"]) if isinstance(ranks, dict) and "from_top" in ranks else None),
                rank_from_bottom=(int(ranks["from_bottom"]) if isinstance(ranks, dict) and "from_bottom" in ranks else None),
                template_workflow_hash=_sha256_json(wf),
                output_workflow_hash=_sha256_json(edited),
            )
            clue_obj = asdict(clue)
            clue_obj["edit_spec"] = edit_spec
            clue_obj["instruction"] = instruction
            clue_obj["template_connections_ref_count"] = len(_all_connection_refs(wf))
            clue_obj["output_connections_ref_count"] = len(_all_connection_refs(edited))
            clues.append(clue_obj)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_jsonl(records, output_dir / "train.jsonl")
    _save_jsonl(clues, output_dir / "oracle_clues.jsonl")
    print(f"Wrote {output_dir / 'train.jsonl'} ({len(records)})")
    print(f"Wrote {output_dir / 'oracle_clues.jsonl'} ({len(clues)})")


if __name__ == "__main__":
    main()

