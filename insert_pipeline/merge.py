from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

JSONDict = Dict[str, Any]


def _find_node(wf: JSONDict, name: str) -> Optional[JSONDict]:
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return None
    for n in nodes:
        if isinstance(n, dict) and n.get("name") == name:
            return n
    return None


def _mid_position(left: Optional[JSONDict], right: Optional[JSONDict]) -> List[int]:
    def pos(n: Optional[JSONDict]) -> Optional[Tuple[int, int]]:
        if not n or not isinstance(n.get("position"), list) or len(n["position"]) < 2:
            return None
        try:
            return int(n["position"][0]), int(n["position"][1])
        except Exception:
            return None

    pl, pr = pos(left), pos(right)
    if pl and pr:
        return [int((pl[0] + pr[0]) / 2), int((pl[1] + pr[1]) / 2)]
    if pl:
        return [pl[0] + 200, pl[1]]
    if pr:
        return [pr[0] - 200, pr[1]]
    return [0, 0]


def _outgoing_targets(wf: JSONDict, source: str) -> List[str]:
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


def _incoming_sources(wf: JSONDict, target: str) -> List[str]:
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


def _has_main_edge(wf: JSONDict, src: str, dst: str) -> bool:
    return dst in _outgoing_targets(wf, src)


def _retarget_outgoing(wf: JSONDict, source: str, old_target: str, new_target: str) -> bool:
    conns = wf.get("connections")
    if not isinstance(conns, dict):
        return False
    block = conns.get(source)
    if not isinstance(block, dict):
        return False
    main = block.get("main")
    if not isinstance(main, list):
        return False
    changed = False
    for group in main:
        if not isinstance(group, list):
            continue
        for link in group:
            if isinstance(link, dict) and str(link.get("node")) == old_target:
                link["node"] = new_target
                if "type" not in link:
                    link["type"] = "main"
                if "index" not in link:
                    link["index"] = 0
                changed = True
    return changed


def _ensure_main_output(wf: JSONDict, node_name: str, target: str) -> None:
    conns = wf.setdefault("connections", {})
    if node_name not in conns:
        conns[node_name] = {"main": [[]]}
    block = conns[node_name]
    if not isinstance(block, dict):
        block = {"main": [[]]}
        conns[node_name] = block
    main = block.setdefault("main", [[]])
    if not main or not isinstance(main[0], list):
        main.clear()
        main.append([])
    main[0].append({"node": target, "type": "main", "index": 0})


def _finalize_node_shell(
    *,
    name: str,
    node_type: str,
    parameters: JSONDict,
    position: List[int],
    type_version: Optional[float] = None,
) -> JSONDict:
    tv = type_version if type_version is not None else 1
    node: JSONDict = {
        "id": str(uuid.uuid4()),
        "name": name,
        "type": node_type,
        "typeVersion": tv,
        "position": position,
        "parameters": parameters,
    }
    if "webhook" in node_type.lower():
        path = parameters.get("path")
        wid = parameters.get("webhookId") if isinstance(parameters.get("webhookId"), str) else None
        if not wid:
            wid = path if isinstance(path, str) and path else str(uuid.uuid4())
            node["webhookId"] = wid
            if not isinstance(path, str) or not path:
                node["parameters"] = dict(parameters)
                node["parameters"]["path"] = wid
    return node


def resolve_splice_endpoints(
    wf: JSONDict,
    loc: JSONDict,
) -> Optional[Tuple[str, str]]:
    kind = str(loc.get("kind") or "")
    if kind == "between":
        pair = loc.get("between")
        if isinstance(pair, list) and len(pair) == 2:
            a, b = str(pair[0]), str(pair[1])
            if _has_main_edge(wf, a, b):
                return a, b
    elif kind == "after":
        a = loc.get("after")
        if isinstance(a, str):
            tgts = _outgoing_targets(wf, a)
            if len(tgts) == 1:
                return a, tgts[0]
    elif kind == "before":
        b = loc.get("before")
        if isinstance(b, str):
            srcs = _incoming_sources(wf, b)
            if len(srcs) == 1:
                return srcs[0], b
    return None


def apply_insert_splice(
    template: JSONDict,
    *,
    new_node_name: str,
    node_type: str,
    parameters: JSONDict,
    location: JSONDict,
    type_version: Optional[float] = None,
) -> JSONDict:
    """
    Deep-copy ``template``, append a new node, and splice it on ``main`` between
    resolved (left, right) endpoints. Fails soft: if endpoints cannot be resolved,
    only appends the node without connection changes.
    """
    wf = copy.deepcopy(template)
    nodes = wf.setdefault("nodes", [])
    if not isinstance(nodes, list):
        wf["nodes"] = []
        nodes = wf["nodes"]

    ends = resolve_splice_endpoints(wf, location)
    left_n = _find_node(wf, ends[0]) if ends else None
    right_n = _find_node(wf, ends[1]) if ends else None
    pos = _mid_position(left_n, right_n)

    new_node = _finalize_node_shell(
        name=new_node_name,
        node_type=node_type,
        parameters=parameters,
        position=pos,
        type_version=type_version,
    )
    nodes.append(new_node)

    if not ends:
        return wf
    left, right = ends
    ok = _retarget_outgoing(wf, left, right, new_node_name)
    if ok:
        _ensure_main_output(wf, new_node_name, right)
    return wf
