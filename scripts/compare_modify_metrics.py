#!/usr/bin/env python3
"""Print a side-by-side summary of two modify metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_a", type=str, help="First metrics.json (e.g. fine-tuned run)")
    ap.add_argument("metrics_b", type=str, help="Second metrics.json (e.g. gpt-4.1 run)")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    args = ap.parse_args()

    a = _load(Path(args.metrics_a))
    b = _load(Path(args.metrics_b))
    n_a = a.get("n") or 0
    n_b = b.get("n") or 0

    def rate(m: Dict[str, Any], key: str) -> float:
        r = m.get("rates") or {}
        return float(r.get(key, 0.0))

    def count_also(m: Dict[str, Any], key: str) -> int:
        also = m.get("also") or {}
        return int(also.get(key, 0))

    keys = [
        ("phrase_ok", "phrase_ok"),
        ("param_updated_ok", "param_updated_ok"),
        ("consistent_ok", "consistent_ok"),
    ]
    print(f"n: {args.label_a}={n_a}  {args.label_b}={n_b}")
    print()
    print(f"{'metric':<22} {args.label_a:>12} {args.label_b:>12}  (delta B-A)")
    for label, k in keys:
        ra, rb = rate(a, k), rate(b, k)
        print(f"{label:<22} {ra:>11.1%} {rb:>11.1%}  {rb - ra:+.1%}")

    print()
    print("also.strict_ok (count):", count_also(a, "strict_ok"), count_also(b, "strict_ok"))
    print()
    print("error_breakdown A:", json.dumps(a.get("error_breakdown") or {}, ensure_ascii=False))
    print("error_breakdown B:", json.dumps(b.get("error_breakdown") or {}, ensure_ascii=False))


if __name__ == "__main__":
    main()
