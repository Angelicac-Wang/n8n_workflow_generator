#!/usr/bin/env python3
"""
Convert Gemma modify infer_outputs.jsonl -> predictions.jsonl for re_eval_modify_predictions.py.

Expects each line to include:
  - input: raw task string (inference.py sets this from raw_task_input)
  - output: model generation (may include markdown / extra text; first JSON object is extracted)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))
import run_modify_inference_and_eval as rm  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-jsonl", type=str, required=True)
    ap.add_argument("--out-jsonl", type=str, required=True)
    args = ap.parse_args()

    inf = Path(args.infer_jsonl)
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out.open("w", encoding="utf-8") as wf:
        for d in _jsonl_iter(inf):
            raw = d.get("input")
            if not isinstance(raw, str):
                rt = d.get("raw_task_input")
                raw = rt if isinstance(rt, str) else ""
            text = d.get("output")
            if not isinstance(text, str):
                text = ""
            pred = rm._extract_first_json_object(text)
            rec: JSONDict = {
                "input_sha256": _sha256_text(raw),
                "input": raw,
                "prediction": pred,
                "raw_text": text,
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} lines to {out}")


if __name__ == "__main__":
    main()
