#!/usr/bin/env python3
"""
Prepare delete inference input: messages format with hints appended.
Reads delete_testing_data_messages.jsonl, appends hints to each user content,
writes valid.jsonl for gemma3-27b inference.
"""

import json
import sys
from pathlib import Path

# Add scripts dir for import
_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from run_delete_inference_and_eval import build_hints

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="n8n_workflow_generator_package/outputs/edit_testing_data_training_style/delete_testing_data_messages.jsonl")
    ap.add_argument("--output", default="iService/gemma3-27b/delete_inference/data/valid.jsonl")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = rec.get("messages", [])
            if not msgs or msgs[0].get("role") != "user":
                continue
            user_content = msgs[0].get("content", "")
            hints = build_hints(user_content)
            if hints:
                user_content = user_content.rstrip() + "\n" + hints
            records.append({"messages": [{"role": "user", "content": user_content}]})

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(records)} samples)")

if __name__ == "__main__":
    main()
