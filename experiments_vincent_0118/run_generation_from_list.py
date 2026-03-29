#!/usr/bin/env python3
"""
Run 0118_vincent generation for a list of templates.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from experiments_vincent_0118.experiment_pipeline import VincentExperimentPipeline
from evaluation.orchestration.progress_tracker import ProgressTracker
from n8n_workflow_recommender.utils.file_loader import load_json


def read_template_list(list_path: str) -> List[str]:
    path = Path(list_path)
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {path}")

    names: List[str] = []
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".csv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return []

            if "template" in header:
                idx = header.index("template")
            else:
                idx = 0
                first = header[idx].strip()
                if first:
                    names.append(first)

            for row in reader:
                if len(row) <= idx:
                    continue
                name = row[idx].strip()
                if name:
                    names.append(name)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def run_generation_for_list(
    pipeline: VincentExperimentPipeline,
    template_files: List[str],
    resume: bool = True,
    limit: int | None = None,
):
    if limit:
        template_files = template_files[:limit]

    tracker = ProgressTracker(len(template_files), "Vincent Generation (List)")
    templates_dir = Path(pipeline.config["templates_dir"])

    for i, filename in enumerate(template_files):
        template_path = templates_dir / filename
        if not template_path.exists():
            tracker.log(f"Skipping {filename} (file not found)", "WARNING")
            tracker.increment_skipped()
            tracker.update(i + 1)
            continue

        try:
            template = load_json(str(template_path))
        except Exception as exc:
            tracker.log(f"Skipping {filename} (load error: {exc})", "WARNING")
            tracker.increment_skipped()
            tracker.update(i + 1)
            continue

        template_id = pipeline._get_template_id(template, i, tracker)
        if template_id is None:
            continue

        if resume and pipeline.result_saver.workflow_exists(template_id):
            tracker.log(f"Skipping {template_id} (already exists)")
            tracker.increment_skipped()
            tracker.update(i + 1)
            continue

        description = pipeline.template_loader.extract_description(template)
        if not description:
            tracker.log(f"Skipping {template_id} (no description)", "WARNING")
            tracker.increment_skipped()
            tracker.update(i + 1)
            continue

        tracker.log(f"Generating workflow for template {template_id}")
        result = pipeline.generator.generate_workflow(description, template_id)
        pipeline.result_saver.save_generated_workflow(result)

        if result.get("error"):
            tracker.log(f"Error for {template_id}: {result['error']}", "ERROR")
            tracker.increment_error()

        time.sleep(pipeline.config.get("api_delay", 0.5))
        tracker.update(i + 1)

    tracker.complete()


def main():
    parser = argparse.ArgumentParser(
        description="Run 0118_vincent generation for a template list"
    )
    parser.add_argument(
        "--config",
        default="experiments_vincent_0118/config/experiment_config_original.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--list-file",
        default="outputs/vincent_0118_experiment_original/short_templates_lt10_nodes.tsv",
        help="Path to TSV/CSV list of templates",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-generated templates",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N templates (for testing)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Vincent 0118 Generation (List)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"List: {args.list_file}")
    print(f"Resume: {args.resume}")
    if args.limit:
        print(f"Limit: {args.limit} templates")
    print("=" * 60)

    pipeline = VincentExperimentPipeline(args.config)
    template_files = read_template_list(args.list_file)
    if not template_files:
        raise ValueError(f"No templates found in list file: {args.list_file}")

    run_generation_for_list(
        pipeline=pipeline,
        template_files=template_files,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
