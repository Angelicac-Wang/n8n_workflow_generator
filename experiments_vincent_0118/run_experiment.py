#!/usr/bin/env python3
"""
Run 0118_vincent Experiment

Generate workflows using the 0118_vincent pipeline, then evaluate node and
connection accuracy against ground truth templates.
"""

import argparse
import sys
from pathlib import Path

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from experiments_vincent_0118.experiment_pipeline import VincentExperimentPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run 0118_vincent experiment on n8n templates"
    )
    parser.add_argument(
        "--config",
        default="experiments_vincent_0118/config/experiment_config.yaml",
        help="Path to experiment config file",
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
    print("Vincent 0118 Experiment Pipeline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    if args.limit:
        print(f"Limit: {args.limit} templates")
    print("=" * 60)

    pipeline = VincentExperimentPipeline(args.config)

    print("\n[1/2] Generating workflows...")
    pipeline.run_generation(resume=args.resume, limit=args.limit)

    print("\n[2/2] Evaluating workflows...")
    pipeline.run_evaluation(limit=args.limit)

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()

