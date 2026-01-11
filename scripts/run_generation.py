#!/usr/bin/env python3
"""
Run LLM Workflow Generation

Generate workflows from templates using GPT-4o.

Usage:
    python scripts/run_generation.py [--resume] [--limit N]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.orchestration.evaluation_pipeline import EvaluationPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Generate LLM workflows from n8n templates'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already-generated templates'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit to N templates (for testing)'
    )
    parser.add_argument(
        '--config',
        default='evaluation/config/evaluation_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM Workflow Generation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    if args.limit:
        print(f"Limit: {args.limit} templates")
    print("="*60)

    # Initialize pipeline
    pipeline = EvaluationPipeline(args.config)

    # Run generation
    pipeline.run_generation(resume=args.resume, limit=args.limit)

    print("\n✓ Generation complete!")


if __name__ == '__main__':
    main()
