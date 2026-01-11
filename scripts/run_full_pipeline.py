#!/usr/bin/env python3
"""
Run Full Evaluation Pipeline

Generate workflows, evaluate, and visualize results.

Usage:
    python scripts/run_full_pipeline.py [--resume] [--limit N]
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.orchestration.evaluation_pipeline import EvaluationPipeline
from evaluation.visualization.report_generator import ReportGenerator
from n8n_workflow_recommender.utils.file_loader import load_json


def main():
    parser = argparse.ArgumentParser(
        description='Run full LLM workflow evaluation pipeline'
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
    print("LLM Workflow Evaluation Pipeline")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume}")
    if args.limit:
        print(f"Limit: {args.limit} templates")
    print("="*60)

    # Initialize pipeline
    pipeline = EvaluationPipeline(args.config)

    # Step 1: Generate workflows
    print("\n[1/3] Generating workflows...")
    pipeline.run_generation(resume=args.resume, limit=args.limit)

    # Step 2: Evaluate workflows
    print("\n[2/3] Evaluating workflows...")
    pipeline.run_evaluation()

    # Step 3: Generate visualizations
    print("\n[3/3] Generating visualizations...")

    # Load results
    results_path = Path(pipeline.config['output_dir']) / 'evaluation_results' / 'detailed_per_template.json'
    results = load_json(str(results_path))

    # Generate visualizations
    viz_generator = ReportGenerator(pipeline.config)
    viz_dir = Path(pipeline.config['output_dir']) / 'visualizations'

    viz_generator.generate_all_visualizations(results, viz_dir)

    # Final summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. Generated Workflows: {pipeline.result_saver.llm_workflows_dir}")
    print(f"  2. Evaluation Results: {pipeline.result_saver.eval_results_dir}")
    print(f"  3. Visualizations: {viz_dir}")
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
