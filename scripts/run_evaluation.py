#!/usr/bin/env python3
"""
Run Evaluation Only

Evaluate existing LLM-generated workflows against ground truth.

Usage:
    python scripts/run_evaluation.py
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.orchestration.evaluation_pipeline import EvaluationPipeline
from evaluation.visualization.report_generator import ReportGenerator
from n8n_workflow_recommender.utils.file_loader import load_yaml, load_json


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LLM-generated workflows'
    )
    parser.add_argument(
        '--config',
        default='evaluation/config/evaluation_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM Workflow Evaluation")
    print("="*60)
    print(f"Config: {args.config}")
    print("="*60)

    # Initialize pipeline
    pipeline = EvaluationPipeline(args.config)

    # Run evaluation
    pipeline.run_evaluation()

    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    # Load results
    results_path = Path(pipeline.config['output_dir']) / 'evaluation_results' / 'detailed_per_template.json'
    results = load_json(str(results_path))

    # Generate visualizations
    viz_generator = ReportGenerator(pipeline.config)
    viz_dir = Path(pipeline.config['output_dir']) / 'visualizations'

    viz_generator.generate_all_visualizations(results, viz_dir)

    print(f"\n✓ Evaluation complete! Results saved to:")
    print(f"  - {pipeline.result_saver.eval_results_dir}")
    print(f"  - {viz_dir}")


if __name__ == '__main__':
    main()
