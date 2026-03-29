#!/usr/bin/env python3
"""
Evaluate only successful generated workflows.

Filters generated outputs to those with error == None and llm_response present,
then evaluates the first N templates.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.orchestration.evaluation_pipeline import EvaluationPipeline
from evaluation.evaluators.llm_json_validity import llm_output_json_validity_metrics
from evaluation.utils.result_saver import ResultSaver
from n8n_workflow_recommender.utils.file_loader import load_json


def _extract_template_id(template: Dict) -> str:
    if 'metadata' in template:
        return str(template['metadata']['id'])
    if 'id' in template:
        return str(template['id'])
    return ""


def _load_successful_generated(generated_dir: Path) -> List[Dict]:
    generated = []
    for path in sorted(generated_dir.glob("generated_*.json")):
        try:
            data = load_json(str(path))
        except Exception:
            continue
        if data.get("error") is None and data.get("llm_response"):
            generated.append(data)
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate successful generated workflows only"
    )
    parser.add_argument(
        "--config",
        default="evaluation/config/evaluation_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of successful templates to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_ft_test_subset_100",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Directory containing generated_*.json files",
    )

    args = parser.parse_args()

    pipeline = EvaluationPipeline(args.config)
    result_saver = ResultSaver(args.output_dir)

    generated_dir = Path(
        args.generated_dir
        if args.generated_dir
        else Path(pipeline.config["output_dir"]) / "llm_generated_workflows"
    )

    print("\n" + "=" * 60)
    print("Evaluating Successful Generated Workflows")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Generated dir: {generated_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Limit: {args.limit}")
    print("=" * 60)

    # Load templates
    templates = pipeline.template_loader.load_all_templates()
    template_map = {}
    for template in templates:
        template_id = _extract_template_id(template)
        if template_id:
            template_map[template_id] = template

    # Load successful generated outputs
    successful_generated = _load_successful_generated(generated_dir)
    selected = successful_generated[: args.limit]

    print(f"\nFound {len(successful_generated)} successful generations")
    print(f"Evaluating {len(selected)} templates\n")

    all_results = []
    for i, generated_data in enumerate(selected):
        template_id = str(generated_data.get("template_id"))
        template = template_map.get(template_id)
        if not template:
            continue

        json_metrics = llm_output_json_validity_metrics(generated_data)

        # Normalize workflows
        gt_workflow = pipeline.normalizer.normalize_ground_truth(template)
        llm_workflow = pipeline.normalizer.normalize_llm_output(
            generated_data["llm_response"]
        )

        # Match nodes
        matching_result = pipeline.node_matcher.match_nodes(
            gt_workflow["nodes"], llm_workflow["nodes"]
        )

        # Evaluate metrics
        node_metrics = pipeline.node_evaluator.evaluate_node_types(matching_result)
        connection_metrics = pipeline.node_evaluator.evaluate_connections(
            gt_workflow, llm_workflow
        )
        param_metrics = pipeline._evaluate_parameters_safe(matching_result)
        cost_metrics = pipeline.cost_tracker.calculate_cost(
            generated_data["usage"],
            model=pipeline.config.get("model", "gpt-4o"),
        )

        template_result = {
            "template_id": template_id,
            "template_name": template.get("workflow", {}).get("name", ""),
            "error": None,
            "metrics": {
                **node_metrics,
                **connection_metrics,
                **param_metrics,
                **cost_metrics,
                **json_metrics,
                "usage": generated_data["usage"],
            },
        }

        all_results.append(template_result)

        if (i + 1) % 10 == 0 or (i + 1) == len(selected):
            print(f"[{i + 1}/{len(selected)}] evaluated")

    summary_stats = pipeline._compute_summary_statistics(all_results)
    cost_report = pipeline._compute_cost_report(all_results)

    result_saver.save_evaluation_results(
        detailed_results=all_results,
        summary_stats=summary_stats,
        cost_report=cost_report,
    )

    print(f"\nResults saved to: {result_saver.eval_results_dir}")
    pipeline._print_summary(summary_stats, cost_report)


if __name__ == "__main__":
    main()
