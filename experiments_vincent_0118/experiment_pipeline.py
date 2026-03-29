#!/usr/bin/env python3
"""
Vincent 0118 Experiment Pipeline

Generate workflows using the 0118_vincent method and evaluate node/connection
accuracy using the existing evaluation utilities.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
import numpy as np

# Add package root to path
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from n8n_workflow_recommender.utils.file_loader import load_yaml
from evaluation.utils.template_loader import TemplateLoader
from evaluation.utils.result_saver import ResultSaver
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.parameter_evaluator import ParameterEvaluator
from evaluation.evaluators.cost_tracker import CostTracker
from evaluation.orchestration.progress_tracker import ProgressTracker

from experiments_vincent_0118.vincent_generator import Vincent0118Generator


class VincentExperimentPipeline:
    """
    Pipeline: 0118_vincent generation + evaluation.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        self.template_loader = TemplateLoader(self.config["templates_dir"])
        self.result_saver = ResultSaver(self.config["output_dir"])
        self.generator = Vincent0118Generator(
            openai_api_key=self.config["openai_key"],
            taxonomy_path=self.config["taxonomy_path"],
            mf_model_dir=self.config["mf_model_dir"],
            vincent_dir=self.config["vincent_dir"],
            kg_path=self.config.get("kg_path"),
            mentor_enabled=self.config.get("mentor_enabled", True),
            mcts_query_source=self.config.get("mcts_query_source", "original"),
            intent_keywords_source=self.config.get("intent_keywords_source", "llm"),
            max_retries=self.config.get("max_retries", 2),
            retry_delay=self.config.get("retry_delay", 2.0),
        )

        self.normalizer = WorkflowNormalizer()
        self.node_matcher = NodeMatcher()
        self.node_evaluator = NodeAccuracyEvaluator()
        self.param_evaluator = ParameterEvaluator(
            model_name=self.config["embedding_model"],
            threshold=self.config["param_similarity_threshold"],
        )
        self.cost_tracker = CostTracker()

    def run_generation(self, resume: bool = True, limit: Optional[int] = None):
        print("\n" + "=" * 60)
        print("Vincent 0118 Generation")
        print("=" * 60)

        templates = self.template_loader.load_all_templates()
        if limit:
            templates = templates[:limit]

        tracker = ProgressTracker(len(templates), "Vincent Generation")

        for i, template in enumerate(templates):
            template_id = self._get_template_id(template, i, tracker)
            if template_id is None:
                continue

            if resume and self.result_saver.workflow_exists(template_id):
                tracker.log(f"Skipping {template_id} (already exists)")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            description = self.template_loader.extract_description(template)
            if not description:
                tracker.log(f"Skipping {template_id} (no description)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            tracker.log(f"Generating workflow for template {template_id}")
            result = self.generator.generate_workflow(description, template_id)
            self.result_saver.save_generated_workflow(result)

            if result.get("error"):
                tracker.log(f"Error for {template_id}: {result['error']}", "ERROR")
                tracker.increment_error()

            time.sleep(self.config.get("api_delay", 0.5))
            tracker.update(i + 1)

        tracker.complete()

    def run_evaluation(self, limit: Optional[int] = None):
        print("\n" + "=" * 60)
        print("Vincent 0118 Evaluation")
        print("=" * 60)

        templates = self.template_loader.load_all_templates()
        if limit:
            templates = templates[:limit]
        tracker = ProgressTracker(len(templates), "Vincent Evaluation")
        all_results = []

        for i, template in enumerate(templates):
            template_id = self._get_template_id(template, i, tracker)
            if template_id is None:
                continue

            if not self.result_saver.workflow_exists(template_id):
                tracker.log(f"Skipping {template_id} (not generated)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            generated_data = self.result_saver.load_generated_workflow(template_id)
            if generated_data.get("error"):
                all_results.append({
                    "template_id": template_id,
                    "template_name": template.get("workflow", {}).get("name", ""),
                    "error": generated_data["error"],
                    "metrics": None,
                })
                tracker.increment_error()
                tracker.update(i + 1)
                continue

            tracker.log(f"Evaluating template {template_id}")

            gt_workflow = self.normalizer.normalize_ground_truth(template)
            llm_workflow = self.normalizer.normalize_llm_output(generated_data["llm_response"])

            matching_result = self.node_matcher.match_nodes(
                gt_workflow["nodes"],
                llm_workflow["nodes"],
            )

            node_metrics = self.node_evaluator.evaluate_node_types(matching_result)
            connection_metrics = self.node_evaluator.evaluate_connections(gt_workflow, llm_workflow)
            param_metrics = self.param_evaluator.evaluate_parameters(matching_result)
            usage = generated_data.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            cost_metrics = self.cost_tracker.calculate_cost(usage)

            template_result = {
                "template_id": template_id,
                "template_name": template.get("workflow", {}).get("name", ""),
                "error": None,
                "metrics": {
                    **node_metrics,
                    **connection_metrics,
                    **param_metrics,
                    **cost_metrics,
                    "usage": usage,
                },
            }

            all_results.append(template_result)
            tracker.update(i + 1)

        tracker.complete()

        print("\n" + "=" * 60)
        print("Saving Evaluation Results")
        print("=" * 60)

        summary_stats = self._compute_summary_statistics(all_results)
        cost_report = self._compute_cost_report(all_results)

        self.result_saver.save_evaluation_results(
            detailed_results=all_results,
            summary_stats=summary_stats,
            cost_report=cost_report,
        )

        print(f"\nResults saved to: {self.result_saver.eval_results_dir}")
        self._print_summary(summary_stats, cost_report)

    def _get_template_id(self, template: Dict, index: int, tracker: ProgressTracker) -> Optional[str]:
        if "metadata" in template:
            return str(template["metadata"].get("id"))
        if "id" in template:
            return str(template.get("id"))

        tracker.log(f"Skipping template {index} (no ID found)", "WARNING")
        tracker.increment_skipped()
        tracker.update(index + 1)
        return None

    def _load_config(self, config_path: str) -> Dict:
        config = load_yaml(config_path)

        if "${OPENAI_API_KEY}" in config.get("openai_key", ""):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                main_config_path = PACKAGE_ROOT / "n8n_workflow_recommender" / "config" / "config.yaml"
                if main_config_path.exists():
                    main_config = load_yaml(str(main_config_path))
                    api_key = main_config.get("api", {}).get("openai_key")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment or config")
            config["openai_key"] = api_key

        config = self._resolve_paths(config)
        return config

    def _resolve_paths(self, config: Dict) -> Dict:
        path_keys = [
            "templates_dir",
            "output_dir",
            "taxonomy_path",
            "mf_model_dir",
            "vincent_dir",
            "kg_path",
        ]

        resolved = dict(config)
        for key in path_keys:
            value = resolved.get(key)
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (PACKAGE_ROOT / path).resolve()
            resolved[key] = str(path)

        return resolved

    def _compute_summary_statistics(self, results: List[Dict]) -> Dict:
        valid_results = [r for r in results if r.get("metrics")]

        if not valid_results:
            return {
                "total_templates": len(results),
                "successful_evaluations": 0,
                "failed_evaluations": len(results),
                "node_accuracy": {
                    "mean_f1": 0.0,
                    "median_f1": 0.0,
                    "std_f1": 0.0,
                    "min_f1": 0.0,
                    "max_f1": 0.0,
                },
                "connection_accuracy": {
                    "mean_f1": 0.0,
                    "median_f1": 0.0,
                    "std_f1": 0.0,
                    "min_f1": 0.0,
                    "max_f1": 0.0,
                },
                "parameter_accuracy": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "error": "No valid results to aggregate",
            }

        node_f1s = [r["metrics"]["node_type_f1"] for r in valid_results]
        conn_f1s = [r["metrics"]["connection_f1"] for r in valid_results]
        param_accs = [r["metrics"]["avg_parameter_accuracy"] for r in valid_results]

        return {
            "total_templates": len(results),
            "successful_evaluations": len(valid_results),
            "failed_evaluations": len(results) - len(valid_results),
            "node_accuracy": {
                "mean_f1": float(np.mean(node_f1s)),
                "median_f1": float(np.median(node_f1s)),
                "std_f1": float(np.std(node_f1s)),
                "min_f1": float(np.min(node_f1s)),
                "max_f1": float(np.max(node_f1s)),
            },
            "connection_accuracy": {
                "mean_f1": float(np.mean(conn_f1s)),
                "median_f1": float(np.median(conn_f1s)),
                "std_f1": float(np.std(conn_f1s)),
                "min_f1": float(np.min(conn_f1s)),
                "max_f1": float(np.max(conn_f1s)),
            },
            "parameter_accuracy": {
                "mean": float(np.mean(param_accs)),
                "median": float(np.median(param_accs)),
                "std": float(np.std(param_accs)),
                "min": float(np.min(param_accs)),
                "max": float(np.max(param_accs)),
            },
        }

    def _compute_cost_report(self, results: List[Dict]) -> Dict:
        results_with_usage = [r for r in results if r.get("metrics") and r["metrics"].get("usage")]
        return self.cost_tracker.aggregate_costs(results_with_usage)

    def _print_summary(self, summary_stats: Dict, cost_report: Dict):
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        print(f"\nTotal Templates: {summary_stats['total_templates']}")
        print(f"Successful: {summary_stats['successful_evaluations']}")
        print(f"Failed: {summary_stats['failed_evaluations']}")

        if summary_stats.get("error"):
            print(f"\nWarning: {summary_stats['error']}")
        else:
            print("\nNode Type Accuracy (F1):")
            print(f"  Mean: {summary_stats['node_accuracy']['mean_f1']:.3f}")
            print(f"  Median: {summary_stats['node_accuracy']['median_f1']:.3f}")
            print(f"  Std: {summary_stats['node_accuracy']['std_f1']:.3f}")

            print("\nConnection Accuracy (F1):")
            print(f"  Mean: {summary_stats['connection_accuracy']['mean_f1']:.3f}")
            print(f"  Median: {summary_stats['connection_accuracy']['median_f1']:.3f}")

            print("\nParameter Accuracy:")
            print(f"  Mean: {summary_stats['parameter_accuracy']['mean']:.3f}")
            print(f"  Median: {summary_stats['parameter_accuracy']['median']:.3f}")

        print("\nCost Report:")
        print(f"  Total Tokens: {cost_report['total_tokens']:,}")
        print(f"  Total Cost: ${cost_report['total_cost']:.2f} USD")
        if cost_report["templates_with_cost"] > 0:
            print(f"  Avg Cost per Template: ${cost_report['avg_cost_per_template']:.4f} USD")
        else:
            print("  Avg Cost per Template: $0.0000 USD (no successful generations)")

        print("=" * 60 + "\n")

