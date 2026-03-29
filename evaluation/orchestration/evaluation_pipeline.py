#!/usr/bin/env python3
"""
Evaluation Pipeline

Main orchestrator for LLM workflow generation and evaluation.
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from n8n_workflow_recommender.utils.file_loader import load_yaml
from evaluation.generators.llm_workflow_generator import LLMWorkflowGenerator
from evaluation.generators.prompt_builder import PromptBuilder
from evaluation.utils.template_loader import TemplateLoader
from evaluation.utils.result_saver import ResultSaver
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.llm_json_validity import llm_output_json_validity_metrics
from evaluation.evaluators.cost_tracker import CostTracker
from evaluation.orchestration.progress_tracker import ProgressTracker


class EvaluationPipeline:
    """
    Main pipeline for LLM workflow generation and evaluation
    """

    def __init__(self, config_path: str):
        """
        Initialize evaluation pipeline

        Args:
            config_path: Path to evaluation config file
        """
        self.config = self._load_config(config_path)

        # Initialize components
        self.template_loader = TemplateLoader(self.config['templates_dir'])
        self.result_saver = ResultSaver(self.config['output_dir'])
        self.prompt_builder = None
        if not self.config.get('use_prompt_id', False):
            self.prompt_builder = PromptBuilder(self.config['prompt_template_path'])

        self.llm_generator = LLMWorkflowGenerator(
            openai_api_key=self.config['openai_key'],
            prompt_builder=self.prompt_builder,
            model=self.config['model'],
            max_retries=self.config.get('max_retries', 3),
            retry_delay=self.config.get('retry_delay', 2.0),
            use_prompt_id=self.config.get('use_prompt_id', False),
            prompt_id=self.config.get('prompt_id'),
            prompt_version=self.config.get('prompt_version'),
            openai_project=self.config.get('openai_project'),
            openai_organization=self.config.get('openai_organization'),
            max_output_tokens=self.config.get('max_output_tokens')
        )

        self.normalizer = WorkflowNormalizer()
        self.node_matcher = NodeMatcher()
        self.node_evaluator = NodeAccuracyEvaluator()
        # Lazy init: ParameterEvaluator requires optional heavy deps
        self.param_evaluator = None
        self.cost_tracker = CostTracker()

    def run_generation(self, resume: bool = True, limit: Optional[int] = None):
        """
        Generate workflows for all templates

        Args:
            resume: If True, skip already-generated templates
            limit: Optional limit on number of templates to process
        """
        print("\n" + "="*60)
        print("Phase 1: LLM Workflow Generation")
        print("="*60)

        # Load templates
        templates = self.template_loader.load_all_templates()

        if limit:
            templates = templates[:limit]

        tracker = ProgressTracker(len(templates), "Workflow Generation")

        for i, template in enumerate(templates):
            # Handle different template structures
            if 'metadata' in template:
                template_id = str(template['metadata']['id'])
            elif 'id' in template:
                template_id = str(template['id'])
            else:
                tracker.log(f"Skipping template {i} (no ID found)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            # Resume: skip if already exists
            if resume and self.result_saver.workflow_exists(template_id):
                tracker.log(f"Skipping {template_id} (already exists)")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            # Extract description
            description = self.template_loader.extract_description(template)

            if not description:
                tracker.log(f"Skipping {template_id} (no description)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            # Generate workflow
            tracker.log(f"Generating workflow for template {template_id}")
            result = self.llm_generator.generate_workflow(description, template_id)

            # Save result
            self.result_saver.save_generated_workflow(result)

            # Check for errors
            if result.get('error'):
                tracker.log(f"Error for {template_id}: {result['error']}", "ERROR")
                tracker.increment_error()

            # Rate limiting
            time.sleep(self.config.get('api_delay', 0.5))

            # Update progress
            tracker.update(i + 1)

        tracker.complete()

    def run_evaluation(self):
        """
        Evaluate generated workflows against ground truth
        """
        print("\n" + "="*60)
        print("Phase 2: Workflow Evaluation")
        print("="*60)

        # Load templates
        templates = self.template_loader.load_all_templates()
        tracker = ProgressTracker(len(templates), "Workflow Evaluation")

        all_results = []

        for i, template in enumerate(templates):
            # Handle different template structures
            if 'metadata' in template:
                template_id = str(template['metadata']['id'])
            elif 'id' in template:
                template_id = str(template['id'])
            else:
                tracker.log(f"Skipping template {i} (no ID found)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            # Check if workflow was generated
            if not self.result_saver.workflow_exists(template_id):
                tracker.log(f"Skipping {template_id} (not generated)", "WARNING")
                tracker.increment_skipped()
                tracker.update(i + 1)
                continue

            # Load generated workflow
            generated_data = self.result_saver.load_generated_workflow(template_id)
            json_metrics = llm_output_json_validity_metrics(generated_data)

            # Check for generation errors
            if generated_data.get('error'):
                all_results.append({
                    "template_id": template_id,
                    "template_name": template.get('workflow', {}).get('name', ''),
                    "error": generated_data['error'],
                    "metrics": dict(json_metrics),
                })
                tracker.increment_error()
                tracker.update(i + 1)
                continue

            tracker.log(f"Evaluating template {template_id}")

            # Normalize workflows
            gt_workflow = self.normalizer.normalize_ground_truth(template)
            llm_workflow = self.normalizer.normalize_llm_output(generated_data['llm_response'])

            # Match nodes
            matching_result = self.node_matcher.match_nodes(
                gt_workflow['nodes'],
                llm_workflow['nodes']
            )

            # Evaluate metrics
            node_metrics = self.node_evaluator.evaluate_node_types(matching_result)
            connection_metrics = self.node_evaluator.evaluate_connections(gt_workflow, llm_workflow)
            param_metrics = self._evaluate_parameters_safe(matching_result)
            cost_metrics = self.cost_tracker.calculate_cost(
                generated_data['usage'],
                model=self.config.get('model', 'gpt-4o')
            )

            # Combine results
            template_result = {
                "template_id": template_id,
                "template_name": template.get('workflow', {}).get('name', ''),
                "error": None,
                "metrics": {
                    **node_metrics,
                    **connection_metrics,
                    **param_metrics,
                    **cost_metrics,
                    **json_metrics,
                    "usage": generated_data['usage']
                }
            }

            all_results.append(template_result)
            tracker.update(i + 1)

        tracker.complete()

        # Save results
        print("\n" + "="*60)
        print("Saving Evaluation Results")
        print("="*60)

        summary_stats = self._compute_summary_statistics(all_results)
        cost_report = self._compute_cost_report(all_results)

        self.result_saver.save_evaluation_results(
            detailed_results=all_results,
            summary_stats=summary_stats,
            cost_report=cost_report
        )

        print(f"\nResults saved to: {self.result_saver.eval_results_dir}")
        self._print_summary(summary_stats, cost_report)

    def _evaluate_parameters_safe(self, matching_result: Dict) -> Dict:
        """
        Evaluate parameter metrics, but gracefully degrade when optional deps
        (sentence-transformers / sklearn / torch) are not installed.
        """
        if self.param_evaluator is None:
            try:
                from evaluation.evaluators.parameter_evaluator import ParameterEvaluator

                self.param_evaluator = ParameterEvaluator(
                    model_name=self.config['embedding_model'],
                    threshold=self.config['param_similarity_threshold']
                )
            except Exception as e:
                print(
                    f"Warning: parameter evaluation skipped (optional deps missing): {e}"
                )
                return {
                    "avg_parameter_accuracy": 0.0,
                    "per_node_accuracy": [],
                }

        return self.param_evaluator.evaluate_parameters(matching_result)

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and process configuration

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        config = load_yaml(config_path)

        # Resolve OpenAI API key
        if "${OPENAI_API_KEY}" in config['openai_key']:
            api_key = os.getenv('OPENAI_API_KEY')

            # Fallback to main config
            if not api_key:
                main_config_path = Path(__file__).parent.parent.parent / \
                    'n8n_workflow_recommender' / 'config' / 'config.yaml'

                if main_config_path.exists():
                    main_config = load_yaml(str(main_config_path))
                    api_key = main_config.get('api', {}).get('openai_key')

            if not api_key:
                raise ValueError("OpenAI API key not found in environment or config")

            config['openai_key'] = api_key

        # Resolve optional OpenAI project and organization
        if "openai_project" in config and "${OPENAI_PROJECT}" in str(config['openai_project']):
            config['openai_project'] = os.getenv('OPENAI_PROJECT')
        if "openai_organization" in config and "${OPENAI_ORGANIZATION}" in str(config['openai_organization']):
            config['openai_organization'] = os.getenv('OPENAI_ORGANIZATION')

        return config

    def _compute_summary_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute aggregate statistics

        Args:
            results: List of result dictionaries

        Returns:
            Summary statistics dictionary
        """
        zero_block = {
            "mean_f1": 0.0,
            "median_f1": 0.0,
            "std_f1": 0.0,
            "min_f1": 0.0,
            "max_f1": 0.0,
        }
        zero_param = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

        if not results:
            return {
                "total_templates": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "node_accuracy": dict(zero_block),
                "connection_accuracy": dict(zero_block),
                "parameter_accuracy": dict(zero_param),
                "json_validity": {"mean": 0.0, "count": 0},
                "error": "No results",
            }

        full_eval = [
            r for r in results
            if r.get("metrics") and "node_type_f1" in r["metrics"]
        ]
        json_scores = [
            float(r["metrics"]["llm_output_valid_json"])
            for r in results
            if r.get("metrics") and "llm_output_valid_json" in r["metrics"]
        ]
        json_block = {
            "mean": float(np.mean(json_scores)) if json_scores else 0.0,
            "count": len(json_scores),
        }

        if not full_eval:
            out = {
                "total_templates": len(results),
                "successful_evaluations": 0,
                "failed_evaluations": len(results),
                "node_accuracy": dict(zero_block),
                "connection_accuracy": dict(zero_block),
                "parameter_accuracy": dict(zero_param),
                "json_validity": json_block,
            }
            if not json_scores:
                out["error"] = "No valid results to aggregate"
            else:
                out["error"] = (
                    "No full workflow evaluations for node/connection metrics "
                    "(json_validity still computed)"
                )
            return out

        node_f1s = [r["metrics"]["node_type_f1"] for r in full_eval]
        conn_f1s = [r["metrics"]["connection_f1"] for r in full_eval]
        param_accs = [r["metrics"]["avg_parameter_accuracy"] for r in full_eval]

        return {
            "total_templates": len(results),
            "successful_evaluations": len(full_eval),
            "failed_evaluations": len(results) - len(full_eval),
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
            "json_validity": json_block,
        }

    def _compute_cost_report(self, results: List[Dict]) -> Dict:
        """
        Compute cost report

        Args:
            results: List of result dictionaries

        Returns:
            Cost report dictionary
        """
        results_with_usage = [
            r for r in results
            if r.get('metrics')
            and r['metrics'].get('usage') is not None
            and 'node_type_f1' in r['metrics']
        ]

        return self.cost_tracker.aggregate_costs(results_with_usage)

    def _print_summary(self, summary_stats: Dict, cost_report: Dict):
        """
        Print summary statistics

        Args:
            summary_stats: Summary statistics
            cost_report: Cost report
        """
        print("\n" + "="*60)
        print("Evaluation Summary")
        print("="*60)

        print(f"\nTotal Templates: {summary_stats['total_templates']}")
        print(f"Successful: {summary_stats['successful_evaluations']}")
        print(f"Failed: {summary_stats['failed_evaluations']}")

        if summary_stats.get('error'):
            print(f"\nWarning: {summary_stats['error']}")

        jv = summary_stats.get("json_validity") or {}
        if jv.get("count", 0):
            print(f"\nLLM output valid JSON (rate): {jv['mean']:.3f}  (n={jv['count']})")

        if summary_stats.get("successful_evaluations", 0) > 0:
            print(f"\nNode Type Accuracy (F1):")
            print(f"  Mean: {summary_stats['node_accuracy']['mean_f1']:.3f}")
            print(f"  Median: {summary_stats['node_accuracy']['median_f1']:.3f}")
            print(f"  Std: {summary_stats['node_accuracy']['std_f1']:.3f}")

            print(f"\nConnection Accuracy (F1):")
            print(f"  Mean: {summary_stats['connection_accuracy']['mean_f1']:.3f}")
            print(f"  Median: {summary_stats['connection_accuracy']['median_f1']:.3f}")

            print(f"\nParameter Accuracy:")
            print(f"  Mean: {summary_stats['parameter_accuracy']['mean']:.3f}")
            print(f"  Median: {summary_stats['parameter_accuracy']['median']:.3f}")

        print(f"\nCost Report:")
        print(f"  Total Tokens: {cost_report['total_tokens']:,}")
        print(f"  Total Cost: ${cost_report['total_cost']:.2f} USD")
        if cost_report['templates_with_cost'] > 0:
            print(f"  Avg Cost per Template: ${cost_report['avg_cost_per_template']:.4f} USD")
        else:
            print(f"  Avg Cost per Template: $0.0000 USD (no successful generations)")

        print("="*60 + "\n")
