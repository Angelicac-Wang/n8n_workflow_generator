#!/usr/bin/env python3
"""
Run Full Evaluation Pipeline with Gemini 2.5 Pro

Generate workflows using Gemini and evaluate against ground truth templates.
"""

import sys
sys.path.insert(0, '.')

import os
import json
import yaml
from pathlib import Path
from datetime import datetime

from evaluation.utils.template_loader import TemplateLoader
from evaluation.generators.gemini_workflow_generator import GeminiWorkflowGenerator
from evaluation.generators.prompt_builder import PromptBuilder
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.parameter_evaluator import ParameterEvaluator
from evaluation.evaluators.cost_tracker import CostTracker
from evaluation.visualization.report_generator import ReportGenerator


def main():
    print("=" * 80)
    print("Gemini 2.5 Pro Workflow Evaluation Pipeline")
    print("=" * 80)
    print()

    # Load configuration
    config_path = Path('evaluation/config/evaluation_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load templates
    templates_dir = config.get('templates_dir', 'n8n_templates/testing_data')
    template_loader = TemplateLoader(templates_dir)
    all_templates = template_loader.load_all_templates()

    print(f"Loaded {len(all_templates)} templates")
    print()

    # Initialize components
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        return

    prompt_template_path = config.get('prompt_template_path', 'evaluation/config/workflow_generation_prompt.txt')
    prompt_builder = PromptBuilder(prompt_template_path)

    # Use gemini-2.5-pro
    workflow_generator = GeminiWorkflowGenerator(
        google_api_key=google_api_key,
        prompt_builder=prompt_builder,
        model="gemini-2.5-pro",
        temperature=config.get('temperature', 0.3)
    )

    normalizer = WorkflowNormalizer()
    matcher = NodeMatcher()
    node_evaluator = NodeAccuracyEvaluator()

    embedding_model = config.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2')
    similarity_threshold = config.get('param_similarity_threshold', 0.8)
    param_evaluator = ParameterEvaluator(embedding_model, similarity_threshold)

    cost_tracker = CostTracker()

    # Output directories
    output_workflows_dir = Path('outputs/gemini_generated_workflows')
    output_workflows_dir.mkdir(parents=True, exist_ok=True)

    output_results_dir = Path('outputs/gemini_evaluation_results')
    output_results_dir.mkdir(parents=True, exist_ok=True)

    # Check for resume capability
    existing_files = list(output_workflows_dir.glob('generated_*.json'))
    existing_ids = set()
    for f in existing_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                existing_ids.add(str(data.get('template_id', '')))
        except:
            continue

    print(f"Found {len(existing_ids)} existing generated workflows")
    print("Will skip already generated templates (resume mode)")
    print()

    # Generate workflows
    print("=" * 80)
    print("Phase 1: Workflow Generation with Gemini 2.5 Pro")
    print("=" * 80)
    print()

    detailed_results = []

    for idx, template in enumerate(all_templates, 1):
        template_id = str(template.get('metadata', {}).get('id') or template.get('id', 'unknown'))
        template_name = template.get('name', 'Unknown')

        # Skip if already generated
        if template_id in existing_ids:
            # Load existing result
            existing_file = output_workflows_dir / f'generated_{template_id}.json'
            with open(existing_file, 'r') as f:
                existing_data = json.load(f)

            print(f"[{idx}/{len(all_templates)}] Skipping {template_id} (already generated)")
            continue

        print(f"[{idx}/{len(all_templates)}] Generating workflow for template {template_id}")
        print(f"  Name: {template_name[:60]}")

        # Extract description
        description = template.get('workflow', {}).get('description', '').strip()

        # Generate workflow
        result = workflow_generator.generate_workflow(description, template_id)

        # Save generated workflow
        output_file = output_workflows_dir / f'generated_{template_id}.json'
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        if result.get('error'):
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✓ Generated successfully")

        # Small delay to avoid rate limits
        if idx % 10 == 0:
            import time
            time.sleep(1)

    print()
    print("=" * 80)
    print("Phase 2: Evaluation")
    print("=" * 80)
    print()

    # Evaluate all generated workflows
    all_workflow_files = sorted(output_workflows_dir.glob('generated_*.json'))

    for idx, workflow_file in enumerate(all_workflow_files, 1):
        # Load generated workflow
        with open(workflow_file, 'r') as f:
            generated_data = json.load(f)

        template_id = str(generated_data.get('template_id', 'unknown'))

        print(f"[{idx}/{len(all_workflow_files)}] Evaluating template {template_id}")

        # Find corresponding ground truth template
        gt_template = None
        for t in all_templates:
            tid = str(t.get('metadata', {}).get('id') or t.get('id', 'unknown'))
            if tid == template_id:
                gt_template = t
                break

        if gt_template is None:
            print(f"  ⚠️  Ground truth template not found")
            detailed_results.append({
                'template_id': template_id,
                'template_name': 'Unknown',
                'error': 'Ground truth not found',
                'metrics': None
            })
            continue

        template_name = gt_template.get('name', 'Unknown')

        # Check for generation error
        if generated_data.get('error'):
            print(f"  ⚠️  Generation error: {generated_data['error']}")
            detailed_results.append({
                'template_id': template_id,
                'template_name': template_name,
                'error': generated_data['error'],
                'metrics': None
            })
            continue

        # Evaluate
        try:
            # Normalize workflows
            gt_workflow = normalizer.normalize_ground_truth(gt_template)
            llm_workflow = normalizer.normalize_llm_output(generated_data['llm_response'])

            # Match nodes
            matching_result = matcher.match_nodes(gt_workflow['nodes'], llm_workflow['nodes'])

            # Evaluate node types
            node_metrics = node_evaluator.evaluate_node_types(matching_result)

            # Evaluate connections
            conn_metrics = node_evaluator.evaluate_connections(gt_workflow, llm_workflow)

            # Evaluate parameters
            param_metrics = param_evaluator.evaluate_parameters(matching_result)

            # Calculate cost
            if generated_data.get('usage'):
                cost_info = cost_tracker.calculate_cost(generated_data['usage'])
                total_cost = cost_info['total_cost']
            else:
                total_cost = 0.0

            # Combine metrics
            metrics = {
                **node_metrics,
                **conn_metrics,
                'avg_parameter_accuracy': param_metrics['avg_parameter_accuracy'],
                'usage': generated_data.get('usage'),
                'total_cost': total_cost
            }

            print(f"  ✓ Node F1: {metrics['node_type_f1']:.3f}, Conn F1: {metrics['connection_f1']:.3f}, Param: {metrics['avg_parameter_accuracy']:.3f}")

            detailed_results.append({
                'template_id': template_id,
                'template_name': template_name,
                'error': None,
                'metrics': metrics
            })

        except Exception as e:
            print(f"  ❌ Evaluation error: {str(e)}")
            detailed_results.append({
                'template_id': template_id,
                'template_name': template_name,
                'error': str(e),
                'metrics': None
            })

    # Save detailed results
    print()
    print("=" * 80)
    print("Phase 3: Saving Results")
    print("=" * 80)
    print()

    with open(output_results_dir / 'detailed_per_template.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Calculate summary statistics
    valid_results = [r for r in detailed_results if r.get('metrics')]

    if valid_results:
        import numpy as np

        node_f1s = [r['metrics']['node_type_f1'] for r in valid_results]
        conn_f1s = [r['metrics']['connection_f1'] for r in valid_results]
        param_accs = [r['metrics']['avg_parameter_accuracy'] for r in valid_results]

        summary_stats = {
            'total_templates': len(detailed_results),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(detailed_results) - len(valid_results),
            'node_accuracy': {
                'mean_f1': float(np.mean(node_f1s)),
                'median_f1': float(np.median(node_f1s)),
                'std_f1': float(np.std(node_f1s)),
                'min_f1': float(np.min(node_f1s)),
                'max_f1': float(np.max(node_f1s))
            },
            'connection_accuracy': {
                'mean_f1': float(np.mean(conn_f1s)),
                'median_f1': float(np.median(conn_f1s)),
                'std_f1': float(np.std(conn_f1s)),
                'min_f1': float(np.min(conn_f1s)),
                'max_f1': float(np.max(conn_f1s))
            },
            'parameter_accuracy': {
                'mean': float(np.mean(param_accs)),
                'median': float(np.median(param_accs)),
                'std': float(np.std(param_accs)),
                'min': float(np.min(param_accs)),
                'max': float(np.max(param_accs))
            }
        }

        # Add cost tracking
        total_cost = sum(r['metrics']['total_cost'] for r in valid_results)
        total_input = sum(r['metrics']['usage']['prompt_tokens'] for r in valid_results if r['metrics'].get('usage'))
        total_output = sum(r['metrics']['usage']['completion_tokens'] for r in valid_results if r['metrics'].get('usage'))

        summary_stats['cost_tracking'] = {
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_tokens': total_input + total_output,
            'total_cost': total_cost,
            'avg_cost_per_template': total_cost / len(valid_results) if valid_results else 0.0,
            'templates_with_cost': len([r for r in valid_results if r['metrics'].get('usage')]),
            'currency': 'USD'
        }

        with open(output_results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)

        print(f"✓ Saved summary statistics")
        print()
        print("Summary Statistics:")
        print(f"  Node F1:            Mean={summary_stats['node_accuracy']['mean_f1']:.3f}, Median={summary_stats['node_accuracy']['median_f1']:.3f}")
        print(f"  Connection F1:      Mean={summary_stats['connection_accuracy']['mean_f1']:.3f}, Median={summary_stats['connection_accuracy']['median_f1']:.3f}")
        print(f"  Parameter Accuracy: Mean={summary_stats['parameter_accuracy']['mean']:.3f}, Median={summary_stats['parameter_accuracy']['median']:.3f}")
        print(f"  Total Cost:         ${summary_stats['cost_tracking']['total_cost']:.2f}")
        print()

    # Generate visualizations
    print("Generating visualizations...")
    viz_config = {
        'visualization': {
            'style': 'seaborn-v0_8-darkgrid',
            'figure_size': [12, 8],
            'dpi': 300
        }
    }
    report_gen = ReportGenerator(viz_config)
    viz_dir = output_results_dir / 'visualizations'
    report_gen.generate_all_visualizations(detailed_results, viz_dir)

    print()
    print("=" * 80)
    print("✓ Evaluation Complete!")
    print("=" * 80)
    print(f"Generated workflows: {output_workflows_dir}")
    print(f"Evaluation results:  {output_results_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
