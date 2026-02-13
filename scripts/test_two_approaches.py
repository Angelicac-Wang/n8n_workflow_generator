#!/usr/bin/env python3
"""
Test Two Approaches for Workflow Generation

Compare:
1. Method A: Direct description (use existing results)
2. Method B: AI-generated description from template structure (new generation)
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
from pathlib import Path
from evaluation.utils.template_loader import TemplateLoader
from evaluation.generators.workflow_description_generator import WorkflowDescriptionGenerator
from evaluation.generators.llm_workflow_generator import LLMWorkflowGenerator
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.parameter_evaluator import ParameterEvaluator
from evaluation.evaluators.cost_tracker import CostTracker


def main():
    print("=" * 80)
    print("Testing Two Approaches for Workflow Generation")
    print("=" * 80)
    print()

    # Configuration
    config_path = Path('evaluation/config/evaluation_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load templates
    templates_dir = config.get('templates_dir', 'n8n_templates/testing_data')
    template_loader = TemplateLoader(templates_dir)
    all_templates = template_loader.load_all_templates()

    # Select 100 random templates for testing
    import random
    random.seed(42)  # For reproducibility
    test_templates = random.sample(all_templates, 100)
    # Convert to strings to match the format in detailed_per_template.json
    test_template_ids = [str(t.get('metadata', {}).get('id') or t.get('id', 'unknown')) for t in test_templates]

    print(f"Selected {len(test_templates)} templates for testing")
    print()

    # Load existing Method A results (from previous full evaluation)
    print("Loading existing Method A results from previous evaluation...")
    detailed_results_path = Path('outputs/evaluation_results/detailed_per_template.json')

    with open(detailed_results_path, 'r') as f:
        all_detailed_results = json.load(f)

    # Filter for our 10 test templates
    method_a_existing = {
        r['template_id']: r
        for r in all_detailed_results
        if r['template_id'] in test_template_ids
    }

    print(f"Found {len(method_a_existing)} existing results for Method A")

    # Also copy the generated workflows for Method A
    existing_workflows_dir = Path('outputs/llm_generated_workflows')
    method_a_dir = Path('outputs/two_methods_comparison/generated_workflows/method_a_direct_description')
    method_a_dir.mkdir(parents=True, exist_ok=True)

    for template_id in test_template_ids:
        src_file = existing_workflows_dir / f'generated_{template_id}.json'
        dst_file = method_a_dir / f'generated_{template_id}.json'
        if src_file.exists():
            with open(src_file, 'r') as f:
                data = json.load(f)
            # Add metrics from detailed results
            if template_id in method_a_existing:
                data['metrics'] = method_a_existing[template_id].get('metrics', {})
            with open(dst_file, 'w') as f:
                json.dump(data, f, indent=2)

    print(f"Copied {len(test_template_ids)} Method A workflows to comparison directory")
    print()

    # Initialize components for Method B
    from evaluation.generators.prompt_builder import PromptBuilder
    import os

    desc_generator = WorkflowDescriptionGenerator()
    prompt_template_path = config.get('prompt_template_path', 'evaluation/config/workflow_generation_prompt.txt')
    prompt_builder = PromptBuilder(prompt_template_path)
    workflow_generator = LLMWorkflowGenerator(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        prompt_builder=prompt_builder,
        model=config.get('model', 'gpt-4o'),
        temperature=config.get('temperature', 0.3)
    )
    normalizer = WorkflowNormalizer()
    matcher = NodeMatcher()
    node_evaluator = NodeAccuracyEvaluator()
    embedding_model = config.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2')
    similarity_threshold = config.get('param_similarity_threshold', 0.8)
    param_evaluator = ParameterEvaluator(embedding_model, similarity_threshold)
    cost_tracker = CostTracker()

    # Results storage
    results_method_a = []  # Direct description (from existing)
    results_method_b = []  # AI-generated description (new)

    # Create output directory for Method B workflows
    method_b_dir = Path('outputs/two_methods_comparison/generated_workflows/method_b_ai_description')
    method_b_dir.mkdir(parents=True, exist_ok=True)

    # Process each template
    for idx, template in enumerate(test_templates, 1):
        template_id = template.get('metadata', {}).get('id') or template.get('id', 'unknown')
        template_name = template.get('name', 'Unknown')

        print(f"\n[{idx}/100] Processing Template {template_id}")
        print(f"Name: {template_name[:80]}")
        print("-" * 80)

        # Extract original description
        original_description = template.get('workflow', {}).get('description', '').strip()

        # ===== Method A: Re-calculate from method_a directory =====
        print("\n🅰️  Method A: Re-calculating from direct description workflow")

        # Load the generated workflow from method_a directory
        method_a_file = method_a_dir / f'generated_{template_id}.json'

        if method_a_file.exists():
            with open(method_a_file, 'r') as f:
                result_a = json.load(f)

            if result_a.get('error'):
                print(f"   ❌ Error: {result_a['error']}")
                results_method_a.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'A_direct_description',
                    'original_description': original_description,
                    'error': result_a['error'],
                    'metrics': None
                })
            else:
                # Re-evaluate using the same evaluation function
                metrics_a = evaluate_workflow(
                    template, result_a, normalizer, matcher,
                    node_evaluator, param_evaluator
                )

                print(f"   ✓ Node F1: {metrics_a['node_type_f1']:.3f}")
                print(f"   ✓ Connection F1: {metrics_a['connection_f1']:.3f}")
                print(f"   ✓ Parameter Accuracy: {metrics_a['avg_parameter_accuracy']:.3f}")
                print(f"   ✓ Cost: ${metrics_a['total_cost']:.4f}")

                results_method_a.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'A_direct_description',
                    'original_description': original_description,
                    'error': None,
                    'metrics': metrics_a
                })

                # Update the file with new metrics
                result_a['metrics'] = metrics_a
                with open(method_a_file, 'w') as f:
                    json.dump(result_a, f, indent=2)
        else:
            print(f"   ⚠️  Workflow file not found in method_a directory")
            results_method_a.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'A_direct_description',
                'original_description': original_description,
                'error': 'Workflow file not found',
                'metrics': None
            })

        # ===== Method B: AI-Generated Description =====
        print("\n🅱️  Method B: Using AI-generated description")

        # Generate description from template structure
        desc_result = desc_generator.generate_description(template)

        if desc_result.get('error'):
            print(f"   ❌ Error generating description: {desc_result['error']}")
            results_method_b.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'B_ai_generated_description',
                'error': desc_result['error'],
                'metrics': None
            })
            # Save error result
            with open(method_b_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(desc_result, f, indent=2)
            continue

        generated_description = desc_result['generated_description']
        desc_usage = desc_result['usage']

        print(f"Generated description: {generated_description[:100]}...")

        # Generate workflow using AI-generated description
        result_b = workflow_generator.generate_workflow(generated_description, template_id)

        if result_b.get('error'):
            print(f"   ❌ Error: {result_b['error']}")
            results_method_b.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'B_ai_generated_description',
                'generated_description': generated_description,
                'description_generation_usage': desc_usage,
                'error': result_b['error'],
                'metrics': None
            })
            # Save error result
            output_data = {
                **result_b,
                'generated_description': generated_description,
                'description_generation_usage': desc_usage
            }
            with open(method_b_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            # Evaluate
            metrics_b = evaluate_workflow(
                template, result_b, normalizer, matcher,
                node_evaluator, param_evaluator
            )

            # Add description generation cost
            desc_cost = cost_tracker.calculate_cost(desc_usage)
            workflow_cost = cost_tracker.calculate_cost(result_b['usage'])
            total_cost_b = desc_cost['total_cost'] + workflow_cost['total_cost']

            metrics_b['description_generation_cost'] = desc_cost['total_cost']
            metrics_b['workflow_generation_cost'] = workflow_cost['total_cost']
            metrics_b['total_cost'] = total_cost_b

            print(f"   ✓ Node F1: {metrics_b['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {metrics_b['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {metrics_b['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Total Cost: ${total_cost_b:.4f} (desc: ${desc_cost['total_cost']:.4f} + workflow: ${workflow_cost['total_cost']:.4f})")

            results_method_b.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'B_ai_generated_description',
                'original_description': original_description,
                'generated_description': generated_description,
                'description_generation_usage': desc_usage,
                'error': None,
                'metrics': metrics_b
            })

            # Save successful result with metrics
            output_data = {
                **result_b,
                'generated_description': generated_description,
                'description_generation_usage': desc_usage,
                'metrics': metrics_b,
                'original_description': original_description
            }
            with open(method_b_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(output_data, f, indent=2)

    # Save results
    output_dir = Path('outputs/two_methods_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'method_a_results.json', 'w') as f:
        json.dump(results_method_a, f, indent=2)

    with open(output_dir / 'method_b_results.json', 'w') as f:
        json.dump(results_method_b, f, indent=2)

    # Generate comparison report
    generate_comparison_report(results_method_a, results_method_b, output_dir)

    print("\n" + "=" * 80)
    print("✓ Testing Complete!")
    print(f"\nResults saved to:")
    print(f"  - Summary & Comparison: {output_dir}")
    print(f"  - Method A Workflows: {method_a_dir}")
    print(f"  - Method B Workflows: {method_b_dir}")
    print("=" * 80)


def evaluate_workflow(template, llm_result, normalizer, matcher, node_evaluator, param_evaluator):
    """
    Evaluate a generated workflow against ground truth

    Returns:
        Dictionary with evaluation metrics
    """
    # Normalize workflows
    gt_workflow = normalizer.normalize_ground_truth(template)
    llm_workflow = normalizer.normalize_llm_output(llm_result['llm_response'])

    # Match nodes
    matching_result = matcher.match_nodes(gt_workflow['nodes'], llm_workflow['nodes'])

    # Evaluate node types (pass matching_result only)
    node_metrics = node_evaluator.evaluate_node_types(matching_result)

    # Evaluate connections
    conn_metrics = node_evaluator.evaluate_connections(gt_workflow, llm_workflow)

    # Evaluate parameters (pass matching_result)
    param_metrics = param_evaluator.evaluate_parameters(matching_result)

    # Combine metrics
    metrics = {
        **node_metrics,
        **conn_metrics,
        'avg_parameter_accuracy': param_metrics['avg_parameter_accuracy'],
        'usage': llm_result['usage']
    }

    # Calculate cost
    cost_tracker = CostTracker()
    cost_info = cost_tracker.calculate_cost(llm_result['usage'])
    metrics['total_cost'] = cost_info['total_cost']

    return metrics


def generate_comparison_report(results_a, results_b, output_dir):
    """
    Generate comparison report and save as JSON and CSV
    """
    # Filter valid results
    valid_a = [r for r in results_a if r.get('metrics')]
    valid_b = [r for r in results_b if r.get('metrics')]

    # Calculate averages
    avg_metrics = {}

    if valid_a:
        avg_metrics['method_a'] = {
            'avg_node_f1': sum(r['metrics']['node_type_f1'] for r in valid_a) / len(valid_a),
            'avg_connection_f1': sum(r['metrics']['connection_f1'] for r in valid_a) / len(valid_a),
            'avg_parameter_accuracy': sum(r['metrics']['avg_parameter_accuracy'] for r in valid_a) / len(valid_a),
            'avg_cost': sum(r['metrics']['total_cost'] for r in valid_a) / len(valid_a),
            'valid_count': len(valid_a)
        }

    if valid_b:
        avg_metrics['method_b'] = {
            'avg_node_f1': sum(r['metrics']['node_type_f1'] for r in valid_b) / len(valid_b),
            'avg_connection_f1': sum(r['metrics']['connection_f1'] for r in valid_b) / len(valid_b),
            'avg_parameter_accuracy': sum(r['metrics']['avg_parameter_accuracy'] for r in valid_b) / len(valid_b),
            'avg_cost': sum(r['metrics']['total_cost'] for r in valid_b) / len(valid_b),
            'avg_description_generation_cost': sum(r['metrics'].get('description_generation_cost', 0) for r in valid_b) / len(valid_b),
            'avg_workflow_generation_cost': sum(r['metrics'].get('workflow_generation_cost', 0) for r in valid_b) / len(valid_b),
            'valid_count': len(valid_b)
        }

    # Calculate improvement
    if valid_a and valid_b:
        avg_metrics['improvement'] = {
            'node_f1_delta': avg_metrics['method_b']['avg_node_f1'] - avg_metrics['method_a']['avg_node_f1'],
            'connection_f1_delta': avg_metrics['method_b']['avg_connection_f1'] - avg_metrics['method_a']['avg_connection_f1'],
            'parameter_accuracy_delta': avg_metrics['method_b']['avg_parameter_accuracy'] - avg_metrics['method_a']['avg_parameter_accuracy'],
            'cost_delta': avg_metrics['method_b']['avg_cost'] - avg_metrics['method_a']['avg_cost']
        }

    # Save summary
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if valid_a:
        print("\n🅰️  Method A (Direct Description):")
        print(f"   Node F1:            {avg_metrics['method_a']['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_metrics['method_a']['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_metrics['method_a']['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Cost:           ${avg_metrics['method_a']['avg_cost']:.4f}")
        print(f"   Valid Results:      {avg_metrics['method_a']['valid_count']}/100")

    if valid_b:
        print("\n🅱️  Method B (AI-Generated Description):")
        print(f"   Node F1:            {avg_metrics['method_b']['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_metrics['method_b']['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_metrics['method_b']['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Total Cost:     ${avg_metrics['method_b']['avg_cost']:.4f}")
        print(f"     - Description:    ${avg_metrics['method_b']['avg_description_generation_cost']:.4f}")
        print(f"     - Workflow:       ${avg_metrics['method_b']['avg_workflow_generation_cost']:.4f}")
        print(f"   Valid Results:      {avg_metrics['method_b']['valid_count']}/100")

    if valid_a and valid_b:
        print("\n📊 Improvement (B - A):")
        imp = avg_metrics['improvement']
        node_f1_pct = (imp['node_f1_delta']/avg_metrics['method_a']['avg_node_f1']*100) if avg_metrics['method_a']['avg_node_f1'] > 0 else 0
        conn_f1_pct = (imp['connection_f1_delta']/avg_metrics['method_a']['avg_connection_f1']*100) if avg_metrics['method_a']['avg_connection_f1'] > 0 else 0
        param_pct = (imp['parameter_accuracy_delta']/avg_metrics['method_a']['avg_parameter_accuracy']*100) if avg_metrics['method_a']['avg_parameter_accuracy'] > 0 else 0

        print(f"   Node F1:            {imp['node_f1_delta']:+.3f} ({node_f1_pct:+.1f}%)")
        print(f"   Connection F1:      {imp['connection_f1_delta']:+.3f} ({conn_f1_pct:+.1f}%)")
        print(f"   Parameter Accuracy: {imp['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%)")
        print(f"   Cost:               ${imp['cost_delta']:+.4f}")


if __name__ == '__main__':
    main()
