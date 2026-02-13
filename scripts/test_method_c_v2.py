#!/usr/bin/env python3
"""
Test Method C with v2 Prompt (Enhanced with Gap Analysis)

Compare Method C v1 (original improved prompt) vs Method C v2 (gap-analysis enhanced prompt)
Uses the SAME 100 templates as the original four methods comparison.
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
import random
from pathlib import Path
from evaluation.utils.template_loader import TemplateLoader
from evaluation.generators.prompt_builder import PromptBuilder
from evaluation.generators.llm_workflow_generator import LLMWorkflowGenerator
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.parameter_evaluator import ParameterEvaluator
from evaluation.evaluators.cost_tracker import CostTracker


def load_test_templates():
    """Load the SAME 100 templates used in four methods comparison"""
    loader = TemplateLoader('n8n_templates/testing_data')
    all_templates = loader.load_all_templates()
    
    # Use SAME random seed as original comparison
    random.seed(42)
    test_templates = random.sample(all_templates, min(100, len(all_templates)))
    
    print(f"Loaded {len(test_templates)} test templates (seed=42)")
    return test_templates


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
    
    # Evaluate node types
    node_metrics = node_evaluator.evaluate_node_types(matching_result)
    
    # Evaluate connections
    conn_metrics = node_evaluator.evaluate_connections(gt_workflow, llm_workflow)
    
    # Evaluate parameters
    param_metrics = param_evaluator.evaluate_parameters(matching_result)
    
    # Combine metrics
    metrics = {
        **node_metrics,
        **conn_metrics,
        'avg_parameter_accuracy': param_metrics['avg_parameter_accuracy'],
        'usage': llm_result['usage']
    }
    
    # Calculate cost (handle both single-stage and two-stage usage structures)
    cost_tracker = CostTracker()
    usage = llm_result.get('usage')
    
    if isinstance(usage, dict) and 'workflow_generation' in usage:
        # Two-stage usage structure
        workflow_usage = usage['workflow_generation']
    elif isinstance(usage, dict) and 'total_tokens' in usage:
        # Single-stage usage structure
        workflow_usage = usage
    else:
        workflow_usage = usage if usage else {}
    
    if workflow_usage:
        cost_info = cost_tracker.calculate_cost(workflow_usage)
        metrics['total_cost'] = cost_info['total_cost']
    else:
        metrics['total_cost'] = 0.0
    
    return metrics


def main():
    print("\n" + "=" * 80)
    print("METHOD C v2 TESTING (Enhanced Prompt)")
    print("=" * 80)
    print("\nComparing:")
    print("  - Method C v1: Original improved prompt")
    print("  - Method C v2: Gap-analysis enhanced prompt (NEW)")
    print("\nUsing SAME 100 templates as original comparison (seed=42)")
    print("=" * 80 + "\n")
    
    # Load test templates
    test_templates = load_test_templates()
    
    # Setup output directories
    output_dir = Path('outputs/method_c_v2_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method_c_v1_dir = Path('outputs/four_methods_comparison/generated_workflows/method_c_improved_prompt')
    method_c_v2_dir = output_dir / 'generated_workflows' / 'method_c_v2_enhanced'
    method_c_v2_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Method C v2: Use NEW enhanced prompt
    prompt_v2_path = 'evaluation/config/workflow_generation_prompt_improved.txt'
    prompt_builder_v2 = PromptBuilder(prompt_v2_path, use_improved=True)
    
    workflow_generator_v2 = LLMWorkflowGenerator(
        openai_api_key=None,  # Will use env var
        prompt_builder=prompt_builder_v2,
        model='gpt-4o',
        temperature=0.3
    )
    
    # Evaluation components
    normalizer = WorkflowNormalizer()
    matcher = NodeMatcher()
    node_evaluator = NodeAccuracyEvaluator()
    param_evaluator = ParameterEvaluator()
    cost_tracker = CostTracker()
    
    # Results storage
    results_v1 = []
    results_v2 = []
    
    # Load existing Method C v1 results
    print("\nLoading existing Method C v1 results...")
    method_c_v1_results_path = Path('outputs/four_methods_comparison/method_c_results.json')
    
    existing_v1 = {}
    if method_c_v1_results_path.exists():
        with open(method_c_v1_results_path, 'r') as f:
            existing_v1_list = json.load(f)
            existing_v1 = {str(r['template_id']): r for r in existing_v1_list}
        print(f"  Loaded {len(existing_v1)} Method C v1 results")
    else:
        print("  ⚠️  No existing v1 results found - will need to regenerate")
    
    print()
    
    # Process each template
    for idx, template in enumerate(test_templates, 1):
        template_id = str(template.get('metadata', {}).get('id') or template.get('id', 'unknown'))
        template_name = template.get('workflow', {}).get('name', 'Unknown')
        
        print(f"\n[{idx}/100] Processing Template {template_id}")
        print(f"Name: {template_name[:80]}")
        print("-" * 80)
        
        # Extract original description
        original_description = template.get('workflow', {}).get('description', '').strip()
        
        # ===== Method C v1: Use existing results =====
        print("\n📊 Method C v1 (Original Improved Prompt):")
        
        if template_id in existing_v1 and existing_v1[template_id].get('metrics'):
            # Use existing result
            result_v1_data = existing_v1[template_id]
            
            print(f"   ✓ Using existing result")
            print(f"   ✓ Node F1: {result_v1_data['metrics']['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {result_v1_data['metrics']['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {result_v1_data['metrics']['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Cost: ${result_v1_data['metrics']['total_cost']:.4f}")
            
            result_v1_data['template_id'] = template_id
            result_v1_data['template_name'] = template_name
            results_v1.append(result_v1_data)
        else:
            print(f"   ⚠️  No existing result - will use v2 result only for this template")
            results_v1.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v1_improved_prompt',
                'error': 'No existing result',
                'metrics': None
            })
        
        # ===== Method C v2: Generate with NEW enhanced prompt =====
        print("\n🆕 Method C v2 (Gap-Analysis Enhanced Prompt):")
        print(f"   Generating workflow with v2 prompt...")
        
        result_v2 = workflow_generator_v2.generate_workflow(
            original_description,
            template_id,
            use_two_stage=False
        )
        
        if result_v2.get('error'):
            print(f"   ❌ Error: {result_v2['error']}")
            results_v2.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v2_enhanced_prompt',
                'original_description': original_description,
                'error': result_v2['error'],
                'metrics': None
            })
        else:
            metrics_v2 = evaluate_workflow(template, result_v2, normalizer, matcher, node_evaluator, param_evaluator)
            
            print(f"   ✓ Node F1: {metrics_v2['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {metrics_v2['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {metrics_v2['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Cost: ${metrics_v2['total_cost']:.4f}")
            
            results_v2.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v2_enhanced_prompt',
                'original_description': original_description,
                'error': None,
                'metrics': metrics_v2
            })
            
            # Save generated workflow
            result_v2['metrics'] = metrics_v2
            with open(method_c_v2_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_v2, f, indent=2)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    with open(output_dir / 'method_c_v1_results.json', 'w') as f:
        json.dump(results_v1, f, indent=2)
    print(f"✓ Saved v1 results: {output_dir / 'method_c_v1_results.json'}")
    
    with open(output_dir / 'method_c_v2_results.json', 'w') as f:
        json.dump(results_v2, f, indent=2)
    print(f"✓ Saved v2 results: {output_dir / 'method_c_v2_results.json'}")
    
    # Generate comparison report
    generate_comparison_report(results_v1, results_v2, output_dir)


def generate_comparison_report(results_v1, results_v2, output_dir):
    """Generate comparison report"""
    
    # Filter valid results
    valid_v1 = [r for r in results_v1 if r.get('metrics')]
    valid_v2 = [r for r in results_v2 if r.get('metrics')]
    
    def calculate_avg(results):
        if not results:
            return None
        return {
            'avg_node_f1': sum(r['metrics']['node_type_f1'] for r in results) / len(results),
            'avg_connection_f1': sum(r['metrics']['connection_f1'] for r in results) / len(results),
            'avg_parameter_accuracy': sum(r['metrics']['avg_parameter_accuracy'] for r in results) / len(results),
            'avg_cost': sum(r['metrics']['total_cost'] for r in results) / len(results),
            'valid_count': len(results)
        }
    
    avg_v1 = calculate_avg(valid_v1)
    avg_v2 = calculate_avg(valid_v2)
    
    # Calculate improvement
    improvement = None
    if avg_v1 and avg_v2:
        improvement = {
            'node_f1_delta': avg_v2['avg_node_f1'] - avg_v1['avg_node_f1'],
            'node_f1_pct': ((avg_v2['avg_node_f1'] - avg_v1['avg_node_f1']) / avg_v1['avg_node_f1'] * 100) if avg_v1['avg_node_f1'] > 0 else 0,
            'connection_f1_delta': avg_v2['avg_connection_f1'] - avg_v1['avg_connection_f1'],
            'connection_f1_pct': ((avg_v2['avg_connection_f1'] - avg_v1['avg_connection_f1']) / avg_v1['avg_connection_f1'] * 100) if avg_v1['avg_connection_f1'] > 0 else 0,
            'param_accuracy_delta': avg_v2['avg_parameter_accuracy'] - avg_v1['avg_parameter_accuracy'],
            'param_accuracy_pct': ((avg_v2['avg_parameter_accuracy'] - avg_v1['avg_parameter_accuracy']) / avg_v1['avg_parameter_accuracy'] * 100) if avg_v1['avg_parameter_accuracy'] > 0 else 0,
            'cost_delta': avg_v2['avg_cost'] - avg_v1['avg_cost']
        }
    
    # Save summary
    summary = {
        'method_c_v1': avg_v1,
        'method_c_v2': avg_v2,
        'improvement_v2_vs_v1': improvement
    }
    
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (Method C v1 vs v2)")
    print("=" * 100)
    
    if avg_v1:
        print("\n📊 Method C v1 (Original Improved Prompt):")
        print(f"   Node F1:            {avg_v1['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_v1['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_v1['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Cost:           ${avg_v1['avg_cost']:.4f}")
        print(f"   Valid Results:      {avg_v1['valid_count']}/100")
    
    if avg_v2:
        print("\n🆕 Method C v2 (Gap-Analysis Enhanced Prompt):")
        print(f"   Node F1:            {avg_v2['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_v2['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_v2['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Cost:           ${avg_v2['avg_cost']:.4f}")
        print(f"   Valid Results:      {avg_v2['valid_count']}/100")
    
    if improvement:
        print("\n📈 Improvement (v2 vs v1):")
        print(f"   Node F1:            {improvement['node_f1_delta']:+.3f} ({improvement['node_f1_pct']:+.1f}%)")
        print(f"   Connection F1:      {improvement['connection_f1_delta']:+.3f} ({improvement['connection_f1_pct']:+.1f}%)")
        print(f"   Parameter Accuracy: {improvement['param_accuracy_delta']:+.3f} ({improvement['param_accuracy_pct']:+.1f}%)")
        print(f"   Cost:               ${improvement['cost_delta']:+.4f}")
        
        # Success indicators
        print("\n🎯 Goal Achievement:")
        target_node_f1 = 0.50
        target_conn_f1 = 0.25
        target_param = 0.20
        
        if avg_v2['avg_node_f1'] >= target_node_f1:
            print(f"   ✅ Node F1 target reached: {avg_v2['avg_node_f1']:.3f} >= {target_node_f1}")
        else:
            print(f"   ⏳ Node F1 progress: {avg_v2['avg_node_f1']:.3f} / {target_node_f1} ({avg_v2['avg_node_f1']/target_node_f1*100:.1f}%)")
        
        if avg_v2['avg_connection_f1'] >= target_conn_f1:
            print(f"   ✅ Connection F1 target reached: {avg_v2['avg_connection_f1']:.3f} >= {target_conn_f1}")
        else:
            print(f"   ⏳ Connection F1 progress: {avg_v2['avg_connection_f1']:.3f} / {target_conn_f1} ({avg_v2['avg_connection_f1']/target_conn_f1*100:.1f}%)")
        
        if avg_v2['avg_parameter_accuracy'] >= target_param:
            print(f"   ✅ Parameter Accuracy target reached: {avg_v2['avg_parameter_accuracy']:.3f} >= {target_param}")
        else:
            print(f"   ⏳ Parameter Accuracy progress: {avg_v2['avg_parameter_accuracy']:.3f} / {target_param} ({avg_v2['avg_parameter_accuracy']/target_param*100:.1f}%)")
    
    print("\n" + "=" * 100)
    print(f"✓ Comparison summary saved: {output_dir / 'comparison_summary.json'}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
