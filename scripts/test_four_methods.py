#!/usr/bin/env python3
"""
Test Four Methods for Workflow Generation

Compare:
1. Method A: Direct description (use existing results)
2. Method B: AI-generated description from template structure (use existing results if available)
3. Method C: Improved prompt engineering + direct description (new)
4. Method D: Improved prompt engineering + two-stage generation (new)
"""

import sys
sys.path.insert(0, '.')

import json
import yaml
from pathlib import Path
from evaluation.utils.template_loader import TemplateLoader
from evaluation.generators.workflow_description_generator import WorkflowDescriptionGenerator
from evaluation.generators.description_optimizer import DescriptionOptimizer
from evaluation.generators.llm_workflow_generator import LLMWorkflowGenerator
from evaluation.generators.prompt_builder import PromptBuilder
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer
from evaluation.comparison.node_matcher import NodeMatcher
from evaluation.evaluators.node_accuracy_evaluator import NodeAccuracyEvaluator
from evaluation.evaluators.parameter_evaluator import ParameterEvaluator
from evaluation.evaluators.cost_tracker import CostTracker
import os


def main():
    print("=" * 80)
    print("Testing Four Methods for Workflow Generation")
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

    # Select 100 random templates for testing (same seed as before)
    import random
    random.seed(42)  # For reproducibility
    test_templates = random.sample(all_templates, 100)
    test_template_ids = [str(t.get('metadata', {}).get('id') or t.get('id', 'unknown')) for t in test_templates]

    print(f"Selected {len(test_templates)} templates for testing")
    print()

    # Initialize components
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    normalizer = WorkflowNormalizer()
    matcher = NodeMatcher()
    node_evaluator = NodeAccuracyEvaluator()
    embedding_model = config.get('embedding_model', 'paraphrase-multilingual-mpnet-base-v2')
    similarity_threshold = config.get('param_similarity_threshold', 0.8)
    param_evaluator = ParameterEvaluator(embedding_model, similarity_threshold)
    cost_tracker = CostTracker()

    # Initialize prompt builders
    base_prompt_path = config.get('prompt_template_path', 'evaluation/config/workflow_generation_prompt.txt')
    improved_prompt_path = Path('evaluation/config/workflow_generation_prompt_improved.txt')
    
    if not improved_prompt_path.exists():
        print(f"WARNING: Improved prompt not found at {improved_prompt_path}")
        print("Falling back to base prompt for Method C and D")
        improved_prompt_path = Path(base_prompt_path)

    base_prompt_builder = PromptBuilder(base_prompt_path, use_improved=False)
    improved_prompt_builder = PromptBuilder(str(improved_prompt_path), use_improved=True)

    # Initialize generators
    desc_generator = WorkflowDescriptionGenerator()
    description_optimizer = DescriptionOptimizer(api_key=api_key, model="gpt-4o-mini")

    # Method A: Base prompt + direct description (use existing)
    workflow_generator_a = LLMWorkflowGenerator(
        openai_api_key=api_key,
        prompt_builder=base_prompt_builder,
        model=config.get('model', 'gpt-4o'),
        temperature=config.get('temperature', 0.3)
    )

    # Method B: Base prompt + AI-generated description (use existing if available)
    workflow_generator_b = LLMWorkflowGenerator(
        openai_api_key=api_key,
        prompt_builder=base_prompt_builder,
        model=config.get('model', 'gpt-4o'),
        temperature=config.get('temperature', 0.3)
    )

    # Method C: Improved prompt + direct description (new)
    workflow_generator_c = LLMWorkflowGenerator(
        openai_api_key=api_key,
        prompt_builder=improved_prompt_builder,
        model=config.get('model', 'gpt-4o'),
        temperature=config.get('temperature', 0.3)
    )

    # Method D: Improved prompt + two-stage generation (new)
    workflow_generator_d = LLMWorkflowGenerator(
        openai_api_key=api_key,
        prompt_builder=improved_prompt_builder,
        model=config.get('model', 'gpt-4o'),
        temperature=config.get('temperature', 0.3),
        description_optimizer=description_optimizer
    )

    # Setup directories
    output_dir = Path('outputs/four_methods_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    method_a_dir = Path('outputs/two_methods_comparison/generated_workflows/method_a_direct_description')
    method_b_dir = Path('outputs/two_methods_comparison/generated_workflows/method_b_ai_description')
    method_c_dir = output_dir / 'generated_workflows' / 'method_c_improved_prompt'
    method_d_dir = output_dir / 'generated_workflows' / 'method_d_two_stage'
    
    method_c_dir.mkdir(parents=True, exist_ok=True)
    method_d_dir.mkdir(parents=True, exist_ok=True)

    # Results storage
    results_method_a = []
    results_method_b = []
    results_method_c = []
    results_method_d = []

    # Load existing Method A and B results if available
    print("Loading existing Method A and B results...")
    method_a_results_path = Path('outputs/two_methods_comparison/method_a_results.json')
    method_b_results_path = Path('outputs/two_methods_comparison/method_b_results.json')
    
    existing_a = {}
    existing_b = {}
    
    if method_a_results_path.exists():
        with open(method_a_results_path, 'r') as f:
            existing_a_list = json.load(f)
            # Convert template_id to string for consistent matching
            existing_a = {str(r['template_id']): r for r in existing_a_list}
        print(f"  Loaded {len(existing_a)} Method A results")
    
    if method_b_results_path.exists():
        with open(method_b_results_path, 'r') as f:
            existing_b_list = json.load(f)
            # Convert template_id to string for consistent matching
            existing_b = {str(r['template_id']): r for r in existing_b_list}
        print(f"  Loaded {len(existing_b)} Method B results")
    
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

        # ===== Method A: Base prompt + Direct description =====
        print("\n🅰️  Method A: Base prompt + Direct description")
        
        if template_id in existing_a and existing_a[template_id].get('metrics'):
            # Use existing result
            result_a_data = existing_a[template_id]
            method_a_file = method_a_dir / f'generated_{template_id}.json'
            
            if method_a_file.exists():
                with open(method_a_file, 'r') as f:
                    result_a = json.load(f)
                
                print(f"   ✓ Using existing result")
                print(f"   ✓ Node F1: {result_a_data['metrics']['node_type_f1']:.3f}")
                print(f"   ✓ Connection F1: {result_a_data['metrics']['connection_f1']:.3f}")
                print(f"   ✓ Parameter Accuracy: {result_a_data['metrics']['avg_parameter_accuracy']:.3f}")
                print(f"   ✓ Cost: ${result_a_data['metrics']['total_cost']:.4f}")
                
                # Ensure template_id is string for consistency
                result_a_data['template_id'] = template_id
                result_a_data['template_name'] = template_name
                results_method_a.append(result_a_data)
            else:
                print(f"   ⚠️  Workflow file not found, skipping")
                results_method_a.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'A_base_prompt_direct',
                    'error': 'Workflow file not found',
                    'metrics': None
                })
        else:
            # Re-generate if needed (should not happen if existing results are loaded correctly)
            if template_id not in existing_a:
                print(f"   ⚠️  Template {template_id} not found in existing results, generating new...")
            elif not existing_a[template_id].get('metrics'):
                print(f"   ⚠️  Template {template_id} has no metrics, generating new...")
            else:
                print(f"   Generating workflow...")
            result_a = workflow_generator_a.generate_workflow(original_description, template_id, use_two_stage=False)
            
            if result_a.get('error'):
                print(f"   ❌ Error: {result_a['error']}")
                results_method_a.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'A_base_prompt_direct',
                    'original_description': original_description,
                    'error': result_a['error'],
                    'metrics': None
                })
            else:
                metrics_a = evaluate_workflow(template, result_a, normalizer, matcher, node_evaluator, param_evaluator)
                print(f"   ✓ Node F1: {metrics_a['node_type_f1']:.3f}")
                print(f"   ✓ Connection F1: {metrics_a['connection_f1']:.3f}")
                print(f"   ✓ Parameter Accuracy: {metrics_a['avg_parameter_accuracy']:.3f}")
                
                results_method_a.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'A_base_prompt_direct',
                    'original_description': original_description,
                    'error': None,
                    'metrics': metrics_a
                })
                
                result_a['metrics'] = metrics_a
                with open(method_a_dir / f'generated_{template_id}.json', 'w') as f:
                    json.dump(result_a, f, indent=2)

        # ===== Method B: Base prompt + AI-generated description =====
        print("\n🅱️  Method B: Base prompt + AI-generated description")
        
        use_existing_b = False
        result_b = None
        
        if template_id in existing_b and existing_b[template_id].get('metrics'):
            # Use existing result
            result_b_data = existing_b[template_id]
            method_b_file = method_b_dir / f'generated_{template_id}.json'
            
            if method_b_file.exists():
                with open(method_b_file, 'r') as f:
                    result_b = json.load(f)
                
                print(f"   ✓ Using existing result")
                print(f"   ✓ Node F1: {result_b_data['metrics']['node_type_f1']:.3f}")
                print(f"   ✓ Connection F1: {result_b_data['metrics']['connection_f1']:.3f}")
                print(f"   ✓ Parameter Accuracy: {result_b_data['metrics']['avg_parameter_accuracy']:.3f}")
                print(f"   ✓ Total Cost: ${result_b_data['metrics']['total_cost']:.4f}")
                
                # Ensure template_id is string for consistency
                result_b_data['template_id'] = template_id
                result_b_data['template_name'] = template_name
                results_method_b.append(result_b_data)
                use_existing_b = True  # Mark that we're using existing result
            else:
                print(f"   ⚠️  Workflow file not found, generating new...")
                # Fall through to generation
        else:
            if template_id not in existing_b:
                print(f"   ⚠️  Template {template_id} not found in existing results, generating new...")
            elif not existing_b[template_id].get('metrics'):
                print(f"   ⚠️  Template {template_id} has no metrics, generating new...")
        
        if not use_existing_b:
            # Generate description from template structure
            desc_result = desc_generator.generate_description(template)
            
            if desc_result.get('error'):
                print(f"   ❌ Error generating description: {desc_result['error']}")
                results_method_b.append({
                    'template_id': template_id,
                    'template_name': template_name,
                    'method': 'B_base_prompt_ai_description',
                    'error': desc_result['error'],
                    'metrics': None
                })
            else:
                generated_description = desc_result['generated_description']
                desc_usage = desc_result['usage']
                
                # Generate workflow
                result_b = workflow_generator_b.generate_workflow(generated_description, template_id, use_two_stage=False)
                
                if result_b.get('error'):
                    print(f"   ❌ Error: {result_b['error']}")
                    results_method_b.append({
                        'template_id': template_id,
                        'template_name': template_name,
                        'method': 'B_base_prompt_ai_description',
                        'generated_description': generated_description,
                        'description_generation_usage': desc_usage,
                        'error': result_b['error'],
                        'metrics': None
                    })
                else:
                    metrics_b = evaluate_workflow(template, result_b, normalizer, matcher, node_evaluator, param_evaluator)
                    
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
                    print(f"   ✓ Total Cost: ${total_cost_b:.4f}")
                    
                    results_method_b.append({
                        'template_id': template_id,
                        'template_name': template_name,
                        'method': 'B_base_prompt_ai_description',
                        'original_description': original_description,
                        'generated_description': generated_description,
                        'description_generation_usage': desc_usage,
                        'error': None,
                        'metrics': metrics_b
                    })
                    
                    output_data = {
                        **result_b,
                        'generated_description': generated_description,
                        'description_generation_usage': desc_usage,
                        'metrics': metrics_b,
                        'original_description': original_description
                    }
                    with open(method_b_dir / f'generated_{template_id}.json', 'w') as f:
                        json.dump(output_data, f, indent=2)

        # ===== Method C: Improved prompt + Direct description =====
        print("\n🅲  Method C: Improved prompt + Direct description")
        
        result_c = workflow_generator_c.generate_workflow(original_description, template_id, use_two_stage=False)
        
        if result_c.get('error'):
            print(f"   ❌ Error: {result_c['error']}")
            results_method_c.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_improved_prompt_direct',
                'original_description': original_description,
                'error': result_c['error'],
                'metrics': None
            })
            with open(method_c_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_c, f, indent=2)
        else:
            metrics_c = evaluate_workflow(template, result_c, normalizer, matcher, node_evaluator, param_evaluator)
            
            print(f"   ✓ Node F1: {metrics_c['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {metrics_c['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {metrics_c['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Cost: ${metrics_c['total_cost']:.4f}")
            
            results_method_c.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_improved_prompt_direct',
                'original_description': original_description,
                'error': None,
                'metrics': metrics_c
            })
            
            result_c['metrics'] = metrics_c
            with open(method_c_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_c, f, indent=2)

        # ===== Method D: Improved prompt + Two-stage generation =====
        print("\n🅳  Method D: Improved prompt + Two-stage generation")
        
        result_d = workflow_generator_d.generate_workflow(original_description, template_id, use_two_stage=True)
        
        if result_d.get('error'):
            print(f"   ❌ Error: {result_d['error']}")
            results_method_d.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'D_improved_prompt_two_stage',
                'original_description': original_description,
                'error': result_d['error'],
                'metrics': None
            })
            with open(method_d_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_d, f, indent=2)
        else:
            metrics_d = evaluate_workflow(template, result_d, normalizer, matcher, node_evaluator, param_evaluator)
            
            # Handle two-stage cost calculation
            usage_d = result_d.get('usage', {})
            opt_result = result_d.get('description_optimization')
            
            if isinstance(usage_d, dict) and 'workflow_generation' in usage_d:
                # Two-stage usage structure
                opt_usage = opt_result.get('usage') if opt_result else None
                workflow_usage = usage_d['workflow_generation']
                
                workflow_cost = cost_tracker.calculate_cost(workflow_usage)
                
                if opt_usage:
                    opt_cost = cost_tracker.calculate_cost(opt_usage)
                    total_cost_d = opt_cost['total_cost'] + workflow_cost['total_cost']
                    
                    metrics_d['description_optimization_cost'] = opt_cost['total_cost']
                    metrics_d['workflow_generation_cost'] = workflow_cost['total_cost']
                    metrics_d['total_cost'] = total_cost_d
                else:
                    metrics_d['total_cost'] = workflow_cost['total_cost']
            else:
                # Single-stage usage structure (fallback)
                workflow_cost = cost_tracker.calculate_cost(usage_d if usage_d else {})
                metrics_d['total_cost'] = workflow_cost['total_cost']
            
            print(f"   ✓ Node F1: {metrics_d['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {metrics_d['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {metrics_d['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Total Cost: ${metrics_d['total_cost']:.4f}")
            
            results_method_d.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'D_improved_prompt_two_stage',
                'original_description': original_description,
                'optimized_description': result_d.get('optimized_description'),
                'description_optimization': result_d.get('description_optimization'),
                'error': None,
                'metrics': metrics_d
            })
            
            result_d['metrics'] = metrics_d
            with open(method_d_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_d, f, indent=2)

    # Save results
    with open(output_dir / 'method_a_results.json', 'w') as f:
        json.dump(results_method_a, f, indent=2)

    with open(output_dir / 'method_b_results.json', 'w') as f:
        json.dump(results_method_b, f, indent=2)

    with open(output_dir / 'method_c_results.json', 'w') as f:
        json.dump(results_method_c, f, indent=2)

    with open(output_dir / 'method_d_results.json', 'w') as f:
        json.dump(results_method_d, f, indent=2)

    # Generate comparison report
    generate_comparison_report(results_method_a, results_method_b, results_method_c, results_method_d, output_dir)

    print("\n" + "=" * 80)
    print("✓ Testing Complete!")
    print(f"\nResults saved to:")
    print(f"  - Summary & Comparison: {output_dir}")
    print(f"  - Method A Workflows: {method_a_dir}")
    print(f"  - Method B Workflows: {method_b_dir}")
    print(f"  - Method C Workflows: {method_c_dir}")
    print(f"  - Method D Workflows: {method_d_dir}")
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
        # Fallback
        workflow_usage = usage if usage else {}
    
    if workflow_usage:
        cost_info = cost_tracker.calculate_cost(workflow_usage)
        metrics['total_cost'] = cost_info['total_cost']
    else:
        metrics['total_cost'] = 0.0

    return metrics


def generate_comparison_report(results_a, results_b, results_c, results_d, output_dir):
    """
    Generate comparison report for all four methods
    """
    # Filter valid results
    valid_a = [r for r in results_a if r.get('metrics')]
    valid_b = [r for r in results_b if r.get('metrics')]
    valid_c = [r for r in results_c if r.get('metrics')]
    valid_d = [r for r in results_d if r.get('metrics')]

    # Calculate averages
    avg_metrics = {}

    def calculate_avg_metrics(results, method_name):
        if not results:
            return None
        
        metrics_dict = {
            'avg_node_f1': sum(r['metrics']['node_type_f1'] for r in results) / len(results),
            'avg_connection_f1': sum(r['metrics']['connection_f1'] for r in results) / len(results),
            'avg_parameter_accuracy': sum(r['metrics']['avg_parameter_accuracy'] for r in results) / len(results),
            'avg_cost': sum(r['metrics']['total_cost'] for r in results) / len(results),
            'valid_count': len(results)
        }
        
        # Add description/optimization costs if available
        if any('description_generation_cost' in r.get('metrics', {}) for r in results):
            metrics_dict['avg_description_generation_cost'] = sum(
                r['metrics'].get('description_generation_cost', 0) for r in results
            ) / len(results)
            metrics_dict['avg_workflow_generation_cost'] = sum(
                r['metrics'].get('workflow_generation_cost', 0) for r in results
            ) / len(results)
        
        if any('description_optimization_cost' in r.get('metrics', {}) for r in results):
            metrics_dict['avg_description_optimization_cost'] = sum(
                r['metrics'].get('description_optimization_cost', 0) for r in results
            ) / len(results)
            metrics_dict['avg_workflow_generation_cost'] = sum(
                r['metrics'].get('workflow_generation_cost', 0) for r in results
            ) / len(results)
        
        return metrics_dict

    if valid_a:
        avg_metrics['method_a'] = calculate_avg_metrics(valid_a, 'A')
    if valid_b:
        avg_metrics['method_b'] = calculate_avg_metrics(valid_b, 'B')
    if valid_c:
        avg_metrics['method_c'] = calculate_avg_metrics(valid_c, 'C')
    if valid_d:
        avg_metrics['method_d'] = calculate_avg_metrics(valid_d, 'D')

    # Calculate improvements relative to Method A
    if valid_a:
        baseline = avg_metrics['method_a']
        if valid_b:
            avg_metrics['improvement_b_vs_a'] = {
                'node_f1_delta': avg_metrics['method_b']['avg_node_f1'] - baseline['avg_node_f1'],
                'connection_f1_delta': avg_metrics['method_b']['avg_connection_f1'] - baseline['avg_connection_f1'],
                'parameter_accuracy_delta': avg_metrics['method_b']['avg_parameter_accuracy'] - baseline['avg_parameter_accuracy'],
                'cost_delta': avg_metrics['method_b']['avg_cost'] - baseline['avg_cost']
            }
        if valid_c:
            avg_metrics['improvement_c_vs_a'] = {
                'node_f1_delta': avg_metrics['method_c']['avg_node_f1'] - baseline['avg_node_f1'],
                'connection_f1_delta': avg_metrics['method_c']['avg_connection_f1'] - baseline['avg_connection_f1'],
                'parameter_accuracy_delta': avg_metrics['method_c']['avg_parameter_accuracy'] - baseline['avg_parameter_accuracy'],
                'cost_delta': avg_metrics['method_c']['avg_cost'] - baseline['avg_cost']
            }
        if valid_d:
            avg_metrics['improvement_d_vs_a'] = {
                'node_f1_delta': avg_metrics['method_d']['avg_node_f1'] - baseline['avg_node_f1'],
                'connection_f1_delta': avg_metrics['method_d']['avg_connection_f1'] - baseline['avg_connection_f1'],
                'parameter_accuracy_delta': avg_metrics['method_d']['avg_parameter_accuracy'] - baseline['avg_parameter_accuracy'],
                'cost_delta': avg_metrics['method_d']['avg_cost'] - baseline['avg_cost']
            }

    # Save summary
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)

    # Print summary in table format
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (All Four Methods)")
    print("=" * 100)

    # Prepare table data
    table_data = []
    headers = ["Method", "Node F1", "Connection F1", "Param Accuracy", "Total Cost", "Valid Results"]
    
    def format_method_row(method_label, metrics_dict):
        if not metrics_dict:
            return None
        
        cost_str = f"${metrics_dict['avg_cost']:.4f}"
        if 'avg_description_generation_cost' in metrics_dict:
            cost_str += f"\n(Desc: ${metrics_dict['avg_description_generation_cost']:.4f}, Workflow: ${metrics_dict['avg_workflow_generation_cost']:.4f})"
        elif 'avg_description_optimization_cost' in metrics_dict:
            cost_str += f"\n(Opt: ${metrics_dict['avg_description_optimization_cost']:.4f}, Workflow: ${metrics_dict['avg_workflow_generation_cost']:.4f})"
        
        return [
            method_label,
            f"{metrics_dict['avg_node_f1']:.3f}",
            f"{metrics_dict['avg_connection_f1']:.3f}",
            f"{metrics_dict['avg_parameter_accuracy']:.3f}",
            cost_str,
            f"{metrics_dict['valid_count']}/100"
        ]
    
    if avg_metrics.get('method_a'):
        table_data.append(format_method_row('🅰️  Method A\n(Base Prompt + Direct)', avg_metrics['method_a']))
    if avg_metrics.get('method_b'):
        table_data.append(format_method_row('🅱️  Method B\n(Base Prompt + AI Desc)', avg_metrics['method_b']))
    if avg_metrics.get('method_c'):
        table_data.append(format_method_row('🅲  Method C\n(Improved Prompt + Direct)', avg_metrics['method_c']))
    if avg_metrics.get('method_d'):
        table_data.append(format_method_row('🅳  Method D\n(Improved Prompt + Two-Stage)', avg_metrics['method_d']))
    
    # Print table
    if table_data:
        col_widths = [30, 12, 15, 15, 25, 15]
        
        # Print header
        header_row = "│ " + " │ ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " │"
        separator = "├" + "".join("─" * (w + 2) + "┼" for w in col_widths[:-1]) + "─" * (col_widths[-1] + 2) + "┤"
        top_border = "┌" + "".join("─" * (w + 2) + "┬" for w in col_widths[:-1]) + "─" * (col_widths[-1] + 2) + "┐"
        bottom_border = "└" + "".join("─" * (w + 2) + "┴" for w in col_widths[:-1]) + "─" * (col_widths[-1] + 2) + "┘"
        
        print("\n" + top_border)
        print(header_row)
        print(separator)
        
        for row in table_data:
            if row:
                # Handle multi-line cost strings
                cost_lines = row[4].split('\n')
                # Print first line with all columns
                first_line = []
                for j in range(len(row)):
                    if j == 4:
                        first_line.append(f"{cost_lines[0]:<{col_widths[j]}}")
                    else:
                        first_line.append(f"{str(row[j]):<{col_widths[j]}}")
                print("│ " + " │ ".join(first_line) + " │")
                
                # Print continuation lines for cost breakdown if needed
                for line in cost_lines[1:]:
                    empty_cols = " │ ".join(" " * col_widths[j] for j in range(4))
                    print(f"│ {empty_cols} │ {line:<{col_widths[4]}} │ {' ' * col_widths[5]} │")
        
        print(bottom_border)

    # Print improvements table
    if valid_a:
        baseline = avg_metrics['method_a']
        print("\n" + "=" * 100)
        print("📊 Improvements vs Method A (Baseline)")
        print("=" * 100)
        
        improvement_data = []
        improvement_headers = ["Comparison", "Node F1 Δ", "Connection F1 Δ", "Param Accuracy Δ", "Cost Δ"]
        
        def format_improvement_row(label, imp_dict):
            if not imp_dict:
                return None
            node_f1_pct = (imp_dict['node_f1_delta']/baseline['avg_node_f1']*100) if baseline['avg_node_f1'] > 0 else 0
            conn_f1_pct = (imp_dict['connection_f1_delta']/baseline['avg_connection_f1']*100) if baseline['avg_connection_f1'] > 0 else 0
            param_pct = (imp_dict['parameter_accuracy_delta']/baseline['avg_parameter_accuracy']*100) if baseline['avg_parameter_accuracy'] > 0 else 0
            
            return [
                label,
                f"{imp_dict['node_f1_delta']:+.3f} ({node_f1_pct:+.1f}%)",
                f"{imp_dict['connection_f1_delta']:+.3f} ({conn_f1_pct:+.1f}%)",
                f"{imp_dict['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%)",
                f"${imp_dict['cost_delta']:+.4f}"
            ]
        
        if avg_metrics.get('improvement_b_vs_a'):
            improvement_data.append(format_improvement_row("Method B vs A", avg_metrics['improvement_b_vs_a']))
        if avg_metrics.get('improvement_c_vs_a'):
            improvement_data.append(format_improvement_row("Method C vs A", avg_metrics['improvement_c_vs_a']))
        if avg_metrics.get('improvement_d_vs_a'):
            improvement_data.append(format_improvement_row("Method D vs A", avg_metrics['improvement_d_vs_a']))
        
        if improvement_data:
            imp_col_widths = [20, 25, 25, 25, 15]
            imp_top_border = "┌" + "".join("─" * (w + 2) + "┬" for w in imp_col_widths[:-1]) + "─" * (imp_col_widths[-1] + 2) + "┐"
            imp_separator = "├" + "".join("─" * (w + 2) + "┼" for w in imp_col_widths[:-1]) + "─" * (imp_col_widths[-1] + 2) + "┤"
            imp_bottom_border = "└" + "".join("─" * (w + 2) + "┴" for w in imp_col_widths[:-1]) + "─" * (imp_col_widths[-1] + 2) + "┘"
            
            print("\n" + imp_top_border)
            print("│ " + " │ ".join(f"{h:<{imp_col_widths[i]}}" for i, h in enumerate(improvement_headers)) + " │")
            print(imp_separator)
            
            for row in improvement_data:
                if row:
                    print("│ " + " │ ".join(f"{str(row[j]):<{imp_col_widths[j]}}" for j in range(len(row))) + " │")
            
            print(imp_bottom_border)


if __name__ == '__main__':
    main()
