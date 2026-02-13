#!/usr/bin/env python3
"""
Test Method C with v4 Prompt (Refined LangChain Detection + httpRequest Fix)

Compare Method C v3 vs Method C v4
Uses the SAME 100 templates as the original four methods comparison.

v4 Focus: 
- Tighten AI agent detection (only conversational + memory workflows)
- Fix httpRequest overuse
- Strengthen intermediate nodes (Set, Code)
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
    print("METHOD C v4 TESTING (Refined Detection + httpRequest Fix)")
    print("=" * 80)
    print("\nComparing:")
    print("  - Method C v3: LangChain-focused improvements")
    print("  - Method C v4: Refined detection + httpRequest fix (NEW)")
    print("\nv4 Focus:")
    print("  - Tighten AI detection (only conversational + memory)")
    print("  - Fix httpRequest overuse (94 unnecessary → target: <20)")
    print("  - Strengthen Set/Code node usage")
    print("\nUsing SAME 100 templates as original comparison (seed=42)")
    print("=" * 80 + "\n")
    
    # Load test templates
    test_templates = load_test_templates()
    
    # Setup output directories
    output_dir = Path('outputs/method_c_v4_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method_c_v3_dir = Path('outputs/method_c_v3_comparison/generated_workflows/method_c_v3_langchain')
    method_c_v4_dir = output_dir / 'generated_workflows' / 'method_c_v4_refined'
    method_c_v4_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    print("Initializing components...")
    
    # Method C v4: Use NEW refined prompt
    prompt_v4_path = 'evaluation/config/workflow_generation_prompt_improved.txt'
    prompt_builder_v4 = PromptBuilder(prompt_v4_path, use_improved=True)
    
    workflow_generator_v4 = LLMWorkflowGenerator(
        openai_api_key=None,  # Will use env var
        prompt_builder=prompt_builder_v4,
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
    results_v3 = []
    results_v4 = []
    
    # Load existing Method C v3 results
    print("\nLoading existing Method C v3 results...")
    method_c_v3_results_path = Path('outputs/method_c_v3_comparison/method_c_v3_results.json')
    
    existing_v3 = {}
    if method_c_v3_results_path.exists():
        with open(method_c_v3_results_path, 'r') as f:
            existing_v3_list = json.load(f)
            existing_v3 = {str(r['template_id']): r for r in existing_v3_list}
        print(f"  Loaded {len(existing_v3)} Method C v3 results")
    else:
        print("  ⚠️  No existing v3 results found - will need to regenerate")
    
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
        
        # Check if this is an AI agent workflow (for tracking) - VERY STRICT detection for v4
        # Must have BOTH conversational pattern AND memory/context indicators
        full_text = (original_description + ' ' + template_name).lower()
        
        # Conversational patterns
        conversational_patterns = [
            'chat with', 'chatbot that', 'chat interface',
            'conversational', 'conversation history',
            'q&a about', 'ask questions about'
        ]
        
        # Memory/context patterns
        memory_patterns = [
            'memory', 'context', 'remember', 'history',
            'rag', 'retrieval', 'vector store', 'embeddings'
        ]
        
        # Check if has BOTH conversational AND memory patterns
        has_conversational = any(pattern in full_text for pattern in conversational_patterns)
        has_memory = any(pattern in full_text for pattern in memory_patterns)
        
        is_ai_workflow = has_conversational and has_memory
        if is_ai_workflow:
            print(f"   🤖 AI Agent Workflow Detected")
        
        # ===== Method C v3: Use existing results =====
        print("\n📊 Method C v3 (LangChain-Enhanced):")
        
        if template_id in existing_v3 and existing_v3[template_id].get('metrics'):
            # Use existing result
            result_v3_data = existing_v3[template_id]
            
            print(f"   ✓ Using existing result")
            print(f"   ✓ Node F1: {result_v3_data['metrics']['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {result_v3_data['metrics']['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {result_v3_data['metrics']['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Cost: ${result_v3_data['metrics']['total_cost']:.4f}")
            
            result_v3_data['template_id'] = template_id
            result_v3_data['template_name'] = template_name
            results_v3.append(result_v3_data)
        else:
            print(f"   ⚠️  No existing result - will use v4 result only for this template")
            results_v3.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v3_langchain_enhanced',
                'error': 'No existing result',
                'metrics': None
            })
        
        # ===== Method C v4: Generate with NEW refined prompt =====
        print("\n🆕 Method C v4 (Refined Detection + httpRequest Fix):")
        print(f"   Generating workflow with v4 prompt...")
        
        result_v4 = workflow_generator_v4.generate_workflow(
            original_description,
            template_id,
            use_two_stage=False
        )
        
        if result_v4.get('error'):
            print(f"   ❌ Error: {result_v4['error']}")
            results_v4.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v4_refined',
                'original_description': original_description,
                'is_ai_workflow': is_ai_workflow,
                'error': result_v4['error'],
                'metrics': None
            })
        else:
            metrics_v4 = evaluate_workflow(template, result_v4, normalizer, matcher, node_evaluator, param_evaluator)
            
            print(f"   ✓ Node F1: {metrics_v4['node_type_f1']:.3f}")
            print(f"   ✓ Connection F1: {metrics_v4['connection_f1']:.3f}")
            print(f"   ✓ Parameter Accuracy: {metrics_v4['avg_parameter_accuracy']:.3f}")
            print(f"   ✓ Cost: ${metrics_v4['total_cost']:.4f}")
            
            # Show improvement for AI workflows
            if is_ai_workflow and template_id in existing_v3 and existing_v3[template_id].get('metrics'):
                v3_f1 = existing_v3[template_id]['metrics']['node_type_f1']
                improvement = metrics_v4['node_type_f1'] - v3_f1
                if improvement > 0:
                    if v3_f1 > 0:
                        print(f"   🎯 AI Workflow Improvement: +{improvement:.3f} ({improvement/v3_f1*100:.1f}%)")
                    else:
                        print(f"   🎯 AI Workflow Improvement: +{improvement:.3f} (from 0.000!)")
                elif improvement < 0:
                    print(f"   ⚠️  AI Workflow Regression: {improvement:.3f}")
            
            results_v4.append({
                'template_id': template_id,
                'template_name': template_name,
                'method': 'C_v4_refined',
                'original_description': original_description,
                'is_ai_workflow': is_ai_workflow,
                'error': None,
                'metrics': metrics_v4
            })
            
            # Save generated workflow
            result_v4['metrics'] = metrics_v4
            result_v4['is_ai_workflow'] = is_ai_workflow
            with open(method_c_v4_dir / f'generated_{template_id}.json', 'w') as f:
                json.dump(result_v4, f, indent=2)
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")
    print("=" * 80)
    
    with open(output_dir / 'method_c_v3_results.json', 'w') as f:
        json.dump(results_v3, f, indent=2)
    print(f"✓ Saved v3 results: {output_dir / 'method_c_v3_results.json'}")
    
    with open(output_dir / 'method_c_v4_results.json', 'w') as f:
        json.dump(results_v4, f, indent=2)
    print(f"✓ Saved v4 results: {output_dir / 'method_c_v4_results.json'}")
    
    # Generate comparison report
    generate_comparison_report(results_v3, results_v4, output_dir)


def generate_comparison_report(results_v3, results_v4, output_dir):
    """Generate comparison report"""
    
    # Filter valid results
    valid_v3 = [r for r in results_v3 if r.get('metrics')]
    valid_v4 = [r for r in results_v4 if r.get('metrics')]
    
    # Separate AI workflows from non-AI
    ai_v3 = [r for r in valid_v3 if r.get('is_ai_workflow')]
    ai_v4 = [r for r in valid_v4 if r.get('is_ai_workflow')]
    non_ai_v3 = [r for r in valid_v3 if not r.get('is_ai_workflow')]
    non_ai_v4 = [r for r in valid_v4 if not r.get('is_ai_workflow')]
    
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
    
    avg_v3 = calculate_avg(valid_v3)
    avg_v4 = calculate_avg(valid_v4)
    avg_ai_v3 = calculate_avg(ai_v3)
    avg_ai_v4 = calculate_avg(ai_v4)
    avg_non_ai_v3 = calculate_avg(non_ai_v3)
    avg_non_ai_v4 = calculate_avg(non_ai_v4)
    
    # Calculate improvement
    improvement = None
    if avg_v3 and avg_v4:
        improvement = {
            'node_f1_delta': avg_v4['avg_node_f1'] - avg_v3['avg_node_f1'],
            'node_f1_pct': ((avg_v4['avg_node_f1'] - avg_v3['avg_node_f1']) / avg_v3['avg_node_f1'] * 100) if avg_v3['avg_node_f1'] > 0 else 0,
            'connection_f1_delta': avg_v4['avg_connection_f1'] - avg_v3['avg_connection_f1'],
            'connection_f1_pct': ((avg_v4['avg_connection_f1'] - avg_v3['avg_connection_f1']) / avg_v3['avg_connection_f1'] * 100) if avg_v3['avg_connection_f1'] > 0 else 0,
            'param_accuracy_delta': avg_v4['avg_parameter_accuracy'] - avg_v3['avg_parameter_accuracy'],
            'param_accuracy_pct': ((avg_v4['avg_parameter_accuracy'] - avg_v3['avg_parameter_accuracy']) / avg_v3['avg_parameter_accuracy'] * 100) if avg_v3['avg_parameter_accuracy'] > 0 else 0,
            'cost_delta': avg_v4['avg_cost'] - avg_v3['avg_cost']
        }
    
    # AI workflow improvement
    ai_improvement = None
    if avg_ai_v3 and avg_ai_v4:
        ai_improvement = {
            'node_f1_delta': avg_ai_v4['avg_node_f1'] - avg_ai_v3['avg_node_f1'],
            'node_f1_pct': ((avg_ai_v4['avg_node_f1'] - avg_ai_v3['avg_node_f1']) / avg_ai_v3['avg_node_f1'] * 100) if avg_ai_v3['avg_node_f1'] > 0 else 0,
        }
    
    # Save summary
    summary = {
        'overall': {
            'method_c_v3': avg_v3,
            'method_c_v4': avg_v4,
            'improvement_v4_vs_v3': improvement
        },
        'ai_workflows': {
            'method_c_v3': avg_ai_v3,
            'method_c_v4': avg_ai_v4,
            'improvement': ai_improvement,
            'count': len(ai_v4)
        },
        'non_ai_workflows': {
            'method_c_v3': avg_non_ai_v3,
            'method_c_v4': avg_non_ai_v4,
            'count': len(non_ai_v4)
        }
    }
    
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (Method C v3 vs v4)")
    print("=" * 100)
    
    if avg_v3:
        print("\n📊 Method C v3 (LangChain-Enhanced):")
        print(f"   Node F1:            {avg_v3['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_v3['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_v3['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Cost:           ${avg_v3['avg_cost']:.4f}")
        print(f"   Valid Results:      {avg_v3['valid_count']}/100")
    
    if avg_v4:
        print("\n🆕 Method C v4 (Refined Detection + httpRequest Fix):")
        print(f"   Node F1:            {avg_v4['avg_node_f1']:.3f}")
        print(f"   Connection F1:      {avg_v4['avg_connection_f1']:.3f}")
        print(f"   Parameter Accuracy: {avg_v4['avg_parameter_accuracy']:.3f}")
        print(f"   Avg Cost:           ${avg_v4['avg_cost']:.4f}")
        print(f"   Valid Results:      {avg_v4['valid_count']}/100")
    
    if improvement:
        print("\n📈 Overall Improvement (v4 vs v3):")
        print(f"   Node F1:            {improvement['node_f1_delta']:+.3f} ({improvement['node_f1_pct']:+.1f}%)")
        print(f"   Connection F1:      {improvement['connection_f1_delta']:+.3f} ({improvement['connection_f1_pct']:+.1f}%)")
        print(f"   Parameter Accuracy: {improvement['param_accuracy_delta']:+.3f} ({improvement['param_accuracy_pct']:+.1f}%)")
        print(f"   Cost:               ${improvement['cost_delta']:+.4f}")
    
    # AI workflow specific results
    if avg_ai_v3 and avg_ai_v4:
        print("\n" + "=" * 100)
        print(f"🤖 AI AGENT WORKFLOWS ANALYSIS")
        print("=" * 100)
        
        print(f"\nv3 detected: {len(ai_v3)} workflows")
        print(f"v4 detected: {len(ai_v4)} workflows")
        print(f"Detection change: {len(ai_v4) - len(ai_v3):+d} workflows")
        
        print("\nMethod C v3 (AI workflows):")
        print(f"   Count: {len(ai_v3)}")
        print(f"   Avg Node F1: {avg_ai_v3['avg_node_f1']:.3f}")
        
        print("\nMethod C v4 (AI workflows):")
        print(f"   Count: {len(ai_v4)}")
        print(f"   Avg Node F1: {avg_ai_v4['avg_node_f1']:.3f}")
        
        if ai_improvement:
            print(f"\n🎯 AI Workflow Improvement: {ai_improvement['node_f1_delta']:+.3f} ({ai_improvement['node_f1_pct']:+.1f}%)")
            if ai_improvement['node_f1_delta'] > 0.05:
                print("   ✅ Significant improvement in AI workflows!")
    
    # Non-AI workflow results
    if avg_non_ai_v3 and avg_non_ai_v4:
        print("\n" + "=" * 100)
        print(f"📊 NON-AI WORKFLOWS ANALYSIS ({len(non_ai_v4)} workflows)")
        print("=" * 100)
        
        print(f"\nv3: {avg_non_ai_v3['avg_node_f1']:.3f}")
        print(f"v4: {avg_non_ai_v4['avg_node_f1']:.3f}")
        delta = avg_non_ai_v4['avg_node_f1'] - avg_non_ai_v3['avg_node_f1']
        print(f"Change: {delta:+.3f}")
        
        if delta > 0:
            print("   ✅ Non-AI workflows improved!")
        elif delta < -0.02:
            print("   ⚠️  Non-AI workflows regressed")
    
    # Success indicators
    print("\n" + "=" * 100)
    print("🎯 Goal Achievement:")
    print("=" * 100)
    target_node_f1 = 0.50
    target_conn_f1 = 0.25
    target_param = 0.20
    
    if avg_v4:
        if avg_v4['avg_node_f1'] >= target_node_f1:
            print(f"   ✅ Node F1 TARGET REACHED: {avg_v4['avg_node_f1']:.3f} >= {target_node_f1}")
        else:
            print(f"   ⏳ Node F1 progress: {avg_v4['avg_node_f1']:.3f} / {target_node_f1} ({avg_v4['avg_node_f1']/target_node_f1*100:.1f}%)")
        
        if avg_v4['avg_connection_f1'] >= target_conn_f1:
            print(f"   ✅ Connection F1 TARGET REACHED: {avg_v4['avg_connection_f1']:.3f} >= {target_conn_f1}")
        else:
            print(f"   ⏳ Connection F1 progress: {avg_v4['avg_connection_f1']:.3f} / {target_conn_f1} ({avg_v4['avg_connection_f1']/target_conn_f1*100:.1f}%)")
        
        if avg_v4['avg_parameter_accuracy'] >= target_param:
            print(f"   ✅ Parameter Accuracy TARGET REACHED: {avg_v4['avg_parameter_accuracy']:.3f} >= {target_param}")
        else:
            print(f"   ⏳ Parameter Accuracy progress: {avg_v4['avg_parameter_accuracy']:.3f} / {target_param} ({avg_v4['avg_parameter_accuracy']/target_param*100:.1f}%)")
    
    print("\n" + "=" * 100)
    print(f"✓ Comparison summary saved: {output_dir / 'comparison_summary.json'}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
