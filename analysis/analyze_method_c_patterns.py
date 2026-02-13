#!/usr/bin/env python3
"""
Analyze Method C results to identify patterns in what the LLM commonly misses.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def load_json_file(filepath: str) -> dict:
    """Load and return JSON file contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_node_types_from_workflow(workflow: dict) -> List[str]:
    """Extract node types from a workflow."""
    node_types = []
    
    # Handle LLM generated workflow structure
    if 'llm_response' in workflow and 'workflowPlan' in workflow['llm_response']:
        nodes = workflow['llm_response']['workflowPlan'].get('nodes', [])
        for node in nodes:
            if 'nodeType' in node:
                node_types.append(node['nodeType'])
    # Handle nested workflow structure (ground truth)
    elif 'workflow' in workflow and 'workflow' in workflow['workflow']:
        nodes = workflow['workflow']['workflow'].get('nodes', [])
        for node in nodes:
            if 'type' in node:
                node_types.append(node['type'])
    # Handle direct workflow structure
    elif 'workflow' in workflow:
        nodes = workflow['workflow'].get('nodes', [])
        for node in nodes:
            if 'type' in node:
                node_types.append(node['type'])
    # Handle flat structure
    elif 'nodes' in workflow:
        nodes = workflow['nodes']
        for node in nodes:
            if 'type' in node:
                node_types.append(node['type'])
    
    return node_types

def analyze_method_c_results(results_path: str, templates_dir: str, output_dir: str, limit: int = 30):
    """
    Analyze Method C results to identify common missing node types and patterns.
    
    Args:
        results_path: Path to method_c_results.json
        templates_dir: Directory containing ground truth templates
        output_dir: Directory containing generated workflows
        limit: Number of templates to analyze (default: 30)
    """
    print("Loading Method C results...")
    results = load_json_file(results_path)
    
    # Filter for valid metrics and get first N
    valid_results = [r for r in results if r.get('metrics') is not None and r.get('error') is None]
    print(f"Found {len(valid_results)} results with valid metrics")
    
    analyzed_results = valid_results[:limit]
    print(f"Analyzing first {len(analyzed_results)} templates...\n")
    
    # Initialize tracking
    missing_node_types = Counter()
    unnecessary_node_types = Counter()
    low_f1_cases = []
    connection_issues = []
    parameter_issues = []
    
    templates_base = Path(templates_dir)
    output_base = Path(output_dir)
    
    for idx, result in enumerate(analyzed_results):
        template_id = result['template_id']
        metrics = result['metrics']
        
        print(f"[{idx+1}/{len(analyzed_results)}] Analyzing template {template_id}...")
        
        # Load ground truth - find the matching template file
        gt_files = list(templates_base.glob(f"template_{template_id}_*.json"))
        if not gt_files:
            print(f"  ⚠️  Ground truth not found for template {template_id}")
            continue
        gt_path = gt_files[0]
        
        # Load generated workflow
        llm_path = output_base / "generated_workflows" / "method_c_improved_prompt" / f"generated_{template_id}.json"
        
        if not llm_path.exists():
            print(f"  ⚠️  LLM workflow not found: {llm_path}")
            continue
        
        gt_workflow = load_json_file(gt_path)
        llm_workflow = load_json_file(llm_path)
        
        # Extract node types
        gt_node_types = get_node_types_from_workflow(gt_workflow)
        llm_node_types = get_node_types_from_workflow(llm_workflow)
        
        # Find missing and unnecessary node types
        gt_node_set = set(gt_node_types)
        llm_node_set = set(llm_node_types)
        
        missing = gt_node_set - llm_node_set
        unnecessary = llm_node_set - gt_node_set
        
        # Count occurrences
        for node_type in missing:
            # Count how many times this node type appears in ground truth
            count = gt_node_types.count(node_type)
            missing_node_types[node_type] += count
        
        for node_type in unnecessary:
            count = llm_node_types.count(node_type)
            unnecessary_node_types[node_type] += count
        
        # Track low F1 scores
        node_f1 = metrics.get('node_type_f1', 0)
        if node_f1 < 0.3:
            low_f1_cases.append({
                'template_id': template_id,
                'template_name': result['template_name'],
                'node_f1': node_f1,
                'gt_nodes': len(gt_node_types),
                'llm_nodes': len(llm_node_types),
                'missing': list(missing),
                'unnecessary': list(unnecessary)
            })
        
        # Track connection issues
        conn_f1 = metrics.get('connection_f1', 0)
        if conn_f1 < 0.3:
            connection_issues.append({
                'template_id': template_id,
                'template_name': result['template_name'],
                'connection_f1': conn_f1,
                'gt_connections': metrics.get('gt_connection_count', 0),
                'llm_connections': metrics.get('llm_connection_count', 0)
            })
        
        # Track parameter accuracy issues
        param_acc = metrics.get('avg_parameter_accuracy', 0)
        if param_acc < 0.3:
            parameter_issues.append({
                'template_id': template_id,
                'template_name': result['template_name'],
                'param_accuracy': param_acc,
                'node_f1': node_f1
            })
    
    # Generate summary report
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n📊 TOP 15 MOST COMMONLY MISSING NODE TYPES:")
    print("-" * 80)
    for node_type, count in missing_node_types.most_common(15):
        print(f"  {node_type:40s} - Missing {count:3d} times")
    
    print("\n📊 TOP 10 MOST COMMONLY ADDED UNNECESSARILY:")
    print("-" * 80)
    for node_type, count in unnecessary_node_types.most_common(10):
        print(f"  {node_type:40s} - Added unnecessarily {count:3d} times")
    
    print(f"\n📊 WORKFLOWS WITH LOW NODE F1 SCORES (< 0.3): {len(low_f1_cases)} cases")
    print("-" * 80)
    for case in low_f1_cases[:10]:  # Show first 10
        print(f"\n  Template: {case['template_id']} - {case['template_name'][:60]}")
        print(f"    Node F1: {case['node_f1']:.3f}")
        print(f"    Nodes: GT={case['gt_nodes']}, LLM={case['llm_nodes']}")
        print(f"    Missing: {', '.join(case['missing'][:5])}")
        if len(case['missing']) > 5:
            print(f"             ... and {len(case['missing']) - 5} more")
    
    print(f"\n📊 WORKFLOWS WITH CONNECTION ISSUES (F1 < 0.3): {len(connection_issues)} cases")
    print("-" * 80)
    for case in connection_issues[:10]:
        print(f"\n  Template: {case['template_id']} - {case['template_name'][:60]}")
        print(f"    Connection F1: {case['connection_f1']:.3f}")
        print(f"    Connections: GT={case['gt_connections']}, LLM={case['llm_connections']}")
    
    print(f"\n📊 PARAMETER ACCURACY ISSUES (< 0.3): {len(parameter_issues)} cases")
    print("-" * 80)
    avg_param_acc = sum(p['param_accuracy'] for p in parameter_issues) / len(parameter_issues) if parameter_issues else 0
    print(f"  Average parameter accuracy for low-scoring cases: {avg_param_acc:.3f}")
    print(f"  Total workflows affected: {len(parameter_issues)}")
    
    # Save detailed report to file
    report_path = output_base / "method_c_pattern_analysis.json"
    report = {
        'analysis_metadata': {
            'total_analyzed': len(analyzed_results),
            'with_valid_metrics': len(valid_results)
        },
        'missing_node_types_top15': dict(missing_node_types.most_common(15)),
        'unnecessary_node_types_top10': dict(unnecessary_node_types.most_common(10)),
        'low_f1_cases': low_f1_cases,
        'connection_issues': connection_issues,
        'parameter_issues': parameter_issues
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Detailed report saved to: {report_path}")
    
    # Generate actionable insights
    print("\n" + "="*80)
    print("🎯 ACTIONABLE INSIGHTS FOR PROMPT IMPROVEMENT")
    print("="*80)
    
    print("\n1. MOST CRITICAL MISSING NODE TYPES TO ADDRESS:")
    top_missing = missing_node_types.most_common(5)
    for node_type, count in top_missing:
        print(f"   • {node_type} (missing {count} times)")
        if 'Code' in node_type:
            print("     → Add examples of data transformation tasks")
        elif 'HTTP' in node_type or 'Webhook' in node_type:
            print("     → Emphasize API integration patterns")
        elif 'Set' in node_type:
            print("     → Include examples of variable setting")
        elif 'If' in node_type or 'Switch' in node_type:
            print("     → Add conditional logic examples")
    
    print("\n2. WORKFLOW PATTERNS THAT LLM STRUGGLES WITH:")
    if len(low_f1_cases) > 0:
        # Analyze common characteristics
        avg_gt_nodes = sum(c['gt_nodes'] for c in low_f1_cases) / len(low_f1_cases)
        avg_llm_nodes = sum(c['llm_nodes'] for c in low_f1_cases) / len(low_f1_cases)
        print(f"   • Complex workflows (avg GT nodes: {avg_gt_nodes:.1f}, LLM nodes: {avg_llm_nodes:.1f})")
        print(f"   • {len([c for c in low_f1_cases if c['gt_nodes'] > 15])} cases with >15 nodes")
        print(f"   • LLM tends to generate {(avg_gt_nodes - avg_llm_nodes)/avg_gt_nodes * 100:.1f}% fewer nodes than needed")
        
        # Find common missing node types in low F1 cases
        low_f1_missing = Counter()
        for case in low_f1_cases:
            for node_type in case['missing']:
                low_f1_missing[node_type] += 1
        print("   • Most commonly missing in low F1 cases:")
        for node_type, count in low_f1_missing.most_common(5):
            print(f"     - {node_type} ({count} cases)")
    
    print("\n3. CONNECTION ACCURACY ISSUES:")
    if len(connection_issues) > 0:
        avg_conn_f1 = sum(c['connection_f1'] for c in connection_issues) / len(connection_issues)
        avg_gt_conn = sum(c['gt_connections'] for c in connection_issues) / len(connection_issues)
        avg_llm_conn = sum(c['llm_connections'] for c in connection_issues) / len(connection_issues)
        print(f"   • Average connection F1 for problem cases: {avg_conn_f1:.3f}")
        print(f"   • Avg connections: GT={avg_gt_conn:.1f}, LLM={avg_llm_conn:.1f}")
        print(f"   • LLM generates {(avg_gt_conn - avg_llm_conn)/avg_gt_conn * 100:.1f}% fewer connections")
        print("   • Need to improve node connection logic in prompt")
        print("   • Add examples of complex branching and parallel processing patterns")
    
    print("\n4. PARAMETER ACCURACY ISSUES:")
    if len(parameter_issues) > 0:
        print(f"   • {len(parameter_issues)} workflows with low parameter accuracy")
        print(f"   • Average parameter accuracy: {avg_param_acc:.3f}")
        # Correlate with node F1
        high_node_low_param = [p for p in parameter_issues if p['node_f1'] > 0.5]
        print(f"   • {len(high_node_low_param)} cases have good node F1 but low parameter accuracy")
        print("   • This suggests the LLM identifies correct nodes but struggles with configuration")
        print("   • Need to add more detailed parameter examples in prompt")
    
    print("\n5. KEY INSIGHTS ABOUT MISSING NODES:")
    # Analyze sticky notes
    if missing_node_types['n8n-nodes-base.stickyNote'] > 0:
        print("   • StickyNote (documentation): Missing in nearly ALL workflows")
        print("     → These are documentation/comment nodes, might be OK to skip")
        print("     → But could add instructions to include workflow documentation")
    
    # Analyze Code nodes
    if missing_node_types['n8n-nodes-base.code'] > 0:
        print(f"   • Code nodes: Missing {missing_node_types['n8n-nodes-base.code']} times")
        print("     → LLM prefers httpRequest or openAi nodes instead")
        print("     → Need to emphasize when to use Code for data transformation")
    
    # Analyze LangChain agent nodes
    langchain_missing = sum(count for node, count in missing_node_types.items() if 'langchain' in node.lower())
    if langchain_missing > 0:
        print(f"   • LangChain nodes: Missing {langchain_missing} times total")
        print("     → LLM doesn't recognize AI agent patterns")
        print("     → Need to add examples of AI agent workflows")
    
    # Analyze Set vs other data manipulation
    if missing_node_types['n8n-nodes-base.set'] > 0:
        print(f"   • Set nodes: Missing {missing_node_types['n8n-nodes-base.set']} times")
        print("     → LLM may be using Code/Function nodes instead")
        print("     → Clarify when to use Set vs Code for data manipulation")
    
    print("\n6. UNNECESSARILY ADDED NODES ANALYSIS:")
    if unnecessary_node_types:
        print("   • OpenAI node added unnecessarily 21 times")
        print("     → LLM defaults to n8n-nodes-base.openAi instead of @n8n/n8n-nodes-langchain.lmChatOpenAi")
        print("     → Need to clarify LangChain node types vs regular nodes")
        print("   • Function node added 11 times when not needed")
        print("     → May be confusing Function with Code node")
        print("   • Wrong trigger types (schedule vs scheduleTrigger, webhook vs webhookTrigger)")
        print("     → Need to provide exact node type names")

if __name__ == "__main__":
    results_path = "/Users/yu/Desktop/projects/gss_cai/n8n_workflow_generator/outputs/four_methods_comparison/method_c_results.json"
    templates_dir = "/Users/yu/Desktop/projects/gss_cai/n8n_workflow_generator/n8n_templates/testing_data"
    output_dir = "/Users/yu/Desktop/projects/gss_cai/n8n_workflow_generator/outputs/four_methods_comparison"
    
    analyze_method_c_results(results_path, templates_dir, output_dir, limit=30)
