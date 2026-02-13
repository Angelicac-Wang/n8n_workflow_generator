#!/usr/bin/env python3
"""
Analyze Method C v2 gaps to identify remaining issues
Compare v2 results with ground truth to find what still needs improvement
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from collections import defaultdict, Counter
from evaluation.comparison.workflow_normalizer import WorkflowNormalizer

def analyze_v2_gaps():
    """Analyze what's still missing in Method C v2"""
    
    # Load results
    results_path = Path('outputs/method_c_v2_comparison/method_c_v2_results.json')
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load templates
    from evaluation.utils.template_loader import TemplateLoader
    loader = TemplateLoader('n8n_templates/testing_data')
    templates = {str(t.get('metadata', {}).get('id') or t.get('id')): t 
                 for t in loader.load_all_templates()}
    
    normalizer = WorkflowNormalizer()
    
    # Analysis data
    missing_nodes = Counter()
    extra_nodes = Counter()
    missing_connections = []
    param_issues = defaultdict(list)
    
    low_node_f1_cases = []
    low_conn_f1_cases = []
    
    print("\n" + "=" * 80)
    print("ANALYZING METHOD C V2 GAPS")
    print("=" * 80 + "\n")
    
    for result in results:
        if not result.get('metrics'):
            continue
        
        template_id = result['template_id']
        template = templates.get(template_id)
        if not template:
            continue
        
        # Load generated workflow
        workflow_path = Path(f'outputs/method_c_v2_comparison/generated_workflows/method_c_v2_enhanced/generated_{template_id}.json')
        if not workflow_path.exists():
            continue
        
        with open(workflow_path, 'r') as f:
            llm_result = json.load(f)
        
        # Normalize
        gt = normalizer.normalize_ground_truth(template)
        llm = normalizer.normalize_llm_output(llm_result['llm_response'])
        
        # Get metrics
        metrics = result['metrics']
        node_f1 = metrics['node_type_f1']
        conn_f1 = metrics['connection_f1']
        
        # Track low performers
        if node_f1 < 0.3:
            low_node_f1_cases.append({
                'id': template_id,
                'name': result['template_name'],
                'node_f1': node_f1,
                'gt_nodes': len(gt['nodes']),
                'llm_nodes': len(llm['nodes'])
            })
        
        if conn_f1 < 0.1:
            low_conn_f1_cases.append({
                'id': template_id,
                'name': result['template_name'],
                'conn_f1': conn_f1,
                'gt_conns': len(gt['connections']),
                'llm_conns': len(llm['connections'])
            })
        
        # Analyze missing/extra nodes
        gt_types = {node['type'] for node in gt['nodes']}
        llm_types = {node['type'] for node in llm['nodes']}
        
        for node_type in gt_types - llm_types:
            missing_nodes[node_type] += 1
        
        for node_type in llm_types - gt_types:
            extra_nodes[node_type] += 1
    
    # Print analysis
    print("📊 TOP 15 MOST FREQUENTLY MISSING NODE TYPES (v2):")
    print("-" * 80)
    for node_type, count in missing_nodes.most_common(15):
        print(f"  {node_type:40s} : {count:3d} times")
    
    print("\n📊 TOP 15 MOST FREQUENTLY EXTRA NODE TYPES (v2):")
    print("-" * 80)
    for node_type, count in extra_nodes.most_common(15):
        print(f"  {node_type:40s} : {count:3d} times")
    
    print(f"\n⚠️  LOW NODE F1 CASES (< 0.3): {len(low_node_f1_cases)}")
    print("-" * 80)
    for case in sorted(low_node_f1_cases, key=lambda x: x['node_f1'])[:10]:
        print(f"  [{case['id']}] F1={case['node_f1']:.3f} | GT:{case['gt_nodes']} nodes, LLM:{case['llm_nodes']} nodes")
        print(f"      {case['name'][:76]}")
    
    print(f"\n⚠️  LOW CONNECTION F1 CASES (< 0.1): {len(low_conn_f1_cases)}")
    print("-" * 80)
    for case in sorted(low_conn_f1_cases, key=lambda x: x['conn_f1'])[:10]:
        print(f"  [{case['id']}] F1={case['conn_f1']:.3f} | GT:{case['gt_conns']} conns, LLM:{case['llm_conns']} conns")
        print(f"      {case['name'][:76]}")
    
    # Improvement suggestions
    print("\n" + "=" * 80)
    print("💡 SUGGESTED IMPROVEMENTS FOR v3")
    print("=" * 80)
    
    print("\n1️⃣  STILL COMMONLY MISSED NODE TYPES:")
    still_missing = missing_nodes.most_common(10)
    for node_type, count in still_missing:
        if count >= 5:
            print(f"   - {node_type} ({count} cases)")
    
    print("\n2️⃣  FREQUENTLY OVER-GENERATED NODE TYPES:")
    over_generated = extra_nodes.most_common(10)
    for node_type, count in over_generated:
        if count >= 5:
            print(f"   - {node_type} ({count} cases)")
    
    print("\n3️⃣  CONNECTION QUALITY ISSUES:")
    print(f"   - {len(low_conn_f1_cases)} cases with very poor connections (< 0.1)")
    print(f"   - Average Connection F1: 0.189 (target: 0.25)")
    print(f"   - Need stronger emphasis on correct node connections")
    
    print("\n4️⃣  PARAMETER ACCURACY REGRESSION:")
    print(f"   - v2: 0.139 vs v1: 0.146 (-5.4%)")
    print(f"   - Longer prompt may dilute focus on parameter details")
    print(f"   - Consider adding parameter-specific examples")
    
    # Compare with v1 gaps
    print("\n" + "=" * 80)
    print("📈 COMPARING V1 vs V2 GAP ANALYSIS")
    print("=" * 80)
    
    # Load v1 analysis if exists
    v1_analysis_path = Path('0205_enhance_with_checking_differences/analyze_method_C_gap.txt')
    if v1_analysis_path.exists():
        print("\n✓ v1 analysis file exists - comparing improvements...")
        print(f"  Location: {v1_analysis_path}")
        print("\nKey improvements in v2:")
        print(f"  - Node F1: 0.371 → 0.415 (+12.0%)")
        print(f"  - Connection F1: 0.172 → 0.189 (+9.7%)")
        print(f"  - But still need +17% to reach Node F1 target of 0.5")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    analyze_v2_gaps()
