#!/usr/bin/env python3
"""
Analyze Method C v3 Results - Find Common Mistakes and Missing Patterns

Similar to v2 analysis, but for v3 to see what improved and what still needs work.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict


def load_results():
    """Load v3 results"""
    results_path = Path('outputs/method_c_v3_comparison/method_c_v3_results.json')
    with open(results_path, 'r') as f:
        return json.load(f)


def load_ground_truth(template_id):
    """Load ground truth template"""
    gt_files = list(Path('n8n_templates/testing_data').glob(f'template_{template_id}_*.json'))
    if not gt_files:
        return None
    
    with open(gt_files[0], 'r') as f:
        return json.load(f)


def load_generated_workflow(template_id):
    """Load v3 generated workflow"""
    gen_path = Path(f'outputs/method_c_v3_comparison/generated_workflows/method_c_v3_langchain/generated_{template_id}.json')
    if not gen_path.exists():
        return None
    
    with open(gen_path, 'r') as f:
        return json.load(f)


def analyze_node_gaps(results):
    """Analyze missing and extra node types"""
    missing_nodes = Counter()
    extra_nodes = Counter()
    
    for result in results:
        if not result.get('metrics'):
            continue
        
        template_id = result['template_id']
        gt = load_ground_truth(template_id)
        gen = load_generated_workflow(template_id)
        
        if not gt or not gen:
            continue
        
        # Extract node types
        gt_nodes = gt.get('workflow', {}).get('nodes', [])
        gt_node_types = [n.get('name', n.get('type', 'unknown')) for n in gt_nodes]
        
        gen_nodes = gen.get('llm_response', {}).get('workflowPlan', {}).get('nodes', [])
        gen_node_types = [n.get('nodeType', 'unknown') for n in gen_nodes]
        
        # Find missing nodes (in GT but not in generated)
        gt_counter = Counter(gt_node_types)
        gen_counter = Counter(gen_node_types)
        
        for node_type, count in gt_counter.items():
            if node_type == 'n8n-nodes-base.stickyNote':
                continue  # Skip sticky notes
            missing_count = max(0, count - gen_counter.get(node_type, 0))
            if missing_count > 0:
                missing_nodes[node_type] += missing_count
        
        # Find extra nodes (in generated but not in GT)
        for node_type, count in gen_counter.items():
            extra_count = max(0, count - gt_counter.get(node_type, 0))
            if extra_count > 0:
                extra_nodes[node_type] += extra_count
    
    return missing_nodes, extra_nodes


def analyze_low_performers(results):
    """Analyze workflows with low F1 scores"""
    low_node_f1 = []
    low_conn_f1 = []
    low_param = []
    
    for result in results:
        if not result.get('metrics'):
            continue
        
        metrics = result['metrics']
        template_id = result['template_id']
        name = result['template_name']
        
        if metrics['node_type_f1'] < 0.3:
            low_node_f1.append({
                'id': template_id,
                'name': name,
                'f1': metrics['node_type_f1'],
                'is_ai': result.get('is_ai_workflow', False)
            })
        
        if metrics['connection_f1'] < 0.2:
            low_conn_f1.append({
                'id': template_id,
                'name': name,
                'f1': metrics['connection_f1'],
                'is_ai': result.get('is_ai_workflow', False)
            })
        
        if metrics['avg_parameter_accuracy'] < 0.1:
            low_param.append({
                'id': template_id,
                'name': name,
                'acc': metrics['avg_parameter_accuracy'],
                'is_ai': result.get('is_ai_workflow', False)
            })
    
    return low_node_f1, low_conn_f1, low_param


def analyze_improvements(results):
    """Compare v2 vs v3 performance"""
    # Load v2 results
    v2_results_path = Path('outputs/method_c_v2_comparison/method_c_v2_results.json')
    with open(v2_results_path, 'r') as f:
        v2_results = json.load(f)
    
    v2_dict = {r['template_id']: r for r in v2_results if r.get('metrics')}
    
    improved = []
    regressed = []
    
    for result in results:
        if not result.get('metrics'):
            continue
        
        tid = result['template_id']
        if tid not in v2_dict or not v2_dict[tid].get('metrics'):
            continue
        
        v2_f1 = v2_dict[tid]['metrics']['node_type_f1']
        v3_f1 = result['metrics']['node_type_f1']
        delta = v3_f1 - v2_f1
        
        if abs(delta) < 0.01:
            continue
        
        entry = {
            'id': tid,
            'name': result['template_name'],
            'v2': v2_f1,
            'v3': v3_f1,
            'delta': delta,
            'is_ai': result.get('is_ai_workflow', False)
        }
        
        if delta > 0:
            improved.append(entry)
        else:
            regressed.append(entry)
    
    return improved, regressed


def main():
    print("\n" + "=" * 100)
    print("METHOD C v3 GAP ANALYSIS")
    print("=" * 100)
    
    # Load results
    results = load_results()
    valid_results = [r for r in results if r.get('metrics')]
    
    print(f"\n📊 Overview:")
    print(f"   Total workflows: {len(results)}")
    print(f"   Valid results: {len(valid_results)}")
    
    # Calculate average metrics
    avg_node_f1 = sum(r['metrics']['node_type_f1'] for r in valid_results) / len(valid_results)
    avg_conn_f1 = sum(r['metrics']['connection_f1'] for r in valid_results) / len(valid_results)
    avg_param = sum(r['metrics']['avg_parameter_accuracy'] for r in valid_results) / len(valid_results)
    
    print(f"   Average Node F1: {avg_node_f1:.3f}")
    print(f"   Average Connection F1: {avg_conn_f1:.3f}")
    print(f"   Average Parameter Accuracy: {avg_param:.3f}")
    
    # Analyze node gaps
    print("\n" + "=" * 100)
    print("📊 TOP 15 MOST COMMONLY MISSING NODE TYPES")
    print("-" * 100)
    
    missing_nodes, extra_nodes = analyze_node_gaps(valid_results)
    
    for node_type, count in missing_nodes.most_common(15):
        print(f"  {node_type:50} - Missing {count:3} times")
    
    print("\n" + "=" * 100)
    print("📊 TOP 10 MOST COMMONLY ADDED UNNECESSARILY")
    print("-" * 100)
    
    for node_type, count in extra_nodes.most_common(10):
        print(f"  {node_type:50} - Added unnecessarily {count:3} times")
    
    # Analyze low performers
    low_node_f1, low_conn_f1, low_param = analyze_low_performers(valid_results)
    
    print("\n" + "=" * 100)
    print(f"📊 WORKFLOWS WITH LOW NODE F1 SCORES (< 0.3): {len(low_node_f1)} cases")
    print("-" * 100)
    
    for case in sorted(low_node_f1, key=lambda x: x['f1'])[:10]:
        ai_marker = "🤖" if case['is_ai'] else "  "
        print(f"\n  {ai_marker} Template: {case['id']} - {case['name'][:60]}")
        print(f"     Node F1: {case['f1']:.3f}")
    
    print("\n" + "=" * 100)
    print(f"📊 WORKFLOWS WITH CONNECTION ISSUES (F1 < 0.2): {len(low_conn_f1)} cases")
    print("-" * 100)
    print(f"   (Showing first 10)")
    
    for case in sorted(low_conn_f1, key=lambda x: x['f1'])[:10]:
        ai_marker = "🤖" if case['is_ai'] else "  "
        print(f"  {ai_marker} {case['id']}: {case['name'][:60]} - F1: {case['f1']:.3f}")
    
    print("\n" + "=" * 100)
    print(f"📊 PARAMETER ACCURACY ISSUES (< 0.1): {len(low_param)} cases")
    print("-" * 100)
    print(f"   Average parameter accuracy for low-scoring cases: {sum(c['acc'] for c in low_param)/len(low_param) if low_param else 0:.3f}")
    
    # Analyze improvements
    print("\n" + "=" * 100)
    print("📈 TOP 10 MOST IMPROVED (v3 vs v2)")
    print("-" * 100)
    
    improved, regressed = analyze_improvements(valid_results)
    
    for case in sorted(improved, key=lambda x: x['delta'], reverse=True)[:10]:
        ai_marker = "🤖" if case['is_ai'] else "  "
        print(f"  {ai_marker} {case['id']}: {case['v2']:.3f} → {case['v3']:.3f} (+{case['delta']:.3f}) - {case['name'][:45]}")
    
    print("\n" + "=" * 100)
    print("📉 TOP 10 MOST REGRESSED (v3 vs v2)")
    print("-" * 100)
    
    for case in sorted(regressed, key=lambda x: x['delta'])[:10]:
        ai_marker = "🤖" if case['is_ai'] else "  "
        print(f"  {ai_marker} {case['id']}: {case['v2']:.3f} → {case['v3']:.3f} ({case['delta']:.3f}) - {case['name'][:45]}")
    
    # AI workflow analysis
    ai_workflows = [r for r in valid_results if r.get('is_ai_workflow')]
    non_ai_workflows = [r for r in valid_results if not r.get('is_ai_workflow')]
    
    print("\n" + "=" * 100)
    print(f"🤖 AI WORKFLOW DETECTION ANALYSIS")
    print("=" * 100)
    print(f"   Detected as AI workflows: {len(ai_workflows)}")
    print(f"   Non-AI workflows: {len(non_ai_workflows)}")
    
    if ai_workflows:
        ai_avg = sum(r['metrics']['node_type_f1'] for r in ai_workflows) / len(ai_workflows)
        print(f"   AI workflows avg Node F1: {ai_avg:.3f}")
    
    if non_ai_workflows:
        non_ai_avg = sum(r['metrics']['node_type_f1'] for r in non_ai_workflows) / len(non_ai_workflows)
        print(f"   Non-AI workflows avg Node F1: {non_ai_avg:.3f}")
    
    # Summary
    print("\n" + "=" * 100)
    print("🎯 KEY INSIGHTS FOR FURTHER IMPROVEMENT")
    print("=" * 100)
    
    print(f"\n1. MOST CRITICAL MISSING NODE TYPES:")
    for node_type, count in missing_nodes.most_common(5):
        print(f"   • {node_type} (missing {count} times)")
    
    print(f"\n2. OVERUSED NODE TYPES:")
    for node_type, count in extra_nodes.most_common(3):
        print(f"   • {node_type} (added unnecessarily {count} times)")
    
    print(f"\n3. PERFORMANCE SUMMARY:")
    print(f"   • {len(low_node_f1)} workflows still have Node F1 < 0.3")
    print(f"   • {len(low_conn_f1)} workflows have Connection F1 < 0.2")
    print(f"   • {len(improved)} workflows improved from v2")
    print(f"   • {len(regressed)} workflows regressed from v2")
    
    if improved:
        avg_improvement = sum(c['delta'] for c in improved) / len(improved)
        print(f"   • Average improvement when improved: +{avg_improvement:.3f}")
    
    if regressed:
        avg_regression = sum(c['delta'] for c in regressed) / len(regressed)
        print(f"   • Average regression when regressed: {avg_regression:.3f}")
    
    print("\n" + "=" * 100)
    
    # Save detailed report
    report = {
        'missing_nodes': dict(missing_nodes.most_common(30)),
        'extra_nodes': dict(extra_nodes.most_common(20)),
        'low_node_f1_cases': low_node_f1,
        'low_connection_cases': low_conn_f1,
        'low_parameter_cases': low_param,
        'improved_cases': improved,
        'regressed_cases': regressed,
        'summary': {
            'avg_node_f1': avg_node_f1,
            'avg_connection_f1': avg_conn_f1,
            'avg_parameter_accuracy': avg_param,
            'num_low_node_f1': len(low_node_f1),
            'num_low_connection': len(low_conn_f1),
            'num_low_parameter': len(low_param),
            'num_improved': len(improved),
            'num_regressed': len(regressed)
        }
    }
    
    output_path = Path('outputs/method_c_v3_comparison/v3_gap_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Detailed report saved to: {output_path}")
    print("=" * 100 + "\n")


if __name__ == '__main__':
    main()
