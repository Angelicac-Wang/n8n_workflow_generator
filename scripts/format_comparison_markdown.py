#!/usr/bin/env python3
"""
Format comparison summary as Markdown table for easy copying
"""

import json
from pathlib import Path

def format_comparison_markdown(summary_path):
    """Format comparison summary as Markdown table"""
    
    with open(summary_path, 'r') as f:
        avg_metrics = json.load(f)
    
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (Markdown Table Format)")
    print("=" * 100)
    print("\nCopy the table below:\n")
    
    # Main comparison table
    print("## Comparison Summary (All Four Methods)")
    print()
    print("| Method | Node F1 | Connection F1 | Param Accuracy | Total Cost | Valid Results |")
    print("|--------|---------|---------------|----------------|------------|---------------|")
    
    def format_method_row(method_label, metrics_dict):
        if not metrics_dict:
            return None
        
        cost_str = f"${metrics_dict['avg_cost']:.4f}"
        if 'avg_description_generation_cost' in metrics_dict:
            cost_str += f"<br>(Desc: ${metrics_dict['avg_description_generation_cost']:.4f}, Workflow: ${metrics_dict['avg_workflow_generation_cost']:.4f})"
        elif 'avg_description_optimization_cost' in metrics_dict:
            cost_str += f"<br>(Opt: ${metrics_dict['avg_description_optimization_cost']:.4f}, Workflow: ${metrics_dict['avg_workflow_generation_cost']:.4f})"
        
        return f"| {method_label} | {metrics_dict['avg_node_f1']:.3f} | {metrics_dict['avg_connection_f1']:.3f} | {metrics_dict['avg_parameter_accuracy']:.3f} | {cost_str} | {metrics_dict['valid_count']}/100 |"
    
    if avg_metrics.get('method_a'):
        print(format_method_row('🅰️ Method A (Base Prompt + Direct)', avg_metrics['method_a']))
    if avg_metrics.get('method_b'):
        print(format_method_row('🅱️ Method B (Base Prompt + AI Desc)', avg_metrics['method_b']))
    if avg_metrics.get('method_c'):
        print(format_method_row('🅲 Method C (Improved Prompt + Direct)', avg_metrics['method_c']))
    if avg_metrics.get('method_d'):
        print(format_method_row('🅳 Method D (Improved Prompt + Two-Stage)', avg_metrics['method_d']))
    
    # Improvements table
    if avg_metrics.get('method_a'):
        baseline = avg_metrics['method_a']
        print()
        print("## Improvements vs Method A (Baseline)")
        print()
        print("| Comparison | Node F1 Δ | Connection F1 Δ | Param Accuracy Δ | Cost Δ |")
        print("|------------|-----------|-----------------|-----------------|--------|")
        
        def format_improvement_row(label, imp_dict):
            if not imp_dict:
                return None
            node_f1_pct = (imp_dict['node_f1_delta']/baseline['avg_node_f1']*100) if baseline['avg_node_f1'] > 0 else 0
            conn_f1_pct = (imp_dict['connection_f1_delta']/baseline['avg_connection_f1']*100) if baseline['avg_connection_f1'] > 0 else 0
            param_pct = (imp_dict['parameter_accuracy_delta']/baseline['avg_parameter_accuracy']*100) if baseline['avg_parameter_accuracy'] > 0 else 0
            
            return f"| {label} | {imp_dict['node_f1_delta']:+.3f} ({node_f1_pct:+.1f}%) | {imp_dict['connection_f1_delta']:+.3f} ({conn_f1_pct:+.1f}%) | {imp_dict['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%) | ${imp_dict['cost_delta']:+.4f} |"
        
        if avg_metrics.get('improvement_b_vs_a'):
            print(format_improvement_row("Method B vs A", avg_metrics['improvement_b_vs_a']))
        if avg_metrics.get('improvement_c_vs_a'):
            print(format_improvement_row("Method C vs A", avg_metrics['improvement_c_vs_a']))
        if avg_metrics.get('improvement_d_vs_a'):
            print(format_improvement_row("Method D vs A", avg_metrics['improvement_d_vs_a']))
    
    print()
    print("=" * 100)
    print("\nCSV Format (for Excel/Google Sheets):")
    print("=" * 100)
    print()
    
    # CSV format
    print("Method,Node F1,Connection F1,Param Accuracy,Total Cost,Valid Results")
    if avg_metrics.get('method_a'):
        m = avg_metrics['method_a']
        print(f"Method A (Base Prompt + Direct),{m['avg_node_f1']:.3f},{m['avg_connection_f1']:.3f},{m['avg_parameter_accuracy']:.3f},${m['avg_cost']:.4f},{m['valid_count']}/100")
    if avg_metrics.get('method_b'):
        m = avg_metrics['method_b']
        print(f"Method B (Base Prompt + AI Desc),{m['avg_node_f1']:.3f},{m['avg_connection_f1']:.3f},{m['avg_parameter_accuracy']:.3f},${m['avg_cost']:.4f},{m['valid_count']}/100")
    if avg_metrics.get('method_c'):
        m = avg_metrics['method_c']
        print(f"Method C (Improved Prompt + Direct),{m['avg_node_f1']:.3f},{m['avg_connection_f1']:.3f},{m['avg_parameter_accuracy']:.3f},${m['avg_cost']:.4f},{m['valid_count']}/100")
    if avg_metrics.get('method_d'):
        m = avg_metrics['method_d']
        print(f"Method D (Improved Prompt + Two-Stage),{m['avg_node_f1']:.3f},{m['avg_connection_f1']:.3f},{m['avg_parameter_accuracy']:.3f},${m['avg_cost']:.4f},{m['valid_count']}/100")
    
    print()
    print("Comparison,Node F1 Delta,Connection F1 Delta,Param Accuracy Delta,Cost Delta")
    if avg_metrics.get('improvement_b_vs_a'):
        i = avg_metrics['improvement_b_vs_a']
        baseline = avg_metrics['method_a']
        node_pct = (i['node_f1_delta']/baseline['avg_node_f1']*100) if baseline['avg_node_f1'] > 0 else 0
        conn_pct = (i['connection_f1_delta']/baseline['avg_connection_f1']*100) if baseline['avg_connection_f1'] > 0 else 0
        param_pct = (i['parameter_accuracy_delta']/baseline['avg_parameter_accuracy']*100) if baseline['avg_parameter_accuracy'] > 0 else 0
        print(f"Method B vs A,{i['node_f1_delta']:+.3f} ({node_pct:+.1f}%),{i['connection_f1_delta']:+.3f} ({conn_pct:+.1f}%),{i['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%),${i['cost_delta']:+.4f}")
    if avg_metrics.get('improvement_c_vs_a'):
        i = avg_metrics['improvement_c_vs_a']
        baseline = avg_metrics['method_a']
        node_pct = (i['node_f1_delta']/baseline['avg_node_f1']*100) if baseline['avg_node_f1'] > 0 else 0
        conn_pct = (i['connection_f1_delta']/baseline['avg_connection_f1']*100) if baseline['avg_connection_f1'] > 0 else 0
        param_pct = (i['parameter_accuracy_delta']/baseline['avg_parameter_accuracy']*100) if baseline['avg_parameter_accuracy'] > 0 else 0
        print(f"Method C vs A,{i['node_f1_delta']:+.3f} ({node_pct:+.1f}%),{i['connection_f1_delta']:+.3f} ({conn_pct:+.1f}%),{i['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%),${i['cost_delta']:+.4f}")
    if avg_metrics.get('improvement_d_vs_a'):
        i = avg_metrics['improvement_d_vs_a']
        baseline = avg_metrics['method_a']
        node_pct = (i['node_f1_delta']/baseline['avg_node_f1']*100) if baseline['avg_node_f1'] > 0 else 0
        conn_pct = (i['connection_f1_delta']/baseline['avg_connection_f1']*100) if baseline['avg_connection_f1'] > 0 else 0
        param_pct = (i['parameter_accuracy_delta']/baseline['avg_parameter_accuracy']*100) if baseline['avg_parameter_accuracy'] > 0 else 0
        print(f"Method D vs A,{i['node_f1_delta']:+.3f} ({node_pct:+.1f}%),{i['connection_f1_delta']:+.3f} ({conn_pct:+.1f}%),{i['parameter_accuracy_delta']:+.3f} ({param_pct:+.1f}%),${i['cost_delta']:+.4f}")


if __name__ == '__main__':
    import sys
    
    # Default path
    summary_path = Path('outputs/four_methods_comparison/comparison_summary.json')
    
    # Allow override via command line
    if len(sys.argv) > 1:
        summary_path = Path(sys.argv[1])
    
    if not summary_path.exists():
        print(f"Error: File not found: {summary_path}")
        sys.exit(1)
    
    format_comparison_markdown(summary_path)
