#!/usr/bin/env python3
"""
Format comparison summary as a table from existing JSON results
"""

import json
from pathlib import Path

def format_comparison_table(summary_path):
    """Format comparison summary as a table"""
    
    with open(summary_path, 'r') as f:
        avg_metrics = json.load(f)
    
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
    if avg_metrics.get('method_a'):
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
    import sys
    
    # Default path
    summary_path = Path('outputs/four_methods_comparison/comparison_summary.json')
    
    # Allow override via command line
    if len(sys.argv) > 1:
        summary_path = Path(sys.argv[1])
    
    if not summary_path.exists():
        print(f"Error: File not found: {summary_path}")
        sys.exit(1)
    
    format_comparison_table(summary_path)
