#!/usr/bin/env python3
"""
Analyze correlations between node accuracy and other factors
- Description length
- Number of nodes
"""

import sys
sys.path.insert(0, '.')

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from scipy import stats
except ImportError:
    # Fallback for basic correlation calculation
    class stats:
        @staticmethod
        def pearsonr(x, y):
            x = np.array(x)
            y = np.array(y)
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            std_x = np.std(x)
            std_y = np.std(y)
            n = len(x)
            r = np.sum((x - mean_x) * (y - mean_y)) / ((n - 1) * std_x * std_y)
            # Simple p-value approximation (not exact)
            t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
            from math import erf
            p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / np.sqrt(2))))
            return r, p_value

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'ggplot')
plt.rcParams['figure.figsize'] = (15, 10)

def load_data():
    """Load evaluation results and extract relevant metrics"""

    # Try to load from two_methods_comparison first
    method_a_path = Path('outputs/two_methods_comparison/method_a_results.json')

    if method_a_path.exists():
        with open(method_a_path, 'r') as f:
            results = json.load(f)
    else:
        # Fallback to main evaluation results
        detailed_path = Path('outputs/evaluation_results/detailed_per_template.json')
        with open(detailed_path, 'r') as f:
            results = json.load(f)

    # Extract data points
    data_points = []

    for result in results:
        # Skip if error or no metrics
        if result.get('error') or not result.get('metrics'):
            continue

        metrics = result['metrics']

        # Get description
        description = result.get('original_description', '')

        # Calculate description length
        desc_length = len(description)

        # Get node counts
        gt_node_count = metrics.get('gt_node_count', 0)
        llm_node_count = metrics.get('llm_node_count', 0)

        # Get F1 score (node accuracy)
        node_f1 = metrics.get('node_type_f1', 0)

        data_points.append({
            'template_id': result.get('template_id'),
            'desc_length': desc_length,
            'gt_node_count': gt_node_count,
            'llm_node_count': llm_node_count,
            'node_f1': node_f1,
            'precision': metrics.get('node_type_precision', 0),
            'recall': metrics.get('node_type_recall', 0),
            'connection_f1': metrics.get('connection_f1', 0),
        })

    return data_points


def create_correlation_plots(data_points):
    """Create correlation visualizations"""

    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract arrays for easier plotting
    desc_lengths = [d['desc_length'] for d in data_points]
    gt_node_counts = [d['gt_node_count'] for d in data_points]
    node_f1_scores = [d['node_f1'] for d in data_points]
    precisions = [d['precision'] for d in data_points]
    recalls = [d['recall'] for d in data_points]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Node Accuracy Correlations Analysis', fontsize=16, fontweight='bold')

    # 1. Node F1 vs Description Length (scatter with regression)
    ax1 = axes[0, 0]
    ax1.scatter(desc_lengths, node_f1_scores, alpha=0.5, s=50)

    # Add regression line
    z = np.polyfit(desc_lengths, node_f1_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(desc_lengths), max(desc_lengths), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Calculate correlation
    corr, p_value = stats.pearsonr(desc_lengths, node_f1_scores)
    ax1.set_xlabel('Description Length (characters)', fontsize=11)
    ax1.set_ylabel('Node Type F1 Score', fontsize=11)
    ax1.set_title(f'F1 vs Description Length\nCorrelation: {corr:.3f} (p={p_value:.4f})', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 2. Node F1 vs Number of Nodes (scatter with regression)
    ax2 = axes[0, 1]
    ax2.scatter(gt_node_counts, node_f1_scores, alpha=0.5, s=50, color='green')

    # Add regression line
    z = np.polyfit(gt_node_counts, node_f1_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(gt_node_counts), max(gt_node_counts), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

    # Calculate correlation
    corr, p_value = stats.pearsonr(gt_node_counts, node_f1_scores)
    ax2.set_xlabel('Ground Truth Node Count', fontsize=11)
    ax2.set_ylabel('Node Type F1 Score', fontsize=11)
    ax2.set_title(f'F1 vs Node Count\nCorrelation: {corr:.3f} (p={p_value:.4f})', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. Hexbin plot for F1 vs Description Length (density visualization)
    ax3 = axes[0, 2]
    hexbin = ax3.hexbin(desc_lengths, node_f1_scores, gridsize=20, cmap='YlOrRd', mincnt=1)
    ax3.set_xlabel('Description Length (characters)', fontsize=11)
    ax3.set_ylabel('Node Type F1 Score', fontsize=11)
    ax3.set_title('F1 vs Description Length (Density)', fontsize=12)
    plt.colorbar(hexbin, ax=ax3, label='Count')

    # 4. Hexbin plot for F1 vs Node Count (density visualization)
    ax4 = axes[1, 0]
    hexbin = ax4.hexbin(gt_node_counts, node_f1_scores, gridsize=15, cmap='YlGnBu', mincnt=1)
    ax4.set_xlabel('Ground Truth Node Count', fontsize=11)
    ax4.set_ylabel('Node Type F1 Score', fontsize=11)
    ax4.set_title('F1 vs Node Count (Density)', fontsize=12)
    plt.colorbar(hexbin, ax=ax4, label='Count')

    # 5. Precision vs Recall colored by Description Length
    ax5 = axes[1, 1]
    scatter = ax5.scatter(recalls, precisions, c=desc_lengths, cmap='viridis', alpha=0.6, s=50)
    ax5.set_xlabel('Recall', fontsize=11)
    ax5.set_ylabel('Precision', fontsize=11)
    ax5.set_title('Precision vs Recall\n(colored by Description Length)', fontsize=12)
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Desc Length')

    # 6. Precision vs Recall colored by Node Count
    ax6 = axes[1, 2]
    scatter = ax6.scatter(recalls, precisions, c=gt_node_counts, cmap='plasma', alpha=0.6, s=50)
    ax6.set_xlabel('Recall', fontsize=11)
    ax6.set_ylabel('Precision', fontsize=11)
    ax6.set_title('Precision vs Recall\n(colored by Node Count)', fontsize=12)
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # diagonal line
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Node Count')

    plt.tight_layout()
    plt.savefig(output_dir / 'node_accuracy_correlations.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'node_accuracy_correlations.png'}")
    plt.close()

    # Create separate detailed plots
    create_binned_analysis(data_points, output_dir)
    create_heatmap_analysis(data_points, output_dir)

    return {
        'desc_length_corr': stats.pearsonr(desc_lengths, node_f1_scores),
        'node_count_corr': stats.pearsonr(gt_node_counts, node_f1_scores)
    }


def create_binned_analysis(data_points, output_dir):
    """Create binned analysis showing average F1 in different ranges"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    desc_lengths = np.array([d['desc_length'] for d in data_points])
    gt_node_counts = np.array([d['gt_node_count'] for d in data_points])
    node_f1_scores = np.array([d['node_f1'] for d in data_points])

    # 1. Binned by description length
    ax1 = axes[0]

    # Create bins for description length
    desc_bins = [0, 500, 1000, 2000, 3000, 5000, 10000, max(desc_lengths)+1]
    desc_bin_labels = ['0-500', '500-1K', '1K-2K', '2K-3K', '3K-5K', '5K-10K', '10K+']

    bin_indices = np.digitize(desc_lengths, desc_bins)

    bin_means = []
    bin_stds = []
    bin_counts = []
    used_labels = []

    for i in range(1, len(desc_bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(node_f1_scores[mask].mean())
            bin_stds.append(node_f1_scores[mask].std())
            bin_counts.append(mask.sum())
            used_labels.append(f"{desc_bin_labels[i-1]}\n(n={mask.sum()})")

    x_pos = np.arange(len(used_labels))
    ax1.bar(x_pos, bin_means, yerr=bin_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Description Length Range (chars)', fontsize=12)
    ax1.set_ylabel('Average Node F1 Score', fontsize=12)
    ax1.set_title('Average F1 Score by Description Length Range', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(used_labels, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Calculate y-axis limit considering error bars and labels
    max_value_with_error = max([mean + std for mean, std in zip(bin_means, bin_stds)])
    ax1.set_ylim([0, max(max_value_with_error * 1.15, max(bin_means) * 1.2)])

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(bin_means, bin_stds)):
        ax1.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)

    # 2. Binned by node count
    ax2 = axes[1]

    # Create bins for node count
    node_bins = [0, 5, 10, 15, 20, 30, max(gt_node_counts)+1]
    node_bin_labels = ['1-5', '6-10', '11-15', '16-20', '21-30', '30+']

    bin_indices = np.digitize(gt_node_counts, node_bins)

    bin_means = []
    bin_stds = []
    bin_counts = []
    used_labels = []

    for i in range(1, len(node_bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(node_f1_scores[mask].mean())
            bin_stds.append(node_f1_scores[mask].std())
            bin_counts.append(mask.sum())
            used_labels.append(f"{node_bin_labels[i-1]}\n(n={mask.sum()})")

    x_pos = np.arange(len(used_labels))
    ax2.bar(x_pos, bin_means, yerr=bin_stds, capsize=5, alpha=0.7, color='forestgreen')
    ax2.set_xlabel('Number of Nodes Range', fontsize=12)
    ax2.set_ylabel('Average Node F1 Score', fontsize=12)
    ax2.set_title('Average F1 Score by Node Count Range', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(used_labels, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Calculate y-axis limit considering error bars and labels
    max_value_with_error = max([mean + std for mean, std in zip(bin_means, bin_stds)])
    ax2.set_ylim([0, max(max_value_with_error * 1.15, max(bin_means) * 1.2)])

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(bin_means, bin_stds)):
        ax2.text(i, mean + std + 0.02, f'{mean:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'node_accuracy_binned_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'node_accuracy_binned_analysis.png'}")
    plt.close()


def create_heatmap_analysis(data_points, output_dir):
    """Create 2D heatmap showing F1 scores across description length and node count"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract data
    desc_lengths = np.array([d['desc_length'] for d in data_points])
    gt_node_counts = np.array([d['gt_node_count'] for d in data_points])
    node_f1_scores = np.array([d['node_f1'] for d in data_points])

    # Create bins
    desc_bins = [0, 1000, 2000, 3000, 5000, 10000]
    node_bins = [0, 5, 10, 15, 20, 30]

    # Create 2D histogram
    heatmap_data = np.zeros((len(node_bins)-1, len(desc_bins)-1))
    count_data = np.zeros((len(node_bins)-1, len(desc_bins)-1))

    for desc_len, node_count, f1 in zip(desc_lengths, gt_node_counts, node_f1_scores):
        # Find bins
        desc_idx = np.digitize(desc_len, desc_bins) - 1
        node_idx = np.digitize(node_count, node_bins) - 1

        # Ensure within bounds
        if 0 <= desc_idx < len(desc_bins)-1 and 0 <= node_idx < len(node_bins)-1:
            heatmap_data[node_idx, desc_idx] += f1
            count_data[node_idx, desc_idx] += 1

    # Calculate averages
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_heatmap = np.divide(heatmap_data, count_data)
        avg_heatmap[np.isnan(avg_heatmap)] = 0

    # Create labels
    desc_labels = ['0-1K', '1K-2K', '2K-3K', '3K-5K', '5K-10K']
    node_labels = ['1-5', '6-10', '11-15', '16-20', '21-30']

    # Plot heatmap using matplotlib
    im = ax.imshow(avg_heatmap, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average F1 Score', rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(desc_labels)))
    ax.set_yticks(np.arange(len(node_labels)))
    ax.set_xticklabels(desc_labels)
    ax.set_yticklabels(node_labels)

    ax.set_xlabel('Description Length (chars)', fontsize=12)
    ax.set_ylabel('Number of Nodes', fontsize=12)
    ax.set_title('Average Node F1 Score Heatmap\n(Description Length × Node Count)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add value and count annotations
    for i in range(len(node_labels)):
        for j in range(len(desc_labels)):
            count = int(count_data[i, j])
            avg_val = avg_heatmap[i, j]
            if count > 0:
                # Determine text color based on background
                text_color = 'white' if avg_val < 0.5 else 'black'
                ax.text(j, i - 0.1, f'{avg_val:.3f}',
                       ha='center', va='center', fontsize=10,
                       color=text_color, fontweight='bold')
                ax.text(j, i + 0.25, f'(n={count})',
                       ha='center', va='center', fontsize=8, color=text_color)

    plt.tight_layout()
    plt.savefig(output_dir / 'node_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'node_accuracy_heatmap.png'}")
    plt.close()


def print_statistics(data_points, correlations):
    """Print statistical summary"""

    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS SUMMARY")
    print("=" * 80)

    desc_lengths = [d['desc_length'] for d in data_points]
    gt_node_counts = [d['gt_node_count'] for d in data_points]
    node_f1_scores = [d['node_f1'] for d in data_points]

    print(f"\nDataset: {len(data_points)} valid samples")

    print("\nDescription Length Statistics:")
    print(f"  Mean: {np.mean(desc_lengths):.0f} chars")
    print(f"  Median: {np.median(desc_lengths):.0f} chars")
    print(f"  Range: {min(desc_lengths)}-{max(desc_lengths)} chars")

    print("\nNode Count Statistics:")
    print(f"  Mean: {np.mean(gt_node_counts):.1f} nodes")
    print(f"  Median: {np.median(gt_node_counts):.0f} nodes")
    print(f"  Range: {min(gt_node_counts)}-{max(gt_node_counts)} nodes")

    print("\nF1 Score Statistics:")
    print(f"  Mean: {np.mean(node_f1_scores):.3f}")
    print(f"  Median: {np.median(node_f1_scores):.3f}")
    print(f"  Range: {min(node_f1_scores):.3f}-{max(node_f1_scores):.3f}")

    print("\nCorrelation Analysis:")
    desc_corr, desc_p = correlations['desc_length_corr']
    node_corr, node_p = correlations['node_count_corr']

    print(f"\nDescription Length vs F1 Score:")
    print(f"  Pearson Correlation: {desc_corr:.4f}")
    print(f"  P-value: {desc_p:.6f}")
    print(f"  Significance: {'***' if desc_p < 0.001 else '**' if desc_p < 0.01 else '*' if desc_p < 0.05 else 'Not significant'}")

    print(f"\nNode Count vs F1 Score:")
    print(f"  Pearson Correlation: {node_corr:.4f}")
    print(f"  P-value: {node_p:.6f}")
    print(f"  Significance: {'***' if node_p < 0.001 else '**' if node_p < 0.01 else '*' if node_p < 0.05 else 'Not significant'}")

    print("\nInterpretation:")
    if abs(desc_corr) < 0.3:
        print(f"  - Description length shows WEAK correlation with F1 score")
    elif abs(desc_corr) < 0.7:
        print(f"  - Description length shows MODERATE correlation with F1 score")
    else:
        print(f"  - Description length shows STRONG correlation with F1 score")

    if abs(node_corr) < 0.3:
        print(f"  - Node count shows WEAK correlation with F1 score")
    elif abs(node_corr) < 0.7:
        print(f"  - Node count shows MODERATE correlation with F1 score")
    else:
        print(f"  - Node count shows STRONG correlation with F1 score")

    print("\n" + "=" * 80)


def main():
    print("Loading evaluation data...")
    data_points = load_data()

    print(f"Loaded {len(data_points)} valid data points")

    print("\nGenerating correlation visualizations...")
    correlations = create_correlation_plots(data_points)

    print_statistics(data_points, correlations)

    print("\n✓ Analysis complete!")
    print("\nGenerated visualizations:")
    print("  1. outputs/visualizations/node_accuracy_correlations.png")
    print("  2. outputs/visualizations/node_accuracy_binned_analysis.png")
    print("  3. outputs/visualizations/node_accuracy_heatmap.png")


if __name__ == '__main__':
    main()
