#!/usr/bin/env python3
"""
Report Generator

Generate visualizations and reports for evaluation results.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class ReportGenerator:
    """
    Generate visualizations for evaluation results
    """

    def __init__(self, config: Dict):
        """
        Initialize report generator

        Args:
            config: Configuration dictionary with visualization settings
        """
        self.config = config
        viz_config = config.get('visualization', {})

        # Set style
        plt.style.use(viz_config.get('style', 'seaborn-v0_8-darkgrid'))

        self.figsize = tuple(viz_config.get('figure_size', [12, 8]))
        self.dpi = viz_config.get('dpi', 300)

    def generate_all_visualizations(self, results: List[Dict], output_dir: Path):
        """
        Generate all visualization reports

        Args:
            results: List of evaluation results
            output_dir: Output directory for visualizations
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_results = [r for r in results if r.get('metrics')]

        if not valid_results:
            print("No valid results to visualize")
            return

        print(f"\nGenerating visualizations in {output_dir}...")

        # 1. Node accuracy distribution
        self._plot_distribution(
            [r['metrics']['node_type_f1'] for r in valid_results],
            "Node Type F1 Score Distribution",
            output_dir / "node_accuracy_distribution.png"
        )

        # 2. Connection accuracy distribution
        self._plot_distribution(
            [r['metrics']['connection_f1'] for r in valid_results],
            "Connection F1 Score Distribution",
            output_dir / "connection_accuracy_distribution.png"
        )

        # 3. Parameter accuracy distribution
        self._plot_distribution(
            [r['metrics']['avg_parameter_accuracy'] for r in valid_results],
            "Parameter Match Accuracy Distribution",
            output_dir / "parameter_accuracy_distribution.png"
        )

        # 4. Cost analysis
        self._plot_cost_analysis(valid_results, output_dir / "cost_analysis.png")

        # 5. Comparison heatmap
        self._plot_comparison_heatmap(valid_results, output_dir / "comparison_heatmap.png")

        # 6. Metric correlations
        self._plot_correlation_matrix(valid_results, output_dir / "metric_correlations.png")

        print("✓ Visualizations complete!")

    def _plot_distribution(self, values: List[float], title: str, output_path: Path):
        """
        Plot histogram + KDE for metric distribution

        Args:
            values: List of metric values
            title: Plot title
            output_path: Output file path
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.histplot(values, kde=True, bins=30, ax=ax)

        mean_val = np.mean(values)
        median_val = np.median(values)

        ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')

        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

    def _plot_cost_analysis(self, results: List[Dict], output_path: Path):
        """
        Multi-panel cost analysis

        Args:
            results: List of results
            output_path: Output file path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        costs = [r['metrics']['total_cost'] for r in results]
        prompt_tokens = [r['metrics']['usage']['prompt_tokens'] for r in results]
        completion_tokens = [r['metrics']['usage']['completion_tokens'] for r in results]

        # 1. Cost distribution
        sns.histplot(costs, kde=True, bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('Cost Distribution (USD)')
        axes[0, 0].set_xlabel('Cost per Template ($)')

        # 2. Token usage scatter
        axes[0, 1].scatter(prompt_tokens, completion_tokens, alpha=0.6)
        axes[0, 1].set_xlabel('Prompt Tokens')
        axes[0, 1].set_ylabel('Completion Tokens')
        axes[0, 1].set_title('Token Usage Pattern')

        # 3. Cumulative cost
        sorted_costs = sorted(costs)
        cumulative_cost = np.cumsum(sorted_costs)
        axes[1, 0].plot(cumulative_cost)
        axes[1, 0].set_xlabel('Template Index (sorted by cost)')
        axes[1, 0].set_ylabel('Cumulative Cost ($)')
        axes[1, 0].set_title('Cumulative Cost Curve')

        # 4. Cost vs Accuracy
        f1_scores = [r['metrics']['node_type_f1'] for r in results]
        axes[1, 1].scatter(costs, f1_scores, alpha=0.6)
        axes[1, 1].set_xlabel('Cost ($)')
        axes[1, 1].set_ylabel('Node F1 Score')
        axes[1, 1].set_title('Cost vs Accuracy')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

    def _plot_comparison_heatmap(self, results: List[Dict], output_path: Path):
        """
        Heatmap showing metrics across templates

        Args:
            results: List of results
            output_path: Output file path
        """
        # Sample top 50 for readability
        sample_results = results[:50] if len(results) > 50 else results

        data = []
        for r in sample_results:
            data.append([
                r['metrics']['node_type_f1'],
                r['metrics']['connection_f1'],
                r['metrics']['avg_parameter_accuracy']
            ])

        df = pd.DataFrame(
            data,
            columns=['Node F1', 'Connection F1', 'Parameter Accuracy'],
            index=[f"T{r['template_id']}" for r in sample_results]
        )

        fig, ax = plt.subplots(figsize=(10, max(8, len(sample_results) * 0.3)))
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
        ax.set_title(f'Evaluation Metrics Heatmap (Top {len(sample_results)} Templates)')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()

    def _plot_correlation_matrix(self, results: List[Dict], output_path: Path):
        """
        Correlation between different metrics

        Args:
            results: List of results
            output_path: Output file path
        """
        df = pd.DataFrame([
            {
                'Node F1': r['metrics']['node_type_f1'],
                'Connection F1': r['metrics']['connection_f1'],
                'Parameter Acc': r['metrics']['avg_parameter_accuracy'],
                'Cost': r['metrics']['total_cost'],
                'Prompt Tokens': r['metrics']['usage']['prompt_tokens'],
                'Completion Tokens': r['metrics']['usage']['completion_tokens']
            }
            for r in results
        ])

        corr = df.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Metric Correlation Matrix')

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi)
        plt.close()
