#!/usr/bin/env python3
"""
Result Saver

Save evaluation results to disk.
"""

import json
from pathlib import Path
from typing import Dict, List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from n8n_workflow_recommender.utils.file_loader import save_json


class ResultSaver:
    """
    Save evaluation results to disk
    """

    def __init__(self, output_dir: str):
        """
        Initialize result saver

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)

        # Create subdirectories
        self.llm_workflows_dir = self.output_dir / "llm_generated_workflows"
        self.eval_results_dir = self.output_dir / "evaluation_results"
        self.visualizations_dir = self.output_dir / "visualizations"

        # Create directories
        self.llm_workflows_dir.mkdir(parents=True, exist_ok=True)
        self.eval_results_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)

    def save_generated_workflow(self, result: Dict):
        """
        Save LLM-generated workflow result

        Args:
            result: Result dictionary from LLMWorkflowGenerator
        """
        template_id = result['template_id']
        output_file = self.llm_workflows_dir / f"generated_{template_id}.json"

        # Save compactly to keep disk usage low (these files can be large).
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, separators=(",", ":"))

    def workflow_exists(self, template_id: str) -> bool:
        """
        Check if workflow already generated

        Args:
            template_id: Template ID

        Returns:
            True if workflow file exists
        """
        output_file = self.llm_workflows_dir / f"generated_{template_id}.json"
        return output_file.exists()

    def load_generated_workflow(self, template_id: str) -> Dict:
        """
        Load previously generated workflow

        Args:
            template_id: Template ID

        Returns:
            Result dictionary
        """
        output_file = self.llm_workflows_dir / f"generated_{template_id}.json"

        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_evaluation_results(
        self,
        detailed_results: List[Dict],
        summary_stats: Dict,
        cost_report: Dict
    ):
        """
        Save evaluation results (JSON and CSV)

        Args:
            detailed_results: List of per-template results
            summary_stats: Summary statistics
            cost_report: Cost report
        """
        # Save detailed results (JSON)
        detailed_json_path = self.eval_results_dir / "detailed_per_template.json"
        save_json(detailed_results, str(detailed_json_path))

        # Save summary statistics
        summary_path = self.eval_results_dir / "summary_statistics.json"
        save_json(summary_stats, str(summary_path))

        # Save cost report
        cost_path = self.eval_results_dir / "cost_report.json"
        save_json(cost_report, str(cost_path))

        # Save detailed results (CSV)
        self._save_results_as_csv(detailed_results)

    def _save_results_as_csv(self, results: List[Dict]):
        """
        Save results as CSV file

        Args:
            results: List of result dictionaries
        """
        import pandas as pd

        # Extract data for CSV
        rows = []
        for r in results:
            m = r.get('metrics')
            if not m:
                continue
            usage = m.get('usage') or {}
            rows.append({
                "template_id": r['template_id'],
                "template_name": r.get('template_name', ''),
                "llm_output_valid_json": m.get('llm_output_valid_json', ''),
                "node_type_precision": m.get('node_type_precision', ''),
                "node_type_recall": m.get('node_type_recall', ''),
                "node_type_f1": m.get('node_type_f1', ''),
                "connection_precision": m.get('connection_precision', ''),
                "connection_recall": m.get('connection_recall', ''),
                "connection_f1": m.get('connection_f1', ''),
                "parameter_accuracy": m.get('avg_parameter_accuracy', ''),
                "total_cost_usd": m.get('total_cost', ''),
                "prompt_tokens": usage.get('prompt_tokens', ''),
                "completion_tokens": usage.get('completion_tokens', ''),
                "error": r.get('error', ''),
            })

        # Create DataFrame and save
        df = pd.DataFrame(rows)
        csv_path = self.eval_results_dir / "detailed_per_template.csv"
        df.to_csv(csv_path, index=False)

    def get_output_dirs(self) -> Dict[str, Path]:
        """
        Get all output directory paths

        Returns:
            Dictionary mapping directory names to paths
        """
        return {
            "llm_workflows": self.llm_workflows_dir,
            "evaluation_results": self.eval_results_dir,
            "visualizations": self.visualizations_dir
        }
