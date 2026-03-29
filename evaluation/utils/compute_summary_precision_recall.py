#!/usr/bin/env python3
"""
Compute aggregate precision/recall statistics from detailed_per_template.json.

This script is useful when you already have evaluation outputs on disk and want
to (re)generate a summary file that includes precision/recall (in addition to F1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Dict, List, Optional

from n8n_workflow_recommender.utils.file_loader import save_json


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _micro_f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, (p + r))


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    mean_v = sum(values) / len(values)
    return {
        "mean": float(mean_v),
        "median": float(median(values)),
        "std": float(pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _load_detailed(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def compute_summary_from_detailed(detailed: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_templates = len(detailed)
    valid = [r for r in detailed if r.get("metrics")]

    if not valid:
        return {
            "total_templates": total_templates,
            "successful_evaluations": 0,
            "failed_evaluations": total_templates,
            "node_accuracy": {
                "precision": _stats([]),
                "recall": _stats([]),
                "f1": _stats([]),
                "micro": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                },
            },
            "connection_accuracy": {
                "precision": _stats([]),
                "recall": _stats([]),
                "f1": _stats([]),
                "micro": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "correct_connections": 0,
                    "gt_connection_count": 0,
                    "llm_connection_count": 0,
                },
            },
            "parameter_accuracy": {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "error": "No valid results to aggregate",
        }

    node_precisions: List[float] = []
    node_recalls: List[float] = []
    node_f1s: List[float] = []

    conn_precisions: List[float] = []
    conn_recalls: List[float] = []
    conn_f1s: List[float] = []

    param_accs: List[float] = []

    sum_tp = 0
    sum_fp = 0
    sum_fn = 0

    sum_correct_conn = 0
    sum_gt_conn = 0
    sum_llm_conn = 0

    for r in valid:
        m = r["metrics"]

        # Node type metrics
        node_precisions.append(float(m.get("node_type_precision", 0.0)))
        node_recalls.append(float(m.get("node_type_recall", 0.0)))
        node_f1s.append(float(m.get("node_type_f1", 0.0)))

        sum_tp += int(m.get("tp", 0))
        sum_fp += int(m.get("fp", 0))
        sum_fn += int(m.get("fn", 0))

        # Connection metrics
        conn_precisions.append(float(m.get("connection_precision", 0.0)))
        conn_recalls.append(float(m.get("connection_recall", 0.0)))
        conn_f1s.append(float(m.get("connection_f1", 0.0)))

        sum_correct_conn += int(m.get("correct_connections", 0))
        sum_gt_conn += int(m.get("gt_connection_count", 0))
        sum_llm_conn += int(m.get("llm_connection_count", 0))

        # Parameter metrics (already aggregate-style)
        param_accs.append(float(m.get("avg_parameter_accuracy", 0.0)))

    micro_node_p = _safe_div(sum_tp, (sum_tp + sum_fp))
    micro_node_r = _safe_div(sum_tp, (sum_tp + sum_fn))
    micro_node_f1 = _micro_f1(micro_node_p, micro_node_r)

    micro_conn_p = _safe_div(sum_correct_conn, sum_llm_conn)
    micro_conn_r = _safe_div(sum_correct_conn, sum_gt_conn)
    micro_conn_f1 = _micro_f1(micro_conn_p, micro_conn_r)

    node_p_stats = _stats(node_precisions)
    node_r_stats = _stats(node_recalls)
    node_f1_stats = _stats(node_f1s)

    conn_p_stats = _stats(conn_precisions)
    conn_r_stats = _stats(conn_recalls)
    conn_f1_stats = _stats(conn_f1s)

    param_stats = _stats(param_accs)

    return {
        "total_templates": total_templates,
        "successful_evaluations": len(valid),
        "failed_evaluations": total_templates - len(valid),
        "node_accuracy": {
            "precision": node_p_stats,
            "recall": node_r_stats,
            "f1": node_f1_stats,
            "micro": {
                "precision": float(micro_node_p),
                "recall": float(micro_node_r),
                "f1": float(micro_node_f1),
                "tp": int(sum_tp),
                "fp": int(sum_fp),
                "fn": int(sum_fn),
            },
        },
        "connection_accuracy": {
            "precision": conn_p_stats,
            "recall": conn_r_stats,
            "f1": conn_f1_stats,
            "micro": {
                "precision": float(micro_conn_p),
                "recall": float(micro_conn_r),
                "f1": float(micro_conn_f1),
                "correct_connections": int(sum_correct_conn),
                "gt_connection_count": int(sum_gt_conn),
                "llm_connection_count": int(sum_llm_conn),
            },
        },
        "parameter_accuracy": {
            "mean": param_stats["mean"],
            "median": param_stats["median"],
            "std": param_stats["std"],
            "min": param_stats["min"],
            "max": param_stats["max"],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a summary JSON with precision/recall from detailed_per_template.json"
    )
    parser.add_argument(
        "--detailed",
        type=str,
        required=True,
        help="Path to detailed_per_template.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: same dir as detailed, name: summary_statistics_precision_recall.json)",
    )

    args = parser.parse_args(argv)

    detailed_path = Path(args.detailed)
    if not detailed_path.exists():
        raise FileNotFoundError(f"File not found: {detailed_path}")

    out_path = Path(args.output) if args.output else (
        detailed_path.parent / "summary_statistics_precision_recall.json"
    )

    detailed = _load_detailed(detailed_path)
    summary = compute_summary_from_detailed(detailed)
    save_json(summary, str(out_path))

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

