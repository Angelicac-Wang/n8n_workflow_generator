#!/usr/bin/env python3
"""
Parameter Evaluator

Evaluate parameter filling accuracy using semantic similarity.
"""

import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class ParameterEvaluator:
    """
    Evaluate parameter filling using semantic similarity
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        threshold: float = 0.8
    ):
        """
        Initialize parameter evaluator

        Args:
            model_name: SentenceTransformer model name
            threshold: Similarity threshold for matching (default: 0.8)
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def evaluate_parameters(self, matching_result: Dict) -> Dict:
        """
        Evaluate parameter filling accuracy

        For each matched node pair:
        1. Extract top-level scalar parameters
        2. Compute embedding similarity
        3. Parameters match if similarity >= threshold

        Args:
            matching_result: Result from NodeMatcher

        Returns:
            Dictionary with parameter metrics
        """
        per_node_results = []
        match_ratios = []

        for match in matching_result['matches']:
            gt_node = match['gt_node']
            llm_node = match['llm_node']

            gt_params = gt_node.get('parameters', {})
            llm_params = llm_node.get('parameters', {})

            # Compare parameters
            param_result = self._compare_parameters(gt_params, llm_params)

            per_node_results.append({
                "gt_node_name": gt_node['name'],
                "llm_node_name": llm_node['name'],
                "match_ratio": param_result['match_ratio'],
                "matched_params": param_result['matched'],
                "missing_params": param_result['missing'],
                "extra_params": param_result['extra']
            })

            match_ratios.append(param_result['match_ratio'])

        # Calculate average
        avg_accuracy = np.mean(match_ratios) if match_ratios else 0.0

        return {
            "avg_parameter_accuracy": float(avg_accuracy),
            "per_node_accuracy": per_node_results
        }

    def _compare_parameters(self, gt_params: Dict, llm_params: Dict) -> Dict:
        """
        Compare parameter dictionaries using semantic similarity

        Args:
            gt_params: Ground truth parameters
            llm_params: LLM-generated parameters

        Returns:
            Dictionary with match_ratio, matched, missing, extra lists
        """
        # Extract scalar parameters
        gt_scalar = self._extract_scalar_params(gt_params)
        llm_scalar = self._extract_scalar_params(llm_params)

        matched = []
        missing = []
        extra = list(llm_scalar.keys())

        # If no GT parameters, consider it a match if LLM also has none
        if not gt_scalar:
            return {
                "match_ratio": 1.0 if not llm_scalar else 0.0,
                "matched": [],
                "missing": [],
                "extra": extra
            }

        # Match each GT parameter
        for gt_key, gt_value in gt_scalar.items():
            best_match = None
            best_similarity = 0.0

            # Try to find matching LLM parameter
            for llm_key, llm_value in llm_scalar.items():
                # Check key similarity
                key_sim = self._compute_similarity(gt_key, llm_key)

                if key_sim >= 0.7:  # Key must be similar
                    # Check value similarity
                    value_sim = self._compute_similarity(str(gt_value), str(llm_value))

                    if value_sim > best_similarity:
                        best_similarity = value_sim
                        best_match = llm_key

            if best_match and best_similarity >= self.threshold:
                matched.append(gt_key)
                if best_match in extra:
                    extra.remove(best_match)
            else:
                missing.append(gt_key)

        # Calculate match ratio
        match_ratio = len(matched) / len(gt_scalar) if gt_scalar else 0.0

        return {
            "match_ratio": match_ratio,
            "matched": matched,
            "missing": missing,
            "extra": extra
        }

    def _extract_scalar_params(self, params: Dict, prefix: str = "") -> Dict:
        """
        Extract top-level scalar parameters

        Args:
            params: Parameter dictionary
            prefix: Prefix for nested parameters

        Returns:
            Dictionary of scalar parameters
        """
        scalar_params = {}

        for key, value in params.items():
            full_key = f"{prefix}{key}" if prefix else key

            # Only include scalars and simple values
            if isinstance(value, (str, int, float, bool)):
                scalar_params[full_key] = value
            elif isinstance(value, dict):
                # Recurse one level for nested dicts
                nested = self._extract_scalar_params(value, prefix=f"{full_key}.")
                scalar_params.update(nested)
            # Skip lists and complex objects

        return scalar_params

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0-1)
        """
        # Exact match shortcut
        if text1 == text2:
            return 1.0

        # Compute embeddings
        embeddings = self.model.encode([text1, text2])

        # Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return float(similarity)
