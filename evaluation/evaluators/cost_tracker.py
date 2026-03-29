#!/usr/bin/env python3
"""
Cost Tracker

Track token usage and costs for LLM API calls.
"""

from typing import Dict, List


class CostTracker:
    """
    Track and calculate costs for LLM API calls
    """

    # GPT-4o pricing (Standard tier, as of 2026-01)
    # Source: https://platform.openai.com/docs/pricing
    PRICING = {
        "gpt-4o": {
            "input": 2.50 / 1_000_000,   # $2.50 per 1M tokens
            "output": 10.0 / 1_000_000   # $10.00 per 1M tokens
        }
    }

    def calculate_cost(self, usage: Dict, model: str = "gpt-4o") -> Dict:
        """
        Calculate cost from token usage

        Args:
            usage: Usage dictionary with prompt_tokens, completion_tokens, total_tokens
            model: Model name (default: gpt-4o)

        Returns:
            Dictionary with input_cost, output_cost, total_cost, currency
        """
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o"])

        input_cost = usage['prompt_tokens'] * pricing['input']
        output_cost = usage['completion_tokens'] * pricing['output']
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }

    def aggregate_costs(self, per_template_results: List[Dict]) -> Dict:
        """
        Aggregate costs across all templates

        Args:
            per_template_results: List of result dictionaries with usage info

        Returns:
            Dictionary with aggregate cost statistics
        """
        total_input = 0
        total_output = 0
        total_cost = 0.0
        valid_count = 0

        for result in per_template_results:
            # Try to get usage from metrics first, then fallback to direct usage key
            usage = result.get('metrics', {}).get('usage') or result.get('usage')

            if usage:
                total_input += usage['prompt_tokens']
                total_output += usage['completion_tokens']

                cost_info = self.calculate_cost(usage)
                total_cost += cost_info['total_cost']
                valid_count += 1

        total_tokens = total_input + total_output
        avg_cost = total_cost / valid_count if valid_count > 0 else 0.0

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_template": avg_cost,
            "templates_with_cost": valid_count,
            "currency": "USD"
        }
