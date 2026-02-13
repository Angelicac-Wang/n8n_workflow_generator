#!/usr/bin/env python3
"""
Description Optimizer

Optimize workflow descriptions from user queries to improve workflow generation quality.
This is the first stage of a two-stage generation process.
"""

import os
from openai import OpenAI
from typing import Dict, Optional
from datetime import datetime


class DescriptionOptimizer:
    """
    Optimize workflow descriptions from user queries using GPT-4o-mini
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize description optimizer

        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model to use for optimization (default: gpt-4o-mini for cost efficiency)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def optimize_description(self, user_query: str, template_id: str = "unknown") -> Dict:
        """
        Optimize a workflow description from user query

        Args:
            user_query: Original user query or template description
            template_id: Template ID for tracking

        Returns:
            Dictionary with optimized description, usage, and metadata
        """
        system_message = """You are an n8n workflow description optimization expert. Your task is to transform user queries or basic descriptions into structured, clear workflow descriptions that are optimal for workflow generation.

Your optimized descriptions should:
1. Be between 1000-5000 characters (optimal range for workflow generation)
2. Clearly specify the trigger (how the workflow starts)
3. Identify data sources and APIs needed
4. List all operations and processing steps
5. Define the final output or action
6. Be structured and easy to parse

Focus on clarity and completeness. Do not add information that wasn't in the original query."""

        user_message = f"""Transform this user query into an optimized workflow description:

{user_query}

Please output an optimized description that:
- Is structured and clear
- Specifies trigger, data sources, operations, and output
- Is between 1000-5000 characters
- Maintains all information from the original query
- Is ready for workflow generation

Output only the optimized description text, no JSON, no markdown formatting."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=2000  # Limit output to control cost
            )

            optimized_description = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            # Validate description length
            desc_length = len(optimized_description)
            if desc_length < 100:
                return {
                    "template_id": template_id,
                    "optimized_description": None,
                    "original_description": user_query,
                    "usage": usage,
                    "error": f"Optimized description too short ({desc_length} chars), likely an error",
                    "generated_at": datetime.now().isoformat()
                }

            return {
                "template_id": template_id,
                "optimized_description": optimized_description,
                "original_description": user_query,
                "description_length": desc_length,
                "usage": usage,
                "error": None,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "template_id": template_id,
                "optimized_description": None,
                "original_description": user_query,
                "usage": None,
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }
