#!/usr/bin/env python3
"""
Workflow Description Generator

Generate natural language descriptions from workflow templates using AI.
"""

import os
from openai import OpenAI
from typing import Dict, Optional
from datetime import datetime


class WorkflowDescriptionGenerator:
    """
    Generate workflow descriptions from templates using GPT-4o
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize description generator

        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: Model to use for generation
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate_description(self, template_data: Dict) -> Dict:
        """
        Generate a natural language description from workflow template

        Args:
            template_data: Full template dictionary

        Returns:
            Dictionary with generated description, usage, and metadata
        """
        template_id = template_data.get('metadata', {}).get('id') or template_data.get('id', 'unknown')

        # Extract workflow structure
        workflow = template_data.get('workflow', {}).get('workflow', {})
        nodes = workflow.get('nodes', [])
        connections = workflow.get('connections', {})

        # Build a structured representation of the workflow
        workflow_summary = self._build_workflow_summary(nodes, connections)

        # Create prompt for AI
        system_message = """You are an expert at understanding n8n automation workflows.
        Your task is to generate a clear, concise natural language description of what the workflow does.

        Focus on:
        1. What triggers the workflow
        2. What data sources or APIs it uses
        3. What operations it performs
        4. What the final output or action is

        Write in a clear, professional tone."""

        user_message = f"""Based on this n8n workflow structure, write a natural language description:

        Nodes ({len(nodes)} total):
        {workflow_summary}

        Describe what this workflow does."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=300
            )

            generated_description = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

            return {
                "template_id": template_id,
                "generated_description": generated_description,
                "usage": usage,
                "error": None,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "template_id": template_id,
                "generated_description": None,
                "usage": None,
                "error": str(e),
                "generated_at": datetime.now().isoformat()
            }

    def _build_workflow_summary(self, nodes: list, connections: Dict) -> str:
        """
        Build a concise summary of workflow structure

        Args:
            nodes: List of node dictionaries
            connections: Connection dictionary

        Returns:
            String summary of workflow
        """
        summary_parts = []

        # Filter out stickyNote nodes
        valid_nodes = [n for n in nodes if 'stickynote' not in n.get('type', '').lower()]

        for i, node in enumerate(valid_nodes[:10], 1):  # Limit to first 10 nodes
            node_type = node.get('type', 'unknown')
            node_name = node.get('name', 'Unknown')

            # Extract readable type
            if '.' in node_type:
                readable_type = node_type.split('.')[-1]
            else:
                readable_type = node_type

            summary_parts.append(f"{i}. {node_name} ({readable_type})")

        if len(valid_nodes) > 10:
            summary_parts.append(f"... and {len(valid_nodes) - 10} more nodes")

        return "\n".join(summary_parts)
