#!/usr/bin/env python3
"""
Prompt Builder

Build prompts for LLM workflow generation from templates.
"""

from pathlib import Path
from typing import Dict


class PromptBuilder:
    """
    Build prompts for LLM workflow generation
    """

    def __init__(self, prompt_template_path: str):
        """
        Initialize prompt builder

        Args:
            prompt_template_path: Path to prompt template file
        """
        self.prompt_template_path = Path(prompt_template_path)

        if not self.prompt_template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {self.prompt_template_path}")

        # Load prompt template
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def build_prompt(self, description: str) -> str:
        """
        Build prompt by replacing placeholder with description

        Args:
            description: Workflow description from template

        Returns:
            Complete prompt string
        """
        # Replace {{ $json.output }} with actual description
        prompt = self.prompt_template.replace("{{ $json.output }}", description)

        return prompt

    def build_system_message(self) -> str:
        """
        Build system message for OpenAI API

        Returns:
            System message string
        """
        return "You are an n8n workflow generation assistant. You must respond with valid JSON only."
