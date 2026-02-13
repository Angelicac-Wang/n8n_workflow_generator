#!/usr/bin/env python3
"""
Prompt Builder

Build prompts for LLM workflow generation from templates.
"""

from pathlib import Path
from typing import Dict, Optional


class PromptBuilder:
    """
    Build prompts for LLM workflow generation
    """

    def __init__(self, prompt_template_path: str, use_improved: bool = True):
        """
        Initialize prompt builder

        Args:
            prompt_template_path: Path to prompt template file
            use_improved: Whether to use improved prompt with CoT reasoning (default: True)
        """
        self.prompt_template_path = Path(prompt_template_path)
        self.use_improved = use_improved

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
        if self.use_improved:
            return "You are an expert n8n workflow generation assistant. You must respond with valid JSON only, including reasoning fields when generating workflows."
        else:
            return "You are an n8n workflow generation assistant. You must respond with valid JSON only."

    @classmethod
    def create_with_improved_prompt(cls, base_prompt_path: str) -> 'PromptBuilder':
        """
        Factory method to create PromptBuilder with improved prompt if available

        Args:
            base_prompt_path: Base path to prompt template (e.g., 'evaluation/config/workflow_generation_prompt.txt')

        Returns:
            PromptBuilder instance using improved prompt if available, otherwise base prompt
        """
        base_path = Path(base_prompt_path)
        improved_path = base_path.parent / f"{base_path.stem}_improved{base_path.suffix}"

        if improved_path.exists():
            return cls(str(improved_path), use_improved=True)
        else:
            return cls(base_prompt_path, use_improved=False)
