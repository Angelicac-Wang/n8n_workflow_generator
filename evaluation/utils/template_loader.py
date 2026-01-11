#!/usr/bin/env python3
"""
Template Loader

Load and parse n8n workflow templates from testing_data directory.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Add parent directory to path to import from n8n_workflow_recommender
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from n8n_workflow_recommender.utils.file_loader import load_json


class TemplateLoader:
    """
    Load n8n workflow templates from testing_data directory
    """

    def __init__(self, templates_dir: str):
        """
        Initialize template loader

        Args:
            templates_dir: Path to templates directory
        """
        self.templates_dir = Path(templates_dir)

        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")

    def load_all_templates(self, exclude_index: bool = True) -> List[Dict]:
        """
        Load all template files from directory

        Args:
            exclude_index: If True, exclude template_index.json

        Returns:
            List of template dictionaries
        """
        templates = []

        # Get all JSON files
        json_files = sorted(self.templates_dir.glob("*.json"))

        for json_file in json_files:
            # Skip index file if requested
            if exclude_index and json_file.name == "template_index.json":
                continue

            try:
                template_data = load_json(str(json_file))
                templates.append(template_data)
            except Exception as e:
                print(f"Warning: Failed to load {json_file.name}: {e}")
                continue

        return templates

    def load_template_by_id(self, template_id: int) -> Optional[Dict]:
        """
        Load a specific template by ID

        Args:
            template_id: Template ID

        Returns:
            Template dictionary or None if not found
        """
        # Try to find template file
        pattern = f"template_{template_id}_*.json"
        matching_files = list(self.templates_dir.glob(pattern))

        if not matching_files:
            return None

        try:
            return load_json(str(matching_files[0]))
        except Exception as e:
            print(f"Error loading template {template_id}: {e}")
            return None

    def extract_description(self, template: Dict) -> str:
        """
        Extract description from template

        Uses workflow.description directly

        Args:
            template: Template dictionary

        Returns:
            Description string (or empty string if not found)
        """
        # Use workflow.description directly
        description = template.get('workflow', {}).get('description', '')

        return description.strip() if description else ''

    def get_template_count(self, exclude_index: bool = True) -> int:
        """
        Get count of template files

        Args:
            exclude_index: If True, exclude template_index.json

        Returns:
            Number of templates
        """
        json_files = list(self.templates_dir.glob("*.json"))

        if exclude_index:
            json_files = [f for f in json_files if f.name != "template_index.json"]

        return len(json_files)
