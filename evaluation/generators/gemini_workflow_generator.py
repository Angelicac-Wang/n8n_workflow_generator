#!/usr/bin/env python3
"""
Gemini Workflow Generator

Generate n8n workflows from descriptions using Google gemini-2.5-pro.
"""

import json
import time
from datetime import datetime
from typing import Dict, Optional
from google import genai
from google.genai import types

from .prompt_builder import PromptBuilder


class GeminiWorkflowGenerator:
    """
    Generate n8n workflows using Google gemini-2.5-pro
    """

    def __init__(
        self,
        google_api_key: str,
        prompt_builder: PromptBuilder,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize Gemini workflow generator

        Args:
            google_api_key: Google API key
            prompt_builder: PromptBuilder instance
            model: Gemini model name (default: gemini-2.5-pro)
            temperature: Temperature for generation (default: 0.3)
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.client = genai.Client(api_key=google_api_key)
        self.prompt_builder = prompt_builder
        self.model_name = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate_workflow(self, description: str, template_id: str) -> Dict:
        """
        Generate workflow from description using Gemini

        Args:
            description: Workflow description
            template_id: Template ID for tracking

        Returns:
            Dictionary containing:
            - template_id: Template ID
            - llm_response: Raw LLM JSON response
            - usage: Token usage statistics
            - error: Error message (if any)
            - generated_at: ISO timestamp
        """
        # Check for empty description
        if not description or description.strip() == "":
            return {
                "template_id": template_id,
                "llm_response": None,
                "usage": None,
                "error": "Empty description",
                "generated_at": datetime.now().isoformat()
            }

        # Remove non-ASCII characters from description
        description = description.encode('ascii', errors='ignore').decode('ascii')

        # Check again if description is empty after removing non-ASCII
        if not description or description.strip() == "":
            return {
                "template_id": template_id,
                "llm_response": None,
                "usage": None,
                "error": "Empty description after removing non-ASCII characters",
                "generated_at": datetime.now().isoformat()
            }

        # Build prompt
        prompt = self.prompt_builder.build_prompt(description)
        system_message = self.prompt_builder.build_system_message()

        # Combine system message and user prompt for Gemini
        full_prompt = f"{system_message}\n\n{prompt}"

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                # Call Gemini API
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                )

                # Extract token usage
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": usage_metadata.prompt_token_count,
                    "completion_tokens": usage_metadata.candidates_token_count,
                    "total_tokens": usage_metadata.total_token_count
                }

                # Parse LLM response
                response_content = response.text
                llm_json = self._parse_json_response(response_content)

                if llm_json is None:
                    return {
                        "template_id": template_id,
                        "llm_response": None,
                        "usage": usage,
                        "error": f"Failed to parse JSON response: {response_content[:200]}",
                        "generated_at": datetime.now().isoformat()
                    }

                # Validate response structure
                if not self._validate_response(llm_json):
                    return {
                        "template_id": template_id,
                        "llm_response": llm_json,
                        "usage": usage,
                        "error": "Invalid response structure",
                        "generated_at": datetime.now().isoformat()
                    }

                return {
                    "template_id": template_id,
                    "llm_response": llm_json,
                    "usage": usage,
                    "error": None,
                    "generated_at": datetime.now().isoformat()
                }

            except Exception as e:
                error_str = str(e)

                # Check for rate limit errors
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        print(f"Rate limit hit for template {template_id}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                # For other errors, retry with exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Error for template {template_id} (attempt {attempt + 1}/{self.max_retries}): {error_str}")
                    time.sleep(wait_time)
                    continue

                # Final attempt failed
                return {
                    "template_id": template_id,
                    "llm_response": None,
                    "usage": None,
                    "error": error_str,
                    "generated_at": datetime.now().isoformat()
                }

        # Should not reach here
        return {
            "template_id": template_id,
            "llm_response": None,
            "usage": None,
            "error": "Max retries exceeded",
            "generated_at": datetime.now().isoformat()
        }

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response, handling common formatting issues

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Try direct parse first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return None

    def _validate_response(self, llm_json: Dict) -> bool:
        """
        Validate LLM response structure

        Args:
            llm_json: Parsed JSON response

        Returns:
            True if valid, False otherwise
        """
        # Must have mode field
        if 'mode' not in llm_json:
            return False

        # For create_workflow mode, must have workflowPlan
        if llm_json['mode'] == 'create_workflow':
            if 'workflowPlan' not in llm_json:
                return False

            workflow_plan = llm_json['workflowPlan']

            # Must have either 'steps' or ('nodes' and 'connections')
            has_steps = 'steps' in workflow_plan
            has_nodes_conn = 'nodes' in workflow_plan or 'connections' in workflow_plan

            if not (has_steps or has_nodes_conn):
                return False

        return True
