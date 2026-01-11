#!/usr/bin/env python3
"""
LLM Workflow Generator

Generate n8n workflows from descriptions using GPT-4o.
"""

import json
import time
from datetime import datetime
from typing import Dict, Optional
from openai import OpenAI
import openai

from .prompt_builder import PromptBuilder


class LLMWorkflowGenerator:
    """
    Generate n8n workflows using GPT-4o
    """

    def __init__(
        self,
        openai_api_key: str,
        prompt_builder: PromptBuilder,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize LLM workflow generator

        Args:
            openai_api_key: OpenAI API key
            prompt_builder: PromptBuilder instance
            model: OpenAI model name (default: gpt-4o)
            temperature: Temperature for generation (default: 0.3)
            max_retries: Maximum number of retries for API calls
            retry_delay: Initial retry delay in seconds (exponential backoff)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.prompt_builder = prompt_builder
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate_workflow(self, description: str, template_id: str) -> Dict:
        """
        Generate workflow from description using GPT-4o

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

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API with proper encoding handling
                # Note: OpenAI SDK should handle Unicode correctly, but we ensure
                # the strings are properly formatted
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": str(system_message)},
                        {"role": "user", "content": str(prompt)}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )

                # Extract token usage
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

                # Parse LLM response
                response_content = response.choices[0].message.content
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
                if not self._validate_llm_response(llm_json):
                    return {
                        "template_id": template_id,
                        "llm_response": llm_json,
                        "usage": usage,
                        "error": "Invalid response structure (missing 'mode' field)",
                        "generated_at": datetime.now().isoformat()
                    }

                # Success
                return {
                    "template_id": template_id,
                    "llm_response": llm_json,
                    "usage": usage,
                    "error": None,
                    "generated_at": datetime.now().isoformat()
                }

            except openai.RateLimitError as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"  Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    # Safely get error message
                    error_msg = self._safe_error_message(e)
                    return {
                        "template_id": template_id,
                        "llm_response": None,
                        "usage": None,
                        "error": f"Rate limit exceeded after {self.max_retries} retries: {error_msg}",
                        "generated_at": datetime.now().isoformat()
                    }

            except Exception as e:
                # Safely get error message
                error_msg = self._safe_error_message(e)

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"  API error, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    return {
                        "template_id": template_id,
                        "llm_response": None,
                        "usage": None,
                        "error": f"API error after {self.max_retries} retries: {error_msg}",
                        "generated_at": datetime.now().isoformat()
                    }

        # Should not reach here
        return {
            "template_id": template_id,
            "llm_response": None,
            "usage": None,
            "error": "Unknown error",
            "generated_at": datetime.now().isoformat()
        }

    def _parse_json_response(self, response_content: str) -> Optional[Dict]:
        """
        Parse JSON response, handling potential errors

        Args:
            response_content: Raw response content

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            # Remove markdown code blocks
            if response_content.strip().startswith('```'):
                lines = response_content.split('\n')
                response_content = '\n'.join([
                    line for line in lines
                    if not line.strip().startswith('```')
                ])

            # Try parsing again
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Try to extract JSON pattern
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return None
                return None

    def _validate_llm_response(self, response: Dict) -> bool:
        """
        Validate LLM response has required structure

        Args:
            response: Parsed LLM response

        Returns:
            True if valid, False otherwise
        """
        # Must have 'mode' field
        if 'mode' not in response:
            return False

        # If mode is 'create_workflow', must have 'workflowPlan'
        if response['mode'] == 'create_workflow' and 'workflowPlan' not in response:
            return False

        return True

    def _safe_error_message(self, error: Exception) -> str:
        """
        Safely extract error message

        Args:
            error: Exception object

        Returns:
            String representation of error
        """
        try:
            error_type = error.__class__.__name__
            error_msg = str(error)

            # Truncate if too long
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."

            return f"{error_type}: {error_msg}"
        except:
            return "UnknownError"
