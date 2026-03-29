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
        prompt_builder: Optional[PromptBuilder],
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        use_prompt_id: bool = False,
        prompt_id: Optional[str] = None,
        prompt_version: Optional[str] = None,
        openai_project: Optional[str] = None,
        openai_organization: Optional[str] = None,
        max_output_tokens: Optional[int] = None
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
        self.client = OpenAI(
            api_key=openai_api_key,
            project=openai_project,
            organization=openai_organization
        )
        self.prompt_builder = prompt_builder
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_prompt_id = use_prompt_id
        self.prompt_id = prompt_id
        self.prompt_version = prompt_version
        self.max_output_tokens = max_output_tokens
        self.raw_response_max_chars = 5000

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

        if self.use_prompt_id and not self.prompt_id:
            return {
                "template_id": template_id,
                "llm_response": None,
                "usage": None,
                "error": "Prompt ID is required when use_prompt_id is enabled",
                "generated_at": datetime.now().isoformat()
            }

        # Build prompt only for chat-completions path
        prompt = None
        system_message = None
        if not self.use_prompt_id:
            if not self.prompt_builder:
                return {
                    "template_id": template_id,
                    "llm_response": None,
                    "usage": None,
                    "error": "Prompt builder is required when use_prompt_id is disabled",
                    "generated_at": datetime.now().isoformat(),
                }
            prompt = self.prompt_builder.build_prompt(description)
            system_message = self.prompt_builder.build_system_message()

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API with proper encoding handling
                # Note: OpenAI SDK should handle Unicode correctly, but we ensure
                # the strings are properly formatted
                if self.use_prompt_id:
                    response = self.client.responses.create(
                        model=self.model,
                        prompt={
                            "id": self.prompt_id,
                            "version": self.prompt_version
                        },
                        input=description,
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": str(system_message)},
                            {"role": "user", "content": str(prompt)}
                        ],
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                        max_tokens=self.max_output_tokens
                    )

                # Extract token usage
                usage = self._extract_usage(response)

                # Parse LLM response
                response_content = self._extract_response_text(response)
                llm_json = self._parse_json_response(response_content)

                if llm_json is None:
                    return {
                        "template_id": template_id,
                        "llm_response": None,
                        "usage": usage,
                        "raw_response": self._truncate_raw_response(response_content),
                        "error": f"Failed to parse JSON response: {response_content[:200]}",
                        "generated_at": datetime.now().isoformat()
                    }

                # Validate response structure
                if not self._validate_llm_response(llm_json):
                    return {
                        "template_id": template_id,
                        "llm_response": llm_json,
                        "usage": usage,
                        "raw_response": self._truncate_raw_response(response_content),
                        "error": "Invalid response structure (missing 'mode' field)",
                        "generated_at": datetime.now().isoformat()
                    }

                # Success
                return {
                    "template_id": template_id,
                    "llm_response": llm_json,
                    "usage": usage,
                    "raw_response": None,
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
                # Try to extract the largest plausible JSON object substring.
                # This helps when the model adds leading/trailing text or we got cut off.
                start = response_content.find("{")
                if start == -1:
                    return None

                # Heuristic 1: take from first '{' to last '}' and parse.
                end = response_content.rfind("}")
                if end > start:
                    candidate = response_content[start:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        pass

                # Heuristic 2: walk backwards trying earlier closing braces.
                # (Useful when output is truncated but contains at least one complete object.)
                for end in range(len(response_content) - 1, start, -1):
                    if response_content[end] != "}":
                        continue
                    candidate = response_content[start:end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue

                # Heuristic 3: regex fallback (greedy between braces)
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return None

                return None

    def _extract_response_text(self, response) -> str:
        """
        Extract text content from OpenAI response objects.

        Args:
            response: OpenAI response object

        Returns:
            Extracted text content (may be empty)
        """
        # Responses API typically exposes output_text
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        # Chat completions path
        choices = getattr(response, "choices", None)
        if choices:
            try:
                return choices[0].message.content or ""
            except Exception:
                pass

        # Fallback: walk output blocks for Responses API
        output = getattr(response, "output", None)
        if output:
            parts = []
            for item in output:
                content = getattr(item, "content", None)
                if not content:
                    continue
                for block in content:
                    block_text = getattr(block, "text", None)
                    if block_text:
                        parts.append(block_text)
            return "\n".join(parts).strip()

        return ""

    def _truncate_raw_response(self, text: str) -> str:
        if text is None:
            return ""
        if len(text) <= self.raw_response_max_chars:
            return text
        return text[: self.raw_response_max_chars] + "...[truncated]"

    def _extract_usage(self, response) -> Optional[Dict]:
        """
        Extract usage from OpenAI response objects.

        Args:
            response: OpenAI response object

        Returns:
            Usage dict or None
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return None

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if prompt_tokens is None:
            prompt_tokens = getattr(usage, "input_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(usage, "output_tokens", None)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }

    def _validate_llm_response(self, response: Dict) -> bool:
        """
        Validate LLM response has required structure

        Args:
            response: Parsed LLM response

        Returns:
            True if valid, False otherwise
        """
        # Accept evaluation prompt format
        if 'mode' in response:
            if response['mode'] == 'create_workflow' and 'workflowPlan' not in response:
                return False
            return True

        # Accept raw n8n workflow JSON format
        if 'nodes' in response and 'connections' in response:
            return True

        return False

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
