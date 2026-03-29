"""
Two-phase n8n INSERT helper: node catalog + authoritative schema → small LLM JSON → programmatic merge.

By default ``NodeSchemaStore`` reads repo ``node_schemas`` first (official exports with
``name`` / ``displayName`` / ``description``), then falls back to ``core_nodes_schemas`` and LangChain schemas.
"""

from __future__ import annotations

from .defaults import (
    deep_merge_parameters,
    merge_parameters_with_defaults,
    parameter_defaults_from_schema,
)
from .instruction_parse import extract_template_workflow, parse_insert_instruction
from .merge import apply_insert_splice
from .schema_store import NodeSchemaStore
from .two_phase import (
    build_neighbor_context,
    build_phase1_messages,
    build_phase2_messages,
    default_phase1_system_prompt,
    default_phase2_system_prompt,
    default_phase2_system_prompt_workflow_oracle,
    parse_phase1_json,
    parse_phase2_json,
)

__all__ = [
    "NodeSchemaStore",
    "parse_insert_instruction",
    "extract_template_workflow",
    "parameter_defaults_from_schema",
    "deep_merge_parameters",
    "merge_parameters_with_defaults",
    "apply_insert_splice",
    "build_neighbor_context",
    "build_phase1_messages",
    "build_phase2_messages",
    "default_phase1_system_prompt",
    "default_phase2_system_prompt",
    "default_phase2_system_prompt_workflow_oracle",
    "parse_phase1_json",
    "parse_phase2_json",
]
