#!/usr/bin/env python3
"""
Workflow Normalizer

Normalize both ground truth templates and LLM outputs to a common format.
"""

from typing import Dict, List


class WorkflowNormalizer:
    """
    Normalize workflows to common format for comparison
    """

    # Common node type aliases
    NODE_TYPE_ALIASES = {
        "httpRequest": ["n8n-nodes-base.httpRequest", "httpRequest", "http"],
        "webhook": ["n8n-nodes-base.webhook", "webhook"],
        "set": ["n8n-nodes-base.set", "set"],
        "code": ["n8n-nodes-base.code", "code", "function"],
        "if": ["n8n-nodes-base.if", "if"],
        "merge": ["n8n-nodes-base.merge", "merge"],
        "splitInBatches": ["n8n-nodes-base.splitInBatches", "splitInBatches", "loop"],
    }

    def normalize_ground_truth(self, template_data: Dict) -> Dict:
        """
        Extract and normalize workflow from ground truth template

        Note: Filters out stickyNote nodes as they are comments/annotations

        Args:
            template_data: Full template dictionary

        Returns:
            Normalized workflow with nodes and connections
        """
        workflow = template_data.get('workflow', {}).get('workflow', {})

        # Extract nodes (skip stickyNote nodes)
        nodes = []
        for node in workflow.get('nodes', []):
            full_type = node.get('type', 'unknown')

            # Skip stickyNote nodes (they are comments/annotations)
            if 'stickynote' in full_type.lower():
                continue

            normalized_type = self._normalize_node_type(full_type)

            nodes.append({
                "id": node.get('id', ''),
                "name": node.get('name', ''),
                "type": normalized_type,
                "full_type": full_type,
                "parameters": node.get('parameters', {})
            })

        # Extract connections (skip connections involving stickyNote nodes)
        # Build set of valid node names (excluding stickyNote)
        valid_node_names = {node['name'] for node in nodes}

        connections = []
        conn_dict = workflow.get('connections', {})

        for source_name, targets in conn_dict.items():
            # Skip if source is a stickyNote (not in valid nodes)
            if source_name not in valid_node_names:
                continue

            # Handle main connections
            main_outputs = targets.get('main', [])

            for output_idx, target_list in enumerate(main_outputs):
                if not target_list:
                    continue

                for target in target_list:
                    target_name = target.get('node', '')

                    # Skip if target is a stickyNote (not in valid nodes)
                    if target_name not in valid_node_names:
                        continue

                    connections.append({
                        "from": source_name,
                        "to": target_name,
                        "from_output": output_idx,
                        "to_input": target.get('index', 0)
                    })

        return {
            "nodes": nodes,
            "connections": connections
        }

    def normalize_llm_output(self, llm_response: Dict) -> Dict:
        """
        Convert LLM response to normalized format

        Handles both "steps" and "nodes+connections" formats

        Args:
            llm_response: LLM response dictionary

        Returns:
            Normalized workflow with nodes and connections
        """
        # Check if mode is create_workflow
        if llm_response.get('mode') != 'create_workflow':
            return {"nodes": [], "connections": []}

        workflow_plan = llm_response.get('workflowPlan', {})

        # Case 1: Steps format (linear workflow)
        if 'steps' in workflow_plan and workflow_plan['steps']:
            return self._normalize_from_steps(workflow_plan['steps'])

        # Case 2: Nodes + connections format
        elif 'nodes' in workflow_plan and workflow_plan['nodes']:
            return self._normalize_from_nodes_connections(
                workflow_plan['nodes'],
                workflow_plan.get('connections', [])
            )

        # Case 3: Empty or invalid
        else:
            return {"nodes": [], "connections": []}

    def _normalize_from_steps(self, steps: List[Dict]) -> Dict:
        """
        Convert linear steps to nodes + connections

        Args:
            steps: List of step dictionaries

        Returns:
            Normalized workflow
        """
        nodes = []
        connections = []

        for i, step in enumerate(steps):
            node_id = step.get('id', f"step_{i}")
            node_type = step.get('type', 'unknown')
            normalized_type = self._normalize_node_type(node_type)

            nodes.append({
                "id": node_id,
                "name": node_id,  # Use ID as name for steps
                "type": normalized_type,
                "full_type": node_type,
                "parameters": step.get('params', {})
            })

            # Create linear connections
            if i > 0:
                prev_id = steps[i - 1].get('id', f"step_{i-1}")
                connections.append({
                    "from": prev_id,
                    "to": node_id,
                    "from_output": 0,
                    "to_input": 0
                })

        return {"nodes": nodes, "connections": connections}

    def _normalize_from_nodes_connections(
        self,
        nodes: List[Dict],
        connections: List[Dict]
    ) -> Dict:
        """
        Convert nodes+connections format to normalized format

        Args:
            nodes: List of node dictionaries
            connections: List of connection dictionaries

        Returns:
            Normalized workflow
        """
        normalized_nodes = []
        # Create ID to name mapping for connection resolution
        id_to_name = {}

        for node in nodes:
            node_id = node.get('id', node.get('name', 'unknown'))
            node_name = node.get('label', node_id)
            node_type = node.get('nodeType', 'unknown')
            normalized_type = self._normalize_node_type(node_type)

            # Map ID to name for connection resolution
            id_to_name[str(node_id)] = node_name

            normalized_nodes.append({
                "id": node_id,
                "name": node_name,
                "type": normalized_type,
                "full_type": node_type,
                "parameters": node.get('params', {})
            })

        # Normalize connections - resolve IDs to names
        normalized_connections = []
        for conn in connections:
            from_ref = str(conn.get('from', ''))
            to_ref = str(conn.get('to', ''))

            # Resolve to names (fallback to original if not found in mapping)
            from_name = id_to_name.get(from_ref, from_ref)
            to_name = id_to_name.get(to_ref, to_ref)

            normalized_connections.append({
                "from": from_name,
                "to": to_name,
                "from_output": conn.get('outputIndex', 0),
                "to_input": conn.get('inputIndex', 0)
            })

        return {"nodes": normalized_nodes, "connections": normalized_connections}

    def _normalize_node_type(self, node_type: str) -> str:
        """
        Normalize node type to base name (lowercase for consistency)

        Examples:
        - "n8n-nodes-base.httpRequest" -> "httprequest"
        - "@n8n/n8n-nodes-langchain.openAi" -> "openai"
        - "@n8n/n8n-nodes-langchain.agent" -> "agent"
        - "httpRequest" -> "httprequest"

        Args:
            node_type: Full or partial node type

        Returns:
            Normalized base type (lowercase)
        """
        # Strip namespace prefix
        if '.' in node_type:
            base_type = node_type.split('.')[-1]
        else:
            base_type = node_type

        # Convert to lowercase for consistency
        base_type_lower = base_type.lower()

        # Check aliases
        for canonical, aliases in self.NODE_TYPE_ALIASES.items():
            if base_type_lower in [a.lower() for a in aliases]:
                return canonical

        return base_type_lower
