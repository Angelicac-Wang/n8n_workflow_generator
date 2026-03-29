#!/usr/bin/env python3
"""
Vincent 0118 Workflow Generator

Wraps the 0118_vincent orchestration flow and converts outputs to the
evaluation-compatible LLM response format.
"""

import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Vincent0118Generator:
    """
    Generate workflows using the 0118_vincent orchestration pipeline.
    """

    def __init__(
        self,
        openai_api_key: str,
        taxonomy_path: str,
        mf_model_dir: str,
        vincent_dir: str,
        kg_path: Optional[str] = None,
        mentor_enabled: bool = True,
        mcts_query_source: str = "original",
        intent_keywords_source: str = "llm",
        max_retries: int = 2,
        retry_delay: float = 2.0,
    ):
        self.openai_api_key = openai_api_key
        self.taxonomy_path = Path(taxonomy_path)
        self.mf_model_dir = Path(mf_model_dir)
        self.vincent_dir = Path(vincent_dir)
        self.kg_path = Path(kg_path) if kg_path else (self.vincent_dir / "n8n_domain_knowledge.json")
        self.mentor_enabled = mentor_enabled
        self.mcts_query_source = mcts_query_source
        self.intent_keywords_source = intent_keywords_source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._usage = self._empty_usage()

        self._orchestrator = self._init_orchestrator()

    def generate_workflow(self, description: str, template_id: str) -> Dict:
        """
        Run the 0118_vincent pipeline and return evaluation-ready format.
        """
        if not description or not description.strip():
            return self._error_result(template_id, "Empty description")

        for attempt in range(self.max_retries):
            try:
                # Reset usage collector for this generation run.
                self._usage = self._empty_usage()
                result = self._orchestrator.process_user_request(description)
                if not result:
                    return self._error_result(template_id, "No workflow generated")

                llm_response = self._build_llm_response(result)

                return {
                    "template_id": template_id,
                    "llm_response": llm_response,
                    "usage": dict(self._usage),
                    "error": None,
                    "generated_at": datetime.now().isoformat(),
                    "raw_result": result,
                }
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                return self._error_result(
                    template_id,
                    f"Vincent generation failed: {exc}"
                )

    def _build_llm_response(self, result: Dict) -> Dict:
        path_nodes = result.get("path", [])
        params_list = result.get("params", [])

        steps = self._build_steps(path_nodes, params_list)

        return {
            "mode": "create_workflow",
            "workflowPlan": {
                "steps": steps
            }
        }

    def _build_steps(self, path_nodes: List[str], params_list: List[Dict]) -> List[Dict]:
        params_by_index = {}
        params_by_node = {}

        for item in params_list:
            if not isinstance(item, dict):
                continue
            if "step" in item:
                params_by_index[item["step"]] = item.get("inputs", {})
            if "node" in item:
                params_by_node[item["node"]] = item.get("inputs", {})

        steps = []
        for idx, node_name in enumerate(path_nodes):
            step_params = params_by_index.get(idx) or params_by_node.get(node_name) or {}

            steps.append({
                "id": f"step_{idx + 1}",
                "type": node_name,
                "params": step_params,
            })

        return steps

    def _init_orchestrator(self):
        vincent_module = self._load_module(
            "NCU_Conversational_UI_n8n_KG_Completion_Vincent_20260115_v5",
            self.vincent_dir / "NCU_Conversational_UI_n8n_KG_Completion_Vincent_20260115_v5.py.txt",
        )
        reranker_module = self._load_module(
            "Predict_chain_Daniel_Angelica_20260115",
            self.vincent_dir / "Predict_chain_Daniel_Angelica_20260115.py.txt",
        )

        # Ensure dependencies are importable by name inside orchestrator module.
        sys.modules[vincent_module.__name__] = vincent_module
        sys.modules[reranker_module.__name__] = reranker_module

        # Override KG loader default path to local repo file.
        original_loader = vincent_module.get_domain_knowledge
        kg_path = str(self.kg_path)

        def _patched_get_domain_knowledge(json_path: str = kg_path):
            return original_loader(json_path=json_path)

        vincent_module.get_domain_knowledge = _patched_get_domain_knowledge

        # Patch keyword extraction to expose LLM keywords for reranker intent coverage.
        hybrid_cls = getattr(vincent_module, "HybridWorkflowSystem", None)
        if hybrid_cls is not None:
            original_keyword_extractor = hybrid_cls._extract_keywords_with_llm

            def _patched_extract_keywords(self, user_query, analysis):
                keywords = original_keyword_extractor(self, user_query, analysis)
                self._last_llm_keywords = list(keywords) if keywords else []
                return keywords

            hybrid_cls._extract_keywords_with_llm = _patched_extract_keywords

            # Patch MCTS semantic query to optionally use original user query (not goal summary).
            search_agent_cls = getattr(vincent_module, "TaxonomySearchAgent", None)
            if search_agent_cls is not None and not hasattr(search_agent_cls, "_patched_semantic_query"):
                original_search = search_agent_cls.search_with_categories

                def _patched_search(self, semantic_query, *args, **kwargs):
                    override_query = getattr(self, "_override_semantic_query", None)
                    if override_query:
                        semantic_query = override_query
                    return original_search(self, semantic_query, *args, **kwargs)

                search_agent_cls.search_with_categories = _patched_search
                search_agent_cls._patched_semantic_query = True

            original_generate = hybrid_cls.generate_workflow

            def _patched_generate_workflow(self, user_query):
                if self._mcts_query_source == "original":
                    self.search_agent._override_semantic_query = user_query
                else:
                    self.search_agent._override_semantic_query = None
                try:
                    return original_generate(self, user_query)
                finally:
                    self.search_agent._override_semantic_query = None

            hybrid_cls.generate_workflow = _patched_generate_workflow
            hybrid_cls._mcts_query_source = self.mcts_query_source

        # Patch missing _fill_params on HybridWorkflowSystem (needed for mentor-corrected paths).
        if hybrid_cls is not None and not hasattr(hybrid_cls, "_fill_params"):
            def _fill_params(self, path, extracted_params):
                filled_steps = []
                for step_index, node_name in enumerate(path):
                    ontology_reqs = self.ontology.get(node_name, {}).get("required_params", [])
                    node_inputs = {}
                    if not ontology_reqs:
                        filled_steps.append({
                            "step": step_index,
                            "node": node_name,
                            "inputs": {}
                        })
                        continue

                    for req_key in ontology_reqs:
                        if req_key in extracted_params:
                            node_inputs[req_key] = extracted_params[req_key]
                        else:
                            found_fuzzy = False
                            for ext_key, ext_val in extracted_params.items():
                                if (ext_key.lower() in req_key.lower()) or (req_key.lower() in ext_key.lower()):
                                    node_inputs[req_key] = ext_val
                                    found_fuzzy = True
                                    break
                            if not found_fuzzy:
                                node_inputs[req_key] = "<NEEDS_VALUE>"

                    filled_steps.append({
                        "step": step_index,
                        "node": node_name,
                        "inputs": node_inputs
                    })

                return filled_steps

            hybrid_cls._fill_params = _fill_params

        # Orchestrator patch is applied after orchestrator_module is loaded.

        orchestrator_module = self._load_module(
            "Main_Orchestrator_20260115_v3",
            self.vincent_dir / "Main_Orchestrator_20260115_v3.py.txt",
        )

        # Patch orchestrator intent keywords to use desired source.
        orch_cls = getattr(orchestrator_module, "WorkflowOrchestrator", None)
        if orch_cls is not None:
            original_intent_extractor = orch_cls._extract_intent_keywords

            def _patched_extract_intent_keywords(self, user_query):
                if self._intent_keywords_source == "llm":
                    keywords = getattr(self.generator, "_last_llm_keywords", None)
                    if keywords:
                        return keywords
                return original_intent_extractor(self, user_query)

            orch_cls._extract_intent_keywords = _patched_extract_intent_keywords
            orch_cls._intent_keywords_source = self.intent_keywords_source

        orchestrator_cls = getattr(orchestrator_module, "WorkflowOrchestrator", None)
        if orchestrator_cls is None:
            raise RuntimeError("WorkflowOrchestrator class not found in 0118_vincent module")

        if not self.mentor_enabled:
            # Disable mentor intervention by forcing direct top-ranked selection.
            def _no_mentor_validation(self, user_query, candidates, structural_requirements):
                if not candidates:
                    return None
                chosen = candidates[0]
                chosen['gpt_validation'] = {
                    'selected_by_gpt': False,
                    'reason': 'Mentor intervention disabled; using top-ranked candidate.'
                }
                return chosen

            orchestrator_cls._gpt_final_validation = _no_mentor_validation

        orchestrator = orchestrator_cls(
            openai_key=self.openai_api_key,
            taxonomy_path=str(self.taxonomy_path),
            mf_model_dir=str(self.mf_model_dir),
        )

        self._patch_openai_usage(orchestrator)
        return orchestrator

    def _patch_openai_usage(self, orchestrator):
        """
        Patch OpenAI client calls to collect token usage.
        """
        generator = getattr(orchestrator, "generator", None)
        if generator is None:
            return
        client = getattr(generator, "client", None)
        if client is None:
            return

        chat = getattr(client, "chat", None)
        completions = getattr(chat, "completions", None) if chat else None
        create_fn = getattr(completions, "create", None) if completions else None
        if create_fn is None:
            return

        if getattr(completions.create, "_usage_patched", False):
            return

        def _wrapped_create(*args, **kwargs):
            response = create_fn(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage:
                self._usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                self._usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                self._usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
            return response

        _wrapped_create._usage_patched = True
        completions.create = _wrapped_create

    def _load_module(self, module_name: str, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"Module file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            # Fallback for non-.py suffixes (e.g., .py.txt)
            from importlib.machinery import SourceFileLoader

            loader = SourceFileLoader(module_name, str(file_path))
            spec = importlib.util.spec_from_loader(module_name, loader)

        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec: {module_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _empty_usage() -> Dict:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _error_result(template_id: str, message: str) -> Dict:
        return {
            "template_id": template_id,
            "llm_response": None,
            "usage": Vincent0118Generator._empty_usage(),
            "error": message,
            "generated_at": datetime.now().isoformat(),
        }

