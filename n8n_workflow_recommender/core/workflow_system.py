#!/usr/bin/env python3
"""
核心工作流程系統

整合 MCTS 搜索、A* 路徑生成、NLU 等組件，生成候選工作流程。
"""

import json
from typing import List, Dict, Set, Optional
from pathlib import Path

from ..search.mcts_search_agent import TaxonomySearchAgent, MCTSNode
from ..generation.workflow_composer import DomainKnowledgeGraph, ModuleAwareWorkflowComposer
from ..nlu.intent_analyzer import IntentAnalyzer
from ..nlu.keyword_extractor import KeywordExtractor


class HybridWorkflowSystem:
    """
    混合工作流程系統
    
    整合所有組件，從用戶查詢生成候選工作流程。
    """
    
    def __init__(
        self,
        triples: List[tuple],
        ontology: Dict,
        taxonomy_path: str,
        openai_api_key: str
    ):
        """
        初始化系統
        
        Args:
            triples: 知識圖三元組列表
            ontology: Ontology 字典
            taxonomy_path: MCTS taxonomy 檔案路徑
            openai_api_key: OpenAI API 密鑰
        """
        print("\n=== Initializing Hybrid Workflow System ===")
        
        # 初始化組件
        print("1. Initializing Taxonomy Search Agent (MCTS)...")
        self.search_agent = TaxonomySearchAgent(taxonomy_path)
        
        print("2. Initializing Domain Knowledge Graph (A*)...")
        aux_keywords = ['通知', '發送', 'Email', 'SMS', '記錄', '日誌', '提醒', '確認']
        self.domain_graph = DomainKnowledgeGraph(triples, ontology, auxiliary_keywords=aux_keywords)
        
        print("3. Initializing Workflow Composer...")
        self.composer = ModuleAwareWorkflowComposer(
            self.domain_graph,
            self.search_agent,
            ontology
        )
        
        print("4. Building Function Categories from Taxonomy...")
        # 從 taxonomy 動態構建 function_categories（像原本的程式碼）
        self.function_categories = self._build_categories_from_taxonomy(taxonomy_path)
        print(f"   - Loaded {len(self.function_categories)} function categories.")
        
        print("5. Initializing NLU Components...")
        # 傳入 ontology 和 function_categories 以提供更好的上下文
        self.intent_analyzer = IntentAnalyzer(
            openai_api_key,
            ontology=ontology,
            function_categories=self.function_categories
        )
        self.keyword_extractor = KeywordExtractor()
        
        self.ontology = ontology
        
        print("✅ All components initialized successfully.")
    
    def generate_workflow(self, user_query: str) -> List[Dict]:
        """
        生成工作流程候選
        
        Args:
            user_query: 用戶查詢字符串
        
        Returns:
            candidates: 候選工作流程列表
        """
        print("\n" + "=" * 80)
        print(f"Received User Query: '{user_query}'")
        print("=" * 80)
        
        # STAGE 0: NLU Analysis
        print("\nSTAGE 0: NLU Analysis")
        analysis = self.intent_analyzer.analyze(user_query)
        goal_description = analysis.get('goal_description', user_query)
        extracted_params = analysis.get('parameters', {})
        function_categories = analysis.get('function_categories', [])
        
        # 提取關鍵字（使用 LLM，像原本的程式碼）
        keywords = self.intent_analyzer.extract_keywords(user_query, analysis)
        # 也添加一些技術術語作為補充
        tech_terms = self.keyword_extractor.extract_technical_terms(user_query)
        keywords.update(tech_terms)
        
        print(f" - Extracted Keywords: {keywords}")
        
        # STAGE 1: MCTS Search
        print("\nSTAGE 1: MCTS Taxonomy Search")
        # ✅ 根據 taxonomy 統計：92 個葉子節點，建議迭代次數為 92 * 3 = 276
        # 設定為 300 次，確保有足夠的探索空間，同時不會過度浪費時間
        matched_leaves = self.search_agent.search_with_categories(
            semantic_query=goal_description,
            function_categories=function_categories,
            extracted_keywords=keywords,
            iterations=300,  # 優化：92 個葉子節點，300 次迭代足夠（約 3.3x 覆蓋率）
            top_n=5
        )
        
        if not matched_leaves:
            print(" - No matches found in taxonomy. Trying keyword search...")
            matched_leaves = self.search_agent.search_by_keywords(keywords)
        
        if not matched_leaves:
            print("⚠️  Warning: No taxonomy matches found.")
            # 嘗試使用更寬鬆的搜索：直接從知識圖中查找相關節點
            print(" - Trying fallback: searching knowledge graph directly...")
            # 從關鍵字中提取可能的節點類型
            fallback_nodes = []
            for kw in keywords:
                # 嘗試在 ontology 中查找包含關鍵字的節點
                for node_type in self.ontology.keys():
                    if kw.lower() in node_type.lower():
                        fallback_nodes.append(node_type)
                        if len(fallback_nodes) >= 5:
                            break
                if len(fallback_nodes) >= 5:
                    break
            
            if fallback_nodes:
                print(f" - Found {len(fallback_nodes)} fallback nodes from ontology")
                # 創建一個簡單的候選，使用 fallback 節點
                initial_concrete_nodes = list(set(fallback_nodes))
                # 創建一個簡單的候選工作流程
                print("\nSTAGE 2: Workflow Composition (A*) - Fallback Mode")
                candidates = self.composer.compose(
                    matched_leaves=[],  # 空的 matched_leaves
                    initial_concrete_nodes=initial_concrete_nodes,
                    user_query=user_query,
                    params=extracted_params
                )
                if candidates:
                    print(f" - Generated {len(candidates)} workflow candidates (fallback mode)")
                    return candidates
            else:
                print("⚠️  No fallback nodes found. Trying to create minimal workflow...")
                # 最後的備用方案：創建一個最小的工作流程
                # 使用常見的觸發節點和處理節點
                minimal_nodes = []
                for node_type in ['n8n-nodes-base.manualTrigger', 'n8n-nodes-base.set', 'n8n-nodes-base.noOp']:
                    if node_type in self.ontology:
                        minimal_nodes.append(node_type)
                
                if minimal_nodes:
                    print(f" - Creating minimal workflow with {len(minimal_nodes)} nodes")
                    candidates = self.composer.compose(
                        matched_leaves=[],
                        initial_concrete_nodes=minimal_nodes,
                        user_query=user_query,
                        params=extracted_params
                    )
                    if candidates:
                        return candidates
            
            print("⚠️  All fallback strategies failed. Returning empty candidates.")
            return []
        
        print(f"\n   ✅ Final Selected Taxonomy Nodes ({len(matched_leaves)} nodes):")
        for i, leaf in enumerate(matched_leaves):
            path_str = leaf.get('path_str', 'N/A')
            semantic = leaf.get('semantic_score', 0.0)
            category = leaf.get('category_score', 0.0)
            avg_reward = leaf.get('avg_reward', 0.0)
            mapped_nodes = leaf.get('mapped_nodes', [])
            print(f"      {i+1}. {path_str}")
            print(f"         - Semantic: {semantic:.4f} | Category: {category:.4f} | Reward: {avg_reward:.4f}")
            print(f"         - Mapped Nodes: {len(mapped_nodes)} nodes")
        
        print(f"\n   📦 Extracting Concrete Node Types from Selected Taxonomy Nodes...")
        # 提取所有 mapped_nodes
        initial_concrete_nodes = []
        node_source_map = {}  # 記錄每個節點來自哪個 taxonomy node
        
        for leaf in matched_leaves:
            mapped_nodes = leaf.get('mapped_nodes', [])
            path_str = leaf.get('path_str', 'N/A')
            if mapped_nodes:
                initial_concrete_nodes.extend(mapped_nodes)
                # 記錄來源
                for node in mapped_nodes:
                    if node not in node_source_map:
                        node_source_map[node] = []
                    node_source_map[node].append(path_str)
        
        # 去重並保持順序
        unique_nodes = []
        seen = set()
        for node in initial_concrete_nodes:
            if node not in seen:
                unique_nodes.append(node)
                seen.add(node)
        
        print(f"   - Extracted {len(unique_nodes)} unique concrete node types:")
        # 按來源分組顯示
        for i, node in enumerate(unique_nodes[:50]):  # 顯示前50個
            sources = node_source_map.get(node, [])
            source_preview = sources[0][:50] + "..." if sources and len(sources[0]) > 50 else (sources[0] if sources else "Unknown")
            print(f"      {i+1}. {node} (from: {source_preview})")
        
        if len(unique_nodes) > 50:
            print(f"      ... and {len(unique_nodes) - 50} more nodes")
        
        initial_concrete_nodes = unique_nodes
        
        # 如果沒有 mapped_nodes，嘗試從關鍵字推斷
        if not initial_concrete_nodes:
            print(" - No mapped_nodes found, trying to infer from keywords...")
            for kw in keywords:
                # 在 ontology 中查找
                for node_type in self.ontology.keys():
                    if kw.lower() in node_type.lower() or any(kw.lower() in str(v).lower() for v in self.ontology[node_type].values()):
                        initial_concrete_nodes.append(node_type)
                        if len(initial_concrete_nodes) >= 3:
                            break
                if len(initial_concrete_nodes) >= 3:
                    break
        
        initial_concrete_nodes = list(set(initial_concrete_nodes))
        print(f" - Extracted {len(initial_concrete_nodes)} concrete node types")
        
        if not initial_concrete_nodes:
            print("⚠️  Warning: No concrete nodes extracted. Cannot generate workflow.")
            return []
        
        # STAGE 2: Workflow Composition
        print("\nSTAGE 2: Workflow Composition (A*)")
        candidates = self.composer.compose(
            matched_leaves=matched_leaves,
            initial_concrete_nodes=initial_concrete_nodes,
            user_query=user_query,
            params=extracted_params
        )
        
        if not candidates:
            print("⚠️  Warning: Failed to generate workflow candidates.")
            return []
        
        print(f" - Generated {len(candidates)} workflow candidates")
        
        return candidates
    
    def _build_categories_from_taxonomy(self, taxonomy_file_path: str) -> Dict:
        """
        從 taxonomy 的第一層（頂層分類）構建 function categories
        
        使用第一層分類更容易獲得 category 分數，因為：
        - 第一層分類更寬泛，更容易匹配
        - 例如："1 Commerce & Revenue Operations", "2 Customer Engagement & Marketing"
        """
        import json
        categories = {}
        try:
            with open(taxonomy_file_path, 'r', encoding='utf-8') as f:
                raw_taxonomy = json.load(f)
            
            # 獲取 Taxonomy 根節點
            taxonomy_root = raw_taxonomy.get("Taxonomy", raw_taxonomy)
            
            # 直接提取第一層（頂層分類）
            for top_key, top_value in taxonomy_root.items():
                if isinstance(top_value, dict):
                    # 提取乾淨的名稱（去掉數字前綴）
                    # 例如："1 Commerce & Revenue Operations" -> "Commerce & Revenue Operations"
                    # 例如："2 Customer Engagement & Marketing" -> "Customer Engagement & Marketing"
                    parts = top_key.split(' ', 1)  # 分割數字和名稱
                    if len(parts) == 2 and parts[0].isdigit():
                        clean_name = parts[1]  # 提取名稱部分
                    else:
                        clean_name = top_key  # 如果格式不對，使用原始名稱
                    
                    # 獲取描述
                    description = top_value.get("Description", top_value.get("description", clean_name))
                    
                    # 添加到 categories
                    categories[clean_name] = description
            
            if not categories:
                print("   ⚠️  Warning: Could not build categories. Taxonomy structure might be unexpected.")
                return {"Default": "Default category"}  # 備援
            
            # 調試：輸出前幾個提取的 categories
            print(f"   - Sample categories extracted (first 10):")
            for i, (cat_name, cat_desc) in enumerate(list(categories.items())[:10]):
                print(f"      {i+1}. {cat_name}: {cat_desc[:60]}...")
            
            return categories
            
        except Exception as e:
            print(f"   ⚠️  Error loading taxonomy for categories: {e}")
            return {
                "Error": "Could not load taxonomy categories dynamically."
            }

