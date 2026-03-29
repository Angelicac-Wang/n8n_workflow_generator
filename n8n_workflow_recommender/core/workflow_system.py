#!/usr/bin/env python3
"""
核心工作流程系統

整合 MCTS 搜索、A* 路徑生成、NLU 等組件，生成候選工作流程。
"""

import json
import os
from typing import List, Dict, Set, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

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
        
        # === 新增：提取所有 mapped_nodes 資訊 ===
        print("6. Extracting all mapped_nodes from taxonomy...")
        self.all_mapped_nodes_info = self._extract_all_mapped_nodes(taxonomy_path)
        
        # === 新增：初始化 embedding model 用於 trigger node 選擇 ===
        print("7. Initializing embedding model for trigger node selection...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        print("   ✅ Embedding model loaded.")
        
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
        
        # === 新增：讓 LLM 直接選擇相關的 mapped_nodes ===
        llm_selected_nodes = self._select_mapped_nodes_with_llm(user_query, goal_description)
        llm_selected_nodes_set = set(llm_selected_nodes)
        print(f" - LLM Selected mapped_nodes: {llm_selected_nodes_set}")
        
        # STAGE 1: MCTS Search
        print("\nSTAGE 1: MCTS Taxonomy Search")
        # ✅ 根據 taxonomy 統計：92 個葉子節點，建議迭代次數為 92 * 3 = 276
        # 設定為 300 次，確保有足夠的探索空間，同時不會過度浪費時間
        matched_leaves = self.search_agent.search_with_categories(
            semantic_query=goal_description,
            function_categories=function_categories,
            extracted_keywords=keywords,
            llm_selected_nodes=llm_selected_nodes_set,  # === 新增參數 ===
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
        
        # === 新增：選擇 trigger node ===
        selected_trigger = self._select_trigger_node(user_query, initial_concrete_nodes)
        if selected_trigger:
            # 確保選中的 trigger node 在 initial_concrete_nodes 中，並且放在最前面
            if selected_trigger in initial_concrete_nodes:
                initial_concrete_nodes.remove(selected_trigger)
            initial_concrete_nodes.insert(0, selected_trigger)
            print(f"   ✅ Trigger node prioritized: {selected_trigger}")
        
        # === 新增：選擇 end node ===
        selected_end = self._select_end_node(user_query, initial_concrete_nodes)
        if selected_end:
            # 確保選中的 end node 在 initial_concrete_nodes 中
            if selected_end not in initial_concrete_nodes:
                initial_concrete_nodes.append(selected_end)
            print(f"   ✅ End node selected: {selected_end}")
        
        # STAGE 2: Workflow Composition
        print("\nSTAGE 2: Workflow Composition (A*)")
        candidates = self.composer.compose(
            matched_leaves=matched_leaves,
            initial_concrete_nodes=initial_concrete_nodes,
            user_query=user_query,
            params=extracted_params,
            selected_trigger=selected_trigger,  # 傳入選定的 trigger
            selected_end=selected_end  # 傳入選定的 end node
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
            
            # 獲取 Taxonomy 根節點（支持 Taxonomy 和 Taxonomy_n8n）
            taxonomy_root = raw_taxonomy.get("Taxonomy", raw_taxonomy.get("Taxonomy_n8n", raw_taxonomy))
            
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
                    
                    # 獲取描述（支持新舊格式）
                    # 新格式可能沒有 Description，需要從子節點或使用名稱作為描述
                    description = top_value.get("Description", top_value.get("description", clean_name))
                    
                    # 如果還是沒有描述，使用名稱作為描述
                    if not description or description == clean_name:
                        description = clean_name
                    
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
    
    def _extract_all_mapped_nodes(self, taxonomy_file_path: str) -> Dict:
        """
        從 taxonomy JSON 中提取所有 mapped_nodes 及其對應的路徑資訊。
        
        Returns:
            dict: {
                "node_to_paths": {"n8n-nodes-base.httpRequest": ["...", ...], ...},
                "all_nodes": ["n8n-nodes-base.httpRequest", ...],
                "node_descriptions": {"n8n-nodes-base.httpRequest": "...", ...}
            }
        """
        try:
            with open(taxonomy_file_path, 'r', encoding='utf-8') as f:
                raw_taxonomy = json.load(f)
        except Exception as e:
            print(f"Error loading taxonomy for mapped_nodes extraction: {e}")
            return {"node_to_paths": {}, "all_nodes": [], "node_descriptions": {}}
        
        node_to_paths = {}      # node -> [paths where it appears]
        node_descriptions = {}  # node -> description from taxonomy
        
        def traverse(node_name: str, node_content: Dict, current_path: List[str]):
            """遞迴遍歷 taxonomy 樹，提取所有 mapped_nodes"""
            if not isinstance(node_content, dict):
                return
            
            # 新格式：如果節點有 name 欄位，使用 name 作為節點名稱
            display_name = node_content.get("name", node_name)
            full_path = " -> ".join(current_path + [display_name])
            
            # 檢查此節點是否有 mapped_nodes (支持新舊格式)
            mapped_nodes = node_content.get("mapped_nodes", node_content.get("Nodes", []))
            description = node_content.get("description", node_content.get("Description", ""))
            
            for node in mapped_nodes:
                if node not in node_to_paths:
                    node_to_paths[node] = []
                    node_descriptions[node] = description
                node_to_paths[node].append(full_path)
            
            # 遞迴處理子節點
            for child_name, child_content in node_content.items():
                if child_name in ["description", "Description", "mapped_nodes", "Nodes", "example_use_cases", "name"]:
                    continue
                if isinstance(child_content, dict):
                    traverse(child_name, child_content, current_path + [display_name])
        
        # 提取 Taxonomy 根（支持 Taxonomy 和 Taxonomy_n8n）
        if "Taxonomy" in raw_taxonomy:
            taxonomy_root = raw_taxonomy["Taxonomy"]
        elif "Taxonomy_n8n" in raw_taxonomy:
            taxonomy_root = raw_taxonomy["Taxonomy_n8n"]
        else:
            taxonomy_root = raw_taxonomy
        
        # 從所有頂層分類開始遍歷
        for root_key, root_content in taxonomy_root.items():
            if isinstance(root_content, dict):
                traverse(root_key, root_content, [])
        
        all_nodes = list(node_to_paths.keys())
        
        print(f"   -> Extracted {len(all_nodes)} unique mapped_nodes from taxonomy.")
        
        return {
            "node_to_paths": node_to_paths,
            "all_nodes": all_nodes,
            "node_descriptions": node_descriptions
        }
    
    def _select_mapped_nodes_with_llm(self, user_query: str, goal_description: str) -> List[str]:
        """
        使用 LLM 從所有可用的 mapped_nodes 中選擇最相關的節點。
        
        Args:
            user_query: 原始用戶查詢
            goal_description: NLU 提取的目標描述
            
        Returns:
            list: LLM 選擇的節點名稱列表
        """
        print(" - Selecting relevant mapped_nodes with LLM...")
        
        # 準備節點資訊給 LLM
        nodes_info = self.all_mapped_nodes_info
        all_nodes = nodes_info.get("all_nodes", [])
        node_descriptions = nodes_info.get("node_descriptions", {})
        node_to_paths = nodes_info.get("node_to_paths", {})
        
        if not all_nodes:
            print("   -> Warning: No mapped_nodes available for selection.")
            return []
        
        # 如果節點太多，只取前200個給 LLM（避免 token 限制）
        if len(all_nodes) > 200:
            print(f"   -> Warning: Too many nodes ({len(all_nodes)}), selecting top 200 for LLM...")
            all_nodes = all_nodes[:200]
        
        # 構建節點描述文字供 LLM 參考
        nodes_context_parts = []
        for node in all_nodes:
            desc = node_descriptions.get(node, "No description")
            paths = node_to_paths.get(node, [])
            path_hint = paths[0] if paths else "Unknown path"
            nodes_context_parts.append(f"- `{node}`: {desc} (Path: {path_hint[:80]})")
        
        nodes_context = "\n".join(nodes_context_parts)
        
        prompt = f"""You are an expert workflow designer for n8n automation platform.

**User's Request:** "{user_query}"
**Interpreted Goal:** "{goal_description}"

**Available System Nodes:**
{nodes_context}

**Your Task:**
Based on the user's request, select the most relevant nodes that should be included in the workflow.
Consider:
1. What actions/operations does the user want to perform?
2. What data needs to be processed or displayed?
3. What flow control is needed (triggers, processing, outputs)?

**Output Format:**
Return a JSON object with a single key "selected_nodes" containing a list of node names (strings).
Only include nodes from the available list above.
Select between 3-8 nodes that are most essential for the user's workflow.

Example output: {{"selected_nodes": ["n8n-nodes-base.webhook", "n8n-nodes-base.httpRequest", "n8n-nodes-base.set"]}}
"""
        
        try:
            response = self.intent_analyzer.client.chat.completions.create(
                model="gpt-4o-mini",  # 使用較快的模型進行節點選擇
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected_nodes = result.get("selected_nodes", [])
            
            # 驗證選擇的節點確實存在於可用列表中
            valid_nodes = [n for n in selected_nodes if n in nodes_info.get("all_nodes", [])]
            
            if len(valid_nodes) != len(selected_nodes):
                invalid = set(selected_nodes) - set(valid_nodes)
                print(f"   -> Warning: LLM selected invalid nodes (ignored): {invalid}")
            
            print(f"   -> LLM selected nodes: {valid_nodes}")
            return valid_nodes
            
        except Exception as e:
            print(f"   -> LLM node selection failed: {e}. Returning empty list.")
            return []
    
    def _select_trigger_node(
        self,
        user_query: str,
        mapped_nodes: List[str]
    ) -> Optional[str]:
        """
        根據 mapping nodes 選擇最適合的 trigger node
        
        邏輯：
        1. 如果 mapping nodes 中有多個 top 10 popular trigger nodes：使用 embedding similarity 比對
        2. 如果只有一個 top 10 popular trigger node：直接使用
        3. 如果沒有 top 10 popular trigger nodes：詢問 LLM，然後模糊搜尋
        
        Args:
            user_query: 用戶查詢
            mapped_nodes: 從 MCTS 搜索得到的 mapped_nodes 列表
            
        Returns:
            選擇的 trigger node type，如果無法選擇則返回 None
        """
        print("\n🔍 Selecting Trigger Node...")
        
        # 加載 top 10 popular trigger nodes（從 package 內的數據文件）
        trigger_info_file = Path(__file__).parent.parent.parent / "data" / "top10_trigger_nodes_info.json"
        try:
            with open(trigger_info_file, 'r', encoding='utf-8') as f:
                trigger_info_data = json.load(f)
            
            # 從 top10_trigger_nodes_info.json 獲取 top 10 trigger nodes
            top_10_trigger_nodes = [
                info.get('node_type', '') 
                for info in trigger_info_data.get('top10_trigger_nodes', [])
            ]
            
            print(f"   - Top 10 popular trigger nodes: {top_10_trigger_nodes}")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load trigger info: {e}")
            top_10_trigger_nodes = []
        
        # 找出 mapped_nodes 中屬於 top 10 的 trigger nodes
        mapped_trigger_nodes = [
            node for node in mapped_nodes
            if node in top_10_trigger_nodes
        ]
        
        print(f"   - Found {len(mapped_trigger_nodes)} top 10 popular trigger nodes in mapped_nodes: {mapped_trigger_nodes}")
        
        # 情況 1：有多個 top 10 popular trigger nodes
        if len(mapped_trigger_nodes) > 1:
            print(f"   📊 Case 1: Multiple top 10 trigger nodes found. Using embedding similarity...")
            return self._select_trigger_by_embedding(user_query, mapped_trigger_nodes)
        
        # 情況 2：只有一個 top 10 popular trigger node
        elif len(mapped_trigger_nodes) == 1:
            print(f"   ✅ Case 2: Single top 10 trigger node found. Using: {mapped_trigger_nodes[0]}")
            return mapped_trigger_nodes[0]
        
        # 情況 3：沒有 top 10 popular trigger nodes
        else:
            print(f"   🤖 Case 3: No top 10 trigger nodes found. Asking LLM...")
            return self._select_trigger_with_llm(user_query)
    
    def _select_trigger_by_embedding(
        self,
        user_query: str,
        candidate_triggers: List[str]
    ) -> Optional[str]:
        """
        使用 embedding similarity 從候選 trigger nodes 中選擇最適合的
        
        Args:
            user_query: 用戶查詢
            candidate_triggers: 候選的 trigger node types
            
        Returns:
            最適合的 trigger node type
        """
        try:
            # 從 package 內的 top10_trigger_nodes_info.json 加載每個 trigger node 的 description
            trigger_descriptions = {}
            # 使用 package 內的數據文件
            trigger_info_file = Path(__file__).parent.parent.parent / "data" / "top10_trigger_nodes_info.json"
            
            # 加載 top 10 trigger nodes 信息
            trigger_info_map = {}
            if trigger_info_file.exists():
                with open(trigger_info_file, 'r', encoding='utf-8') as f:
                    trigger_info_data = json.load(f)
                    for info in trigger_info_data.get('top10_trigger_nodes', []):
                        node_type = info.get('node_type', '')
                        description = info.get('description', info.get('display_name', node_type))
                        trigger_info_map[node_type] = description
            
            # 為每個候選 trigger 獲取 description
            for trigger in candidate_triggers:
                if trigger in trigger_info_map:
                    trigger_descriptions[trigger] = trigger_info_map[trigger]
                else:
                    # 如果不在 top 10 列表中，使用 node type 名稱作為描述
                    trigger_descriptions[trigger] = trigger
            
            # 構建文本：trigger node name + description
            trigger_texts = []
            for trigger in candidate_triggers:
                desc = trigger_descriptions.get(trigger, trigger)
                combined_text = f"{trigger}: {desc}"
                trigger_texts.append(combined_text)
            
            # 計算 embedding
            query_embedding = self.embedding_model.encode(user_query, convert_to_tensor=True)
            trigger_embeddings = self.embedding_model.encode(trigger_texts, convert_to_tensor=True)
            
            # 計算相似度
            similarities = util.cos_sim(query_embedding, trigger_embeddings)[0]
            
            # 找出最高相似度的 trigger
            best_idx = similarities.argmax().item()
            best_trigger = candidate_triggers[best_idx]
            best_similarity = similarities[best_idx].item()
            
            print(f"   - Embedding similarity results:")
            for i, trigger in enumerate(candidate_triggers):
                sim = similarities[i].item()
                print(f"      {trigger}: {sim:.4f}")
            print(f"   ✅ Selected: {best_trigger} (similarity: {best_similarity:.4f})")
            
            return best_trigger
            
        except Exception as e:
            print(f"   ⚠️  Error in embedding-based selection: {e}")
            # Fallback: 返回第一個候選 trigger
            if candidate_triggers:
                print(f"   - Fallback: Using first candidate: {candidate_triggers[0]}")
                return candidate_triggers[0]
            return None
    
    def _select_trigger_with_llm(self, user_query: str) -> Optional[str]:
        """
        詢問 LLM 要用什麼 trigger node，然後在資料庫中模糊搜尋
        
        Args:
            user_query: 用戶查詢
            
        Returns:
            找到的 trigger node type，如果找不到則返回 None
        """
        try:
            # 詢問 LLM
            prompt = f"""You are an expert workflow designer for n8n automation platform.

**User's Request:** "{user_query}"

**Your Task:**
Based on the user's request, determine what type of trigger node would be most appropriate to start the workflow.

**Output Format:**
Return a JSON object with a single key "trigger_node_type" containing the n8n trigger node type (e.g., "n8n-nodes-base.gmailTrigger", "n8n-nodes-base.scheduleTrigger", etc.).

Only return the node type string, nothing else.

Example output: {{"trigger_node_type": "n8n-nodes-base.gmailTrigger"}}
"""
            
            response = self.intent_analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            llm_suggested_type = result.get("trigger_node_type", "")
            
            print(f"   - LLM suggested: {llm_suggested_type}")
            
            if not llm_suggested_type:
                print(f"   ⚠️  LLM did not return a valid trigger node type")
                return None
            
            # 在 ontology 中模糊搜尋
            # 先嘗試精確匹配
            if llm_suggested_type in self.ontology:
                print(f"   ✅ Found exact match in ontology: {llm_suggested_type}")
                return llm_suggested_type
            
            # 模糊搜尋：找出包含關鍵字的 trigger nodes
            suggested_name = llm_suggested_type.lower()
            # 提取關鍵字（例如：從 "n8n-nodes-base.gmailTrigger" 提取 "gmail" 和 "trigger"）
            keywords = [w for w in suggested_name.split('.')[-1].replace('trigger', '').split() if len(w) > 2]
            
            # 在所有 ontology 節點中搜尋 trigger nodes
            matching_triggers = []
            for node_type in self.ontology.keys():
                if 'trigger' in node_type.lower():
                    node_lower = node_type.lower()
                    # 檢查是否包含關鍵字
                    if any(kw in node_lower for kw in keywords):
                        matching_triggers.append(node_type)
            
            if matching_triggers:
                # 選擇最相似的（優先選擇包含最多關鍵字的）
                best_match = max(
                    matching_triggers,
                    key=lambda x: sum(1 for kw in keywords if kw in x.lower())
                )
                print(f"   ✅ Found fuzzy match in ontology: {best_match}")
                return best_match
            else:
                print(f"   ⚠️  No matching trigger node found in ontology")
                # 最後的 fallback：使用 manualTrigger
                if "n8n-nodes-base.manualTrigger" in self.ontology:
                    print(f"   - Fallback: Using manualTrigger")
                    return "n8n-nodes-base.manualTrigger"
                return None
                
        except Exception as e:
            print(f"   ⚠️  Error in LLM-based trigger selection: {e}")
            # Fallback: 使用 manualTrigger
            if "n8n-nodes-base.manualTrigger" in self.ontology:
                print(f"   - Fallback: Using manualTrigger")
                return "n8n-nodes-base.manualTrigger"
            return None
    
    def _select_end_node(
        self,
        user_query: str,
        mapped_nodes: List[str]
    ) -> Optional[str]:
        """
        根據 mapping nodes 選擇最適合的 end node
        
        邏輯與 trigger node 相同：
        1. 如果 mapping nodes 中有多個 top 10 popular end nodes：使用 embedding similarity 比對
        2. 如果只有一個 top 10 popular end node：直接使用
        3. 如果沒有 top 10 popular end nodes：詢問 LLM，然後模糊搜尋
        
        Args:
            user_query: 用戶查詢
            mapped_nodes: 從 MCTS 搜索得到的 mapped_nodes 列表
            
        Returns:
            選擇的 end node type，如果無法選擇則返回 None
        """
        print("\n🔍 Selecting End Node...")
        
        # 加載 top 10 popular end nodes（從 package 內的數據文件）
        end_info_file = Path(__file__).parent.parent.parent / "data" / "top10_end_nodes_info.json"
        try:
            with open(end_info_file, 'r', encoding='utf-8') as f:
                end_info_data = json.load(f)
            
            # 從 top10_end_nodes_info.json 獲取 top 10 end nodes
            top_10_end_nodes = [
                info.get('node_type', '') 
                for info in end_info_data.get('top10_end_nodes', [])
            ]
            
            print(f"   - Top 10 popular end nodes: {top_10_end_nodes}")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load end node info: {e}")
            top_10_end_nodes = []
        
        # 找出 mapped_nodes 中屬於 top 10 的 end nodes
        mapped_end_nodes = [
            node for node in mapped_nodes
            if node in top_10_end_nodes
        ]
        
        print(f"   - Found {len(mapped_end_nodes)} top 10 popular end nodes in mapped_nodes: {mapped_end_nodes}")
        
        # 情況 1：有多個 top 10 popular end nodes
        if len(mapped_end_nodes) > 1:
            print(f"   📊 Case 1: Multiple top 10 end nodes found. Using embedding similarity...")
            return self._select_end_by_embedding(user_query, mapped_end_nodes)
        
        # 情況 2：只有一個 top 10 popular end node
        elif len(mapped_end_nodes) == 1:
            print(f"   ✅ Case 2: Single top 10 end node found. Using: {mapped_end_nodes[0]}")
            return mapped_end_nodes[0]
        
        # 情況 3：沒有 top 10 popular end nodes
        else:
            print(f"   🤖 Case 3: No top 10 end nodes found. Asking LLM...")
            return self._select_end_with_llm(user_query)
    
    def _select_end_by_embedding(
        self,
        user_query: str,
        candidate_ends: List[str]
    ) -> Optional[str]:
        """
        使用 embedding similarity 從候選 end nodes 中選擇最適合的
        
        Args:
            user_query: 用戶查詢
            candidate_ends: 候選的 end node types
            
        Returns:
            最適合的 end node type
        """
        try:
            # 從 package 內的 top10_end_nodes_info.json 加載每個 end node 的 description
            end_descriptions = {}
            # 使用 package 內的數據文件
            end_info_file = Path(__file__).parent.parent.parent / "data" / "top10_end_nodes_info.json"
            
            # 加載 top 10 end nodes 信息
            end_info_map = {}
            if end_info_file.exists():
                with open(end_info_file, 'r', encoding='utf-8') as f:
                    end_info_data = json.load(f)
                    for info in end_info_data.get('top10_end_nodes', []):
                        node_type = info.get('node_type', '')
                        description = info.get('description', info.get('display_name', node_type))
                        end_info_map[node_type] = description
            
            # 為每個候選 end 獲取 description
            for end in candidate_ends:
                if end in end_info_map:
                    end_descriptions[end] = end_info_map[end]
                else:
                    # 如果不在 top 10 列表中，使用 node type 名稱作為描述
                    end_descriptions[end] = end
            
            # 構建文本：end node name + description
            end_texts = []
            for end in candidate_ends:
                desc = end_descriptions.get(end, end)
                combined_text = f"{end}: {desc}"
                end_texts.append(combined_text)
            
            # 計算 embedding
            query_embedding = self.embedding_model.encode(user_query, convert_to_tensor=True)
            end_embeddings = self.embedding_model.encode(end_texts, convert_to_tensor=True)
            
            # 計算相似度
            similarities = util.cos_sim(query_embedding, end_embeddings)[0]
            
            # 找出最高相似度的 end
            best_idx = similarities.argmax().item()
            best_end = candidate_ends[best_idx]
            best_similarity = similarities[best_idx].item()
            
            print(f"   - Embedding similarity results:")
            for i, end in enumerate(candidate_ends):
                sim = similarities[i].item()
                print(f"      {end}: {sim:.4f}")
            print(f"   ✅ Selected: {best_end} (similarity: {best_similarity:.4f})")
            
            return best_end
            
        except Exception as e:
            print(f"   ⚠️  Error in embedding-based selection: {e}")
            # Fallback: 返回第一個候選 end
            if candidate_ends:
                print(f"   - Fallback: Using first candidate: {candidate_ends[0]}")
                return candidate_ends[0]
            return None
    
    def _select_end_with_llm(self, user_query: str) -> Optional[str]:
        """
        詢問 LLM 要用什麼 end node，然後在資料庫中模糊搜尋
        
        Args:
            user_query: 用戶查詢
            
        Returns:
            找到的 end node type，如果找不到則返回 None
        """
        try:
            # 詢問 LLM
            prompt = f"""You are an expert workflow designer for n8n automation platform.

**User's Request:** "{user_query}"

**Your Task:**
Based on the user's request, determine what type of end node would be most appropriate to finish the workflow.

**Output Format:**
Return a JSON object with a single key "end_node_type" containing the n8n end node type (e.g., "n8n-nodes-base.telegram", "n8n-nodes-base.gmail", etc.).

Only return the node type string, nothing else.

Example output: {{"end_node_type": "n8n-nodes-base.telegram"}}
"""
            
            response = self.intent_analyzer.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            llm_suggested_type = result.get("end_node_type", "")
            
            print(f"   - LLM suggested: {llm_suggested_type}")
            
            if not llm_suggested_type:
                print(f"   ⚠️  LLM did not return a valid end node type")
                return None
            
            # 在 ontology 中模糊搜尋
            # 先嘗試精確匹配
            if llm_suggested_type in self.ontology:
                print(f"   ✅ Found exact match in ontology: {llm_suggested_type}")
                return llm_suggested_type
            
            # 模糊搜尋：找出包含關鍵字的 end nodes
            suggested_name = llm_suggested_type.lower()
            # 提取關鍵字
            keywords = [w for w in suggested_name.split('.')[-1].replace('trigger', '').split() if len(w) > 2]
            
            # 在所有 ontology 節點中搜尋（不一定是 trigger）
            matching_ends = []
            for node_type in self.ontology.keys():
                node_lower = node_type.lower()
                # 檢查是否包含關鍵字
                if any(kw in node_lower for kw in keywords):
                    matching_ends.append(node_type)
            
            if matching_ends:
                # 選擇最相似的（優先選擇包含最多關鍵字的）
                best_match = max(
                    matching_ends,
                    key=lambda x: sum(1 for kw in keywords if kw in x.lower())
                )
                print(f"   ✅ Found fuzzy match in ontology: {best_match}")
                return best_match
            else:
                print(f"   ⚠️  No matching end node found in ontology")
                # 最後的 fallback：使用 noOp
                if "n8n-nodes-base.noOp" in self.ontology:
                    print(f"   - Fallback: Using noOp")
                    return "n8n-nodes-base.noOp"
                return None
                
        except Exception as e:
            print(f"   ⚠️  Error in LLM-based end node selection: {e}")
            # Fallback: 使用 noOp
            if "n8n-nodes-base.noOp" in self.ontology:
                print(f"   - Fallback: Using noOp")
                return "n8n-nodes-base.noOp"
            return None

