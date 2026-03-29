#!/usr/bin/env python3
"""
MCTS 搜索代理

適配 n8n 的 taxonomy，使用 MCTS 算法進行搜索。
"""

import json
import math
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import torch

# 避免 huggingface tokenizers 的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MCTSNode:
    """MCTS 節點"""
    def __init__(self, name, parent=None, taxonomy_data=None):
        self.name = name
        self.parent = parent
        self.taxonomy_data = taxonomy_data
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
    
    def is_fully_expanded(self):
        if 'children' not in self.taxonomy_data or not self.taxonomy_data['children']:
            return True
        return len(self.children) == len(self.taxonomy_data['children'])
    
    def select_best_child(self, c_param=1.414):
        """UCT 公式實現"""
        best_score, best_child = -float('inf'), None
        for child in self.children:
            exploit = child.total_reward / (child.visits + 1e-6)
            explore = math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + c_param * explore
            if score > best_score:
                best_score, best_child = score, child
        return best_child


class TaxonomySearchAgent:
    """
    Taxonomy 搜索代理（MCTS）
    
    使用 MCTS 算法在 n8n taxonomy 中搜索相關節點。
    """
    
    def __init__(self, taxonomy_path: str, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        初始化搜索代理
        
        Args:
            taxonomy_path: MCTS 格式的 taxonomy JSON 檔案路徑
            model_name: SentenceTransformer 模型名稱
        """
        print("PHASE 1A: Initializing Taxonomy Search Agent (MCTS)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=device)
        # ✅ 優化：緩存 use_case embeddings，避免重複計算
        self.use_case_embedding_cache = {}  # {use_case_text: embedding}
        self._prepare_data(taxonomy_path)
        print(" - MCTS Taxonomy data prepared.")
        
        # === 新增：儲存 LLM 選擇的目標節點（供 R_category 使用）===
        self.llm_selected_nodes = set()
    
    def _is_leaf_node(self, node_content: Dict) -> bool:
        """檢查是否為葉子節點（包含 Nodes 或 mapped_nodes）"""
        if not isinstance(node_content, dict):
            return False
        # 如果有 Nodes 或 mapped_nodes，且不是空的，則是葉子節點
        # 或者如果有 name 和 mapped_nodes（新格式），也是葉子節點
        has_nodes = bool(node_content.get("Nodes") or node_content.get("mapped_nodes"))
        # 新格式：如果包含 name 和 mapped_nodes，是葉子節點
        has_name_and_nodes = bool(node_content.get("name") and node_content.get("mapped_nodes"))
        # 或者明確標記為 is_leaf
        is_leaf_flag = node_content.get('is_leaf', False)
        return has_nodes or has_name_and_nodes or is_leaf_flag
    
    def _prepare_data(self, taxonomy_path: str):
        """準備 taxonomy 數據"""
        taxonomy_path = Path(taxonomy_path)
        
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy 檔案不存在: {taxonomy_path}")
        
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            raw_taxonomy = json.load(f)
        
        # 保存原始 taxonomy 數據（用於 category 搜索）
        # 提取 Taxonomy 根（支持 Taxonomy 和 Taxonomy_n8n）
        if "Taxonomy" in raw_taxonomy:
            self.raw_taxonomy_for_search = raw_taxonomy["Taxonomy"]
        elif "Taxonomy_n8n" in raw_taxonomy:
            self.raw_taxonomy_for_search = raw_taxonomy["Taxonomy_n8n"]
        else:
            self.raw_taxonomy_for_search = raw_taxonomy
        
        self.node_database = []
        texts_to_encode = []
        
        def traverse(node_name: str, node_content: Dict, current_path: List[str]):
            """遞歸遍歷 taxonomy（為所有節點生成 embedding）"""
            # 新格式：如果節點有 name 欄位，使用 name 作為節點名稱
            display_name = node_content.get("name", node_name)
            full_path_list = current_path + [display_name]
            full_path_str = " -> ".join(full_path_list)
            
            # 支持新舊格式：Description (舊) 或 description (新)
            description = node_content.get("Description", node_content.get("description", ""))
            # 支持新舊格式：Nodes (舊) 或 mapped_nodes (新)
            mapped_nodes = node_content.get("Nodes", node_content.get("mapped_nodes", []))
            
            # 核心描述用於語義匹配（統一格式，不包含 Nodes，因為語義匹配主要看路徑和描述）
            combined_text = f"{full_path_str}: {description}"
            
            # 為所有節點（包括中間節點）生成 embedding
            texts_to_encode.append(combined_text)
            
            if self._is_leaf_node(node_content):
                # 保存 example_use_cases（如果有的話，舊格式才有）
                example_use_cases = node_content.get("example_use_cases", [])
                self.node_database.append({
                    "description": description,
                    "path_str": full_path_str,
                    "mapped_nodes": mapped_nodes,
                    "example_use_cases": example_use_cases,  # 保存以支持關鍵字匹配
                    "combined_text": combined_text
                })
            else:
                # 遞歸處理子節點（子節點是直接作為字典的鍵，而不是在 "children" 鍵下）
                for child_name, child_content in node_content.items():
                    # 跳過特殊鍵（支持新舊格式）
                    if child_name in ["Description", "description", "Nodes", "mapped_nodes", "example_use_cases", "name"]:
                        continue
                    # 只處理字典類型的子節點
                    if isinstance(child_content, dict):
                        traverse(child_name, child_content, full_path_list)
        
        # 提取 Taxonomy 根（支持 Taxonomy 和 Taxonomy_n8n）
        if "Taxonomy" in raw_taxonomy:
            taxonomy_root = raw_taxonomy["Taxonomy"]
        elif "Taxonomy_n8n" in raw_taxonomy:
            taxonomy_root = raw_taxonomy["Taxonomy_n8n"]
        else:
            taxonomy_root = raw_taxonomy
        
        # 遍歷所有頂層分類
        for root_key, root_content in taxonomy_root.items():
            traverse(root_key, root_content, [])
        
        # 生成 embeddings
        print(f"   📊 編碼 {len(texts_to_encode)} 個節點描述...")
        embeddings = self.model.encode(texts_to_encode, convert_to_tensor=True, show_progress_bar=True)
        self.text_embedding_map = {text: emb for text, emb in zip(texts_to_encode, embeddings)}
        
        # 構建 MCTS 樹
        def build_mcts_tree(node_name: str, node_content: Dict, current_path: List[str]) -> Dict:
            """構建 MCTS 樹結構"""
            # 新格式：如果節點有 name 欄位，使用 name 作為節點名稱
            display_name = node_content.get("name", node_name)
            full_path_list = current_path + [display_name]
            full_path_str = " -> ".join(full_path_list)
            is_leaf = self._is_leaf_node(node_content)
            # 支持新舊格式：Description (舊) 或 description (新)
            description = node_content.get("Description", node_content.get("description", ""))
            combined_text = f"{full_path_str}: {description}"
            
            # 查找 embedding（必須與 traverse 中生成的格式一致）
            embedding = self.text_embedding_map.get(combined_text)
            if embedding is None:
                # 調試：輸出找不到 embedding 的節點
                print(f"   - Warning: No embedding for {full_path_str}")
            
            processed_node = {
                'embedding': embedding,
                'description': description,
                'mapped_nodes': node_content.get("Nodes", node_content.get("mapped_nodes", [])),
                'children': {},
                'is_leaf': is_leaf
            }
            
            if not is_leaf:
                # 子節點是直接作為字典的鍵，而不是在 "children" 鍵下
                for child_name, child_content in node_content.items():
                    # 跳過特殊鍵（支持新舊格式）
                    if child_name in ["Description", "description", "Nodes", "mapped_nodes", "example_use_cases", "name"]:
                        continue
                    # 只處理字典類型的子節點
                    if isinstance(child_content, dict):
                        processed_node['children'][child_name] = build_mcts_tree(
                            child_name, child_content, full_path_list
                        )
            
            return processed_node
        
        # ✅ 構建虛擬根節點 "Taxonomy"，將所有頂層分類作為其子節點（第二層）
        # 提取 Taxonomy 根（支持 Taxonomy 和 Taxonomy_n8n）
        if "Taxonomy" in raw_taxonomy:
            taxonomy_root_for_tree = raw_taxonomy["Taxonomy"]
        elif "Taxonomy_n8n" in raw_taxonomy:
            taxonomy_root_for_tree = raw_taxonomy["Taxonomy_n8n"]
        else:
            taxonomy_root_for_tree = raw_taxonomy
        
        # 創建虛擬根節點 "Taxonomy"，將所有頂層分類（1-9）作為其子節點
        virtual_root_children = {}
        for root_key, root_content in taxonomy_root_for_tree.items():
            if isinstance(root_content, dict):
                virtual_root_children[root_key] = build_mcts_tree(root_key, root_content, [])
        
        # 為虛擬根節點創建一個描述和 embedding
        virtual_root_description = "n8n Taxonomy: Complete workflow automation node categories"
        virtual_root_combined_text = f"Taxonomy: {virtual_root_description}"
        
        # 為虛擬根節點生成 embedding（如果還沒有的話）
        if virtual_root_combined_text not in self.text_embedding_map:
            virtual_root_embedding = self.model.encode(virtual_root_combined_text, convert_to_tensor=True)
            self.text_embedding_map[virtual_root_combined_text] = virtual_root_embedding
        
        # 構建虛擬根節點
        self.mcts_taxonomy_tree = {
            "Taxonomy": {
                'embedding': self.text_embedding_map.get(virtual_root_combined_text),
                'description': virtual_root_description,
                'mapped_nodes': [],
                'children': virtual_root_children,
                'is_leaf': False
            }
        }
    
    def _filter_keywords(self, keywords: Set[str], function_categories: List[str]) -> Set[str]:
        """
        過濾關鍵字：移除類別名稱，只保留技術關鍵字
        
        Args:
            keywords: 原始關鍵字集合
            function_categories: 功能類別列表
        
        Returns:
            filtered_keywords: 過濾後的關鍵字集合
        """
        if not keywords:
            return set()
        
        # 將類別名稱轉換為小寫集合，用於過濾
        category_lower = {cat.lower() for cat in function_categories} if function_categories else set()
        
        # 常見的無用關鍵字（類別名稱、停用詞等）
        stop_words = {
            'productivity', 'collaboration', 'automation', 'intelligence',
            'ai, ml & automation intelligence', 'productivity & collaboration',
            'ml', 'ai', '&', 'and', 'or', 'the', 'a', 'an'
        }
        
        filtered = set()
        for kw in keywords:
            kw_lower = kw.lower().strip()
            
            # 跳過空字串
            if not kw_lower:
                continue
            
            # 跳過類別名稱
            if kw_lower in category_lower:
                continue
            
            # 跳過停用詞
            if kw_lower in stop_words:
                continue
            
            # 跳過太短的關鍵字（少於 2 個字符）
            if len(kw_lower) < 2:
                continue
            
            # 保留技術關鍵字
            filtered.add(kw)
        
        return filtered if filtered else keywords  # 如果過濾後為空，返回原始關鍵字
    
    def _calculate_node_match_reward(self, db_node: Dict, llm_selected_nodes: Set[str]) -> float:
        """
        [NEW] 基於 LLM 直接選擇的 mapped_nodes 計算獎勵。
        
        核心邏輯：
        1. 取得當前 taxonomy 葉節點的 mapped_nodes
        2. 計算與 LLM 選擇的節點的交集比例
        
        Args:
            db_node: node_database 中的一個條目，包含 'mapped_nodes' 欄位
            llm_selected_nodes: LLM 選擇的節點集合
            
        Returns:
            float: 匹配分數 (0.0 ~ 1.0)
        """
        if not llm_selected_nodes:
            return 0.0
        
        # 取得此 taxonomy 節點對應的具體系統節點
        node_mapped = set(db_node.get("mapped_nodes", []))
        
        if not node_mapped:
            return 0.0
        
        # 計算交集
        intersection = node_mapped & llm_selected_nodes
        
        if not intersection:
            return 0.0
        
        # 計算匹配分數：交集大小 / LLM 選擇的節點數量
        match_ratio = len(intersection) / len(llm_selected_nodes)
        
        return match_ratio
    
    def _calculate_category_reward(self, node: MCTSNode, categories: List[str]) -> float:
        """
        基於 GPT 識別的功能類別計算獎勵
        
        GPT 的 categories 是第三層的名稱（如 "Campaign Automation", "Calendar & Booking"）
        需要檢查節點路徑中是否包含這些 category 名稱
        """
        if not categories:
            return 0.0
        
        path_nodes = []
        curr = node
        while curr is not None:
            path_nodes.append(curr.name)
            curr = curr.parent
        
        # 構建完整路徑字符串（用於匹配）
        full_path_str = " -> ".join(reversed(path_nodes))
        full_path_lower = full_path_str.lower()
        
        # 檢查每個 GPT category 是否在路徑中
        matches = 0
        matched_categories = []
        for category in categories:
            category_lower = category.lower()
            # 檢查 category 名稱是否在路徑中
            # 例如："Campaign Automation" 應該匹配 "2.1.1 Campaign Automation"
            if category_lower in full_path_lower:
                matches += 1
                matched_categories.append(category)
        
        # 返回匹配比例
        reward = matches / len(categories) if categories else 0.0
        
        # 調試：如果 reward > 0，輸出匹配信息
        if reward > 0:
            print(f"   - Category match found: {full_path_str[:60]}... matches {matched_categories}")
        
        return reward
    
    def search_with_categories(
        self,
        semantic_query: str,
        function_categories: List[str],
        extracted_keywords: Optional[Set[str]] = None,
        llm_selected_nodes: Optional[Set[str]] = None,  # === 新增參數 ===
        iterations: int = 2000,
        top_n: int = 5
    ) -> List[Dict]:
        """
        使用 GPT 提取的功能類別 + 關鍵字匹配進行搜索
        
        Args:
            semantic_query: 語義查詢字符串
            function_categories: GPT 提取的功能類別列表
            extracted_keywords: 提取的關鍵字集合
            iterations: MCTS 迭代次數
            top_n: 返回前 n 個結果
        
        Returns:
            results: 搜索結果列表
        """
        print(f" - MCTS searching with semantic_query: '{semantic_query[:40]}...'")
        print(f" - Using function categories from GPT: {function_categories}")
        if extracted_keywords:
            print(f" - Extracted keywords for matching: {extracted_keywords}")
        # === 新增日誌 ===
        if llm_selected_nodes:
            print(f" - LLM selected mapped_nodes for R_category: {llm_selected_nodes}")
        
        # === 儲存 LLM 選擇的節點供後續使用 ===
        self.llm_selected_nodes = llm_selected_nodes if llm_selected_nodes else set()
        
        # 生成查詢 embedding
        query_embedding = self.model.encode(semantic_query, convert_to_tensor=True)
        print(f"   - Query text: '{semantic_query[:100]}...'")
        print(f"   - Query embedding shape: {query_embedding.shape}")
        
        # ✅ 使用虛擬根節點 "Taxonomy"，將所有頂層分類（1-9）作為第二層
        root_name = "Taxonomy"
        root = MCTSNode(name=root_name, taxonomy_data=self.mcts_taxonomy_tree[root_name])
        
        print(f"   - Using virtual root node: {root_name}")
        print(f"   - Second level nodes: {list(self.mcts_taxonomy_tree[root_name]['children'].keys())}")
        print(f"   - Running {iterations} MCTS iterations...")
        
        # 對單一根節點進行 MCTS 搜索（1222_vincent 的方法）
        for i in range(iterations):
                # 進度輸出（但不影響 MCTS 邏輯）
                if i % 500 == 0 and i > 0:
                    print(f"   - Progress: {i}/{iterations} iterations")
                
                # MCTS 核心邏輯：每次迭代都要執行（1222_vincent 的方法）
                node = root
                
                # Selection: 選擇最佳子節點，直到找到未完全展開的節點
                while node.is_fully_expanded() and node.children:
                    node = node.select_best_child()
                
                # Expansion: 如果節點未完全展開，展開一個新的子節點
                if not node.is_fully_expanded():
                    unexpanded = [
                        n for n in node.taxonomy_data.get('children', {})
                        if n not in [c.name for c in node.children]
                    ]
                    if unexpanded:
                        child_name = np.random.choice(unexpanded)
                        child_node = MCTSNode(
                            child_name,
                            node,
                            node.taxonomy_data['children'][child_name]
                        )
                        node.children.append(child_node)
                        node = child_node
                
                # 計算語義獎勵
                semantic_reward = 0.0
                if node.taxonomy_data.get('embedding') is not None:
                    node_embedding = node.taxonomy_data.get('embedding')
                    if isinstance(node_embedding, list):
                        node_embedding = torch.tensor(node_embedding)
                    semantic_reward = util.cos_sim(
                        query_embedding,
                        node_embedding
                    ).item()
                
                # === 修改：計算類別匹配獎勵 (R_category) - 使用新方法 ===
                category_reward = 0
                if self.llm_selected_nodes and node.taxonomy_data.get('is_leaf', False):
                    # 找到對應的 db_node
                    path_nodes = []
                    curr = node
                    while curr is not None:
                        path_nodes.append(curr.name)
                        curr = curr.parent
                    path_str = " -> ".join(reversed(path_nodes))
                    
                    # 修復：去掉 "Taxonomy -> " 前綴
                    search_path = path_str
                    if path_str.startswith("Taxonomy -> "):
                        search_path = path_str[len("Taxonomy -> "):]
                    
                    db_node = next(
                        (item for item in self.node_database if item["path_str"] == search_path),
                        None
                    )
                    
                    if db_node:
                        category_reward = self._calculate_node_match_reward(db_node, self.llm_selected_nodes)
                else:
                    # Fallback: 如果沒有 llm_selected_nodes，使用舊方法
                    category_reward = self._calculate_category_reward(node, function_categories)
                
                # ✅ 計算關鍵字匹配獎勵
                keyword_reward = 0.0
                if extracted_keywords and node.taxonomy_data.get('is_leaf', False):
                    # 獲取該節點的 example_use_cases
                    path_nodes = []
                    curr = node
                    while curr is not None:
                        path_nodes.append(curr.name)
                        curr = curr.parent
                    path_str = " -> ".join(reversed(path_nodes))
                    
                    # ✅ 修復：node_database 中的 path_str 不包含 "Taxonomy ->" 前綴
                    search_path = path_str
                    if path_str.startswith("Taxonomy -> "):
                        search_path = path_str[len("Taxonomy -> "):]  # 去掉 "Taxonomy -> " 前綴
                    
                    db_node = next(
                        (item for item in self.node_database if item["path_str"] == search_path),
                        None
                    )
                    
                    if db_node:
                        use_cases = db_node.get("example_use_cases", [])
                        match_result = self._fuzzy_match_use_cases(extracted_keywords, use_cases)
                        keyword_reward = match_result["match_score"]
                
                # 混合獎勵（調整權重：semantic*0.5 + categories*0.2 + keywords*0.3）
                total_reward = 0.5 * semantic_reward + 0.2 * category_reward + 0.3 * keyword_reward
                
                # 回溯更新（1222_vincent 的方法）
                temp_node = node
                while temp_node is not None:
                    temp_node.visits += 1
                    temp_node.total_reward += total_reward
                    temp_node = temp_node.parent
        
        # 收集結果：從所有根節點的訪問過的葉子節點中收集
        leaf_nodes_visited = []
        
        def collect_visited_leafs(node: MCTSNode):
            """遞歸收集所有訪問過的葉子節點（從已展開的節點）"""
            # 優先收集標記為 is_leaf 且有訪問次數的節點
            if node.taxonomy_data.get('is_leaf', False):
                if node.visits > 0:  # 只收集訪問過的節點
                    leaf_nodes_visited.append(node)
                return
            
            # 如果節點有 mapped_nodes 但沒有標記為 is_leaf，也收集（可能是標記錯誤）
            if node.taxonomy_data.get('mapped_nodes') and node.visits > 0:
                leaf_nodes_visited.append(node)
                return
            
            # 遞歸處理子節點
            for child in node.children:
                collect_visited_leafs(child)
        
        # ✅ 從單一根節點收集（1222_vincent 的方法）
        collect_visited_leafs(root)
        
        # 調試信息
        print(f"   - MCTS visited {root.visits} times")
        print(f"   - Found {len(leaf_nodes_visited)} visited leaf nodes")
        
        # 如果還是沒有找到葉子節點，嘗試收集所有訪問過的節點（不僅僅是葉子節點）
        if not leaf_nodes_visited:
            print("   - No visited leaf nodes found, collecting all visited nodes...")
            all_visited_nodes = []
            
            def collect_all_visited(node: MCTSNode):
                """收集所有訪問過的節點"""
                if node.visits > 0:
                    all_visited_nodes.append(node)
                for child in node.children:
                    collect_all_visited(child)
            
            collect_all_visited(root)
            print(f"   - Found {len(all_visited_nodes)} total visited nodes (including non-leaves)")
            
            # 如果找到訪問過的節點，嘗試從中找出最接近葉子節點的節點
            if all_visited_nodes:
                # 優先查找有 mapped_nodes 的節點（即使標記為非葉子）
                nodes_with_mapped = [n for n in all_visited_nodes if n.taxonomy_data.get('mapped_nodes')]
                
                if nodes_with_mapped:
                    # 如果找到有 mapped_nodes 的節點，直接使用它們
                    leaf_nodes_visited.extend(nodes_with_mapped)
                    print(f"   - Found {len(nodes_with_mapped)} nodes with mapped_nodes from visited nodes")
                else:
                    # 如果沒有找到，找出深度最深的節點（最接近葉子節點）
                    deepest_nodes = []
                    max_depth = 0
                    for n in all_visited_nodes:
                        depth = 0
                        curr = n
                        while curr.parent:
                            depth += 1
                            curr = curr.parent
                        if depth > max_depth:
                            max_depth = depth
                            deepest_nodes = [n]
                        elif depth == max_depth:
                            deepest_nodes.append(n)
                    
                    # 使用最深的節點（即使沒有 mapped_nodes）
                    leaf_nodes_visited.extend(deepest_nodes[:10])  # 限制為前10個
                    print(f"   - Found {len(deepest_nodes)} deepest nodes (depth={max_depth}), using top 10")
        
        # 移除關鍵字匹配的 fallback（n8n 不需要）
        
        # 排序並轉換為結果格式
        sorted_leafs = sorted(
            leaf_nodes_visited,
            key=lambda n: n.total_reward / (n.visits + 1e-6),
            reverse=True
        )
        
        print(f"   - Sorted {len(sorted_leafs)} visited leaf nodes by reward")
        
        all_leaf_results = []
        semantic_results = []  # 用於收集語義匹配結果
        category_results = []  # 用於收集類別匹配結果
        
        for leaf in sorted_leafs:
            path_nodes = []
            curr = leaf
            while curr is not None:
                path_nodes.append(curr.name)
                curr = curr.parent
            path_str = " -> ".join(reversed(path_nodes))
            
            # ✅ 修復：node_database 中的 path_str 不包含 "Taxonomy ->" 前綴
            # 所以需要去掉 "Taxonomy ->" 前綴來匹配
            search_path = path_str
            if path_str.startswith("Taxonomy -> "):
                search_path = path_str[len("Taxonomy -> "):]  # 去掉 "Taxonomy -> " 前綴
            
            # 嘗試精確匹配
            db_node = next(
                (item for item in self.node_database if item["path_str"] == search_path),
                None
            )
            
            # 如果精確匹配失敗，嘗試模糊匹配（因為節點名稱可能有差異）
            if not db_node:
                # 嘗試匹配路徑的最後部分（葉子節點名稱可能不同）
                path_parts = search_path.split(" -> ")
                if len(path_parts) >= 2:
                    # 嘗試匹配除最後一個節點外的路徑
                    parent_path = " -> ".join(path_parts[:-1])
                    # 查找所有以這個父路徑開頭的節點
                    for item in self.node_database:
                        item_path_parts = item["path_str"].split(" -> ")
                        if len(item_path_parts) >= len(path_parts):
                            item_parent_path = " -> ".join(item_path_parts[:len(path_parts)-1])
                            # 如果父路徑匹配，且最後一部分相似，則認為匹配
                            if item_parent_path == parent_path:
                                # 檢查最後一部分是否有重疊（可能是名稱 vs 編號的差異）
                                last_part_1 = path_parts[-1].lower()
                                last_part_2 = item_path_parts[len(path_parts)-1].lower()
                                if last_part_1 in last_part_2 or last_part_2 in last_part_1 or last_part_1 == last_part_2:
                                    db_node = item
                                    print(f"   - Fuzzy match found: '{search_path}' -> '{item['path_str']}'")
                                    break
            
            # 如果還是沒有找到，嘗試使用節點的 mapped_nodes 來反向查找
            if not db_node and leaf.taxonomy_data.get('mapped_nodes'):
                mapped_nodes = leaf.taxonomy_data.get('mapped_nodes', [])
                # 查找所有包含這些 mapped_nodes 的節點
                for item in self.node_database:
                    item_mapped = set(item.get('mapped_nodes', []))
                    if mapped_nodes and item_mapped:
                        # 如果有交集，可能是同一個節點
                        intersection = set(mapped_nodes) & item_mapped
                        if len(intersection) >= min(2, len(mapped_nodes), len(item_mapped)):  # 至少有2個節點匹配，或全部匹配
                            db_node = item
                            print(f"   - Matched by mapped_nodes: '{search_path}' -> '{item['path_str']}' (intersection: {len(intersection)} nodes)")
                            break
            
            if db_node:
                avg_reward = leaf.total_reward / (leaf.visits + 1e-6)
                
                # 重新計算 semantic 和 category 分數用於調試
                semantic_score = 0.0
                category_score = 0.0
                
                # 計算語義分數
                # 檢查是否有 embedding，如果沒有，從 node_database 中獲取
                leaf_embedding = leaf.taxonomy_data.get('embedding')
                if leaf_embedding is None:
                    # 嘗試從 node_database 中獲取 embedding
                    db_node_embedding = db_node.get('embedding')
                    if db_node_embedding is not None:
                        leaf_embedding = db_node_embedding
                
                if leaf_embedding is not None:
                    # 確保 embedding 是 tensor
                    if isinstance(leaf_embedding, list):
                        leaf_embedding = torch.tensor(leaf_embedding)
                    semantic_score = util.cos_sim(
                        query_embedding,
                        leaf_embedding
                    ).item()
                    
                    # 調試：顯示 semantic matching 的詳細信息
                    description = db_node.get('description', '')
                    combined_text_used = db_node.get('combined_text', f"{path_str}: {description}")
                    print(f"   - Semantic Match Details for '{path_str[:60]}...':")
                    print(f"      Combined text used: '{combined_text_used[:100]}...'")
                    print(f"      Semantic score: {semantic_score:.4f}")
                    print(f"      Query: '{semantic_query[:60]}...'")
                else:
                    # 嘗試使用 db_node 的 combined_text 來查找 embedding
                    db_combined_text = db_node.get('combined_text', '')
                    if db_combined_text and db_combined_text in self.text_embedding_map:
                        leaf_embedding = self.text_embedding_map[db_combined_text]
                        print(f"   - Found embedding via db_node.combined_text")
                        # 重新計算 semantic_score
                        if isinstance(leaf_embedding, list):
                            leaf_embedding = torch.tensor(leaf_embedding)
                        semantic_score = util.cos_sim(
                            query_embedding,
                            leaf_embedding
                        ).item()
                    else:
                        print(f"   - Warning: No embedding for {path_str}")
                        # 調試：找出為什麼沒有 embedding
                        description = db_node.get('description', '')
                        expected_combined_text = f"{search_path}: {description}"
                        print(f"      Expected combined_text: '{expected_combined_text[:80]}...'")
                        print(f"      Available in text_embedding_map: {expected_combined_text in self.text_embedding_map}")
                        # 檢查是否有類似的 combined_text
                        path_parts = search_path.split(' -> ')
                        similar_texts = [text for text in self.text_embedding_map.keys() 
                                       if path_parts and (path_parts[-1] in text or search_path.split(' -> ')[-1] in text)]
                        if similar_texts:
                            print(f"      Similar texts found: {len(similar_texts)}")
                            for st in similar_texts[:3]:
                                print(f"        - '{st[:80]}...'")
                            # 使用第一個相似的 text 的 embedding
                            if similar_texts:
                                leaf_embedding = self.text_embedding_map[similar_texts[0]]
                                if isinstance(leaf_embedding, list):
                                    leaf_embedding = torch.tensor(leaf_embedding)
                                semantic_score = util.cos_sim(
                                    query_embedding,
                                    leaf_embedding
                                ).item()
                                print(f"      Using embedding from similar text, semantic_score={semantic_score:.4f}")
                
                # 計算類別分數
                category_score = self._calculate_category_reward(leaf, function_categories)
                
                # ✅ 計算關鍵字匹配分數
                keyword_score = 0.0
                keyword_matches = []
                if extracted_keywords:
                    use_cases = db_node.get("example_use_cases", [])
                    match_result = self._fuzzy_match_use_cases(extracted_keywords, use_cases)
                    keyword_score = match_result["match_score"]
                    keyword_matches = match_result.get("matched_cases", [])
                
                # 調試：輸出類別匹配詳情
                if category_score == 0.0 and function_categories:
                    path_nodes_for_cat = []
                    curr = leaf
                    while curr is not None:
                        path_nodes_for_cat.append(curr.name)
                        curr = curr.parent
                    full_path_str = " ".join(reversed(path_nodes_for_cat)).lower()
                    print(f"   - Debug category: path='{full_path_str[:50]}...', categories={function_categories}")
                
                # 檢查 LLM 節點匹配
                has_node_match = False
                node_matched = set()
                if self.llm_selected_nodes:
                    node_mapped = set(db_node.get("mapped_nodes", []))
                    node_matched = node_mapped & self.llm_selected_nodes
                    has_node_match = bool(node_matched)
                
                # ===== 舊的複雜閾值邏輯（已註解） =====
                # # ✅ 改進：使用多種條件來決定是否收集節點
                # # 不僅僅依賴 avg_reward，還要考慮 semantic_score 和 category_score
                # has_keyword_match = keyword_score > 0
                # 
                # # 動態計算閾值
                # base_threshold = 0.20 if (has_keyword_match or has_node_match) else 0.35
                # 
                # # 計算綜合分數：考慮 semantic 和 category 分數
                # # 如果 semantic 或 category 分數很高，即使 avg_reward 稍低也應該被收集
                # semantic_bonus = 1.0 if semantic_score > 0.3 else 0.0  # semantic 高時給予獎勵
                # category_bonus = 1.0 if category_score > 0.4 else 0.0  # category 高時給予獎勵
                # 
                # # 調整閾值：如果有 semantic 或 category 高分，降低閾值
                # if semantic_bonus > 0 or category_bonus > 0:
                #     threshold = base_threshold * 0.75  # 降低閾值一半
                # else:
                #     threshold = base_threshold
                # 
                # # 或者使用綜合分數來決定：avg_reward 或者 semantic/category 分數高都可以
                # meets_threshold = (
                #     avg_reward > threshold or 
                #     semantic_score > 0.3 or  # semantic 分數高
                #     category_score > 0.4 or  # category 分數高
                #     (has_keyword_match and avg_reward > 0.15)  # 有關鍵字匹配且 reward > 0.15
                # )
                # ===== 舊的複雜閾值邏輯（結束） =====
                
                # ✅ 使用 0105_vincent 版本的簡單閾值邏輯
                has_keyword_match = keyword_score > 0
                
                # 動態調整閾值（0105_vincent 的方法）
                threshold = 0.2
                if has_keyword_match or has_node_match:
                    threshold = 0.15
                
                # 簡單判斷：只檢查 avg_reward 是否超過閾值
                meets_threshold = avg_reward > threshold
                
                # 記錄語義和類別匹配結果（包含閾值信息）
                semantic_results.append({
                    'path_str': path_str,
                    'semantic_score': semantic_score,
                    'avg_reward': avg_reward
                })
                
                category_results.append({
                    'path_str': path_str,
                    'category_score': category_score,
                    'avg_reward': avg_reward,
                    'keyword_score': keyword_score,
                    'has_keyword_match': has_keyword_match,
                    'has_node_match': has_node_match,
                    'threshold': threshold,
                    'passed_threshold': meets_threshold
                })
                
                if meets_threshold:
                    # ✅ 將匹配結果也存入
                    db_node_with_match = {
                        **db_node,
                        'avg_reward': avg_reward,
                        'semantic_score': semantic_score,
                        'category_score': category_score,
                        'keyword_score': keyword_score,
                        'visits': leaf.visits,
                        'total_reward': leaf.total_reward,
                        'keyword_matches': keyword_matches if has_keyword_match else [],
                        'node_matches': list(node_matched) if has_node_match else []  # === 新增 ===
                    }
                    all_leaf_results.append(db_node_with_match)
            else:
                # 即使找不到 db_node，如果節點有 mapped_nodes，也嘗試使用它
                if leaf.taxonomy_data.get('mapped_nodes'):
                    print(f"   - Warning: Could not find db_node for path: {path_str}, but node has mapped_nodes, creating synthetic entry...")
                    # 創建一個合成條目
                    synthetic_node = {
                        "description": leaf.taxonomy_data.get('description', ''),
                        "path_str": search_path,
                        "mapped_nodes": leaf.taxonomy_data.get('mapped_nodes', []),
                        "example_use_cases": [],
                        "combined_text": f"{search_path}: {leaf.taxonomy_data.get('description', '')}"
                    }
                    # 嘗試計算 semantic score
                    leaf_embedding = leaf.taxonomy_data.get('embedding')
                    semantic_score = 0.0
                    if leaf_embedding is not None:
                        if isinstance(leaf_embedding, list):
                            leaf_embedding = torch.tensor(leaf_embedding)
                        semantic_score = util.cos_sim(
                            query_embedding,
                            leaf_embedding
                        ).item()
                    
                    avg_reward = leaf.total_reward / (leaf.visits + 1e-6)
                    
                    # 計算 category_score
                    category_score = 0.0
                    if self.llm_selected_nodes:
                        category_score = self._calculate_node_match_reward(synthetic_node, self.llm_selected_nodes)
                    else:
                        category_score = self._calculate_category_reward(leaf, function_categories)
                    
                    # 計算 keyword_score
                    keyword_score = 0.0
                    keyword_matches = []
                    if extracted_keywords:
                        # 簡單的關鍵字匹配（因為沒有 use_cases）
                        path_text = f"{search_path} {synthetic_node['description']}".lower()
                        for kw in extracted_keywords:
                            if kw.lower() in path_text:
                                keyword_matches.append(kw)
                        if keyword_matches:
                            keyword_score = len(keyword_matches) / len(extracted_keywords)
                    
                    # 使用相同的閾值邏輯檢查
                    has_keyword_match = keyword_score > 0
                    has_node_match = bool(set(leaf.taxonomy_data.get('mapped_nodes', [])) & self.llm_selected_nodes) if self.llm_selected_nodes else False
                    
                    base_threshold = 0.15 if (has_keyword_match or has_node_match) else 0.2
                    meets_threshold = (
                        avg_reward > base_threshold * 0.5 or 
                        semantic_score > 0.3 or
                        category_score > 0.4 or
                        (has_keyword_match and avg_reward > 0.15)
                    )
                    
                    if meets_threshold:
                        # 使用合成的節點
                        synthetic_node_with_match = {
                            **synthetic_node,
                            'avg_reward': avg_reward,
                            'semantic_score': semantic_score,
                            'category_score': category_score,
                            'keyword_score': keyword_score,
                            'visits': leaf.visits,
                            'total_reward': leaf.total_reward,
                            'keyword_matches': keyword_matches,
                            'node_matches': list(set(leaf.taxonomy_data.get('mapped_nodes', [])) & self.llm_selected_nodes) if self.llm_selected_nodes else []
                        }
                        all_leaf_results.append(synthetic_node_with_match)
                else:
                    print(f"   - Warning: Could not find db_node for path: {path_str} (and no mapped_nodes)")
        
        # 輸出語義匹配結果（MCTS 找到的節點）
        print(f"\n   📊 Semantic Matching Results (MCTS found {len(semantic_results)} nodes):")
        semantic_sorted = sorted(semantic_results, key=lambda x: x['semantic_score'], reverse=True)
        for i, result in enumerate(semantic_sorted):
            path_preview = result['path_str'][:70] + "..." if len(result['path_str']) > 70 else result['path_str']
            print(f"      {i+1}. {path_preview}")
            print(f"         Semantic Score: {result['semantic_score']:.4f} | Avg Reward: {result['avg_reward']:.4f}")
            
            # 顯示該節點使用的 combined_text（用於調試 embedding 質量）
            db_node = next(
                (item for item in self.node_database if item["path_str"] == result['path_str']),
                None
            )
            if db_node:
                combined_text = db_node.get('combined_text', 'N/A')
                description = db_node.get('description', 'N/A')
                print(f"         Combined text: '{combined_text[:90]}...'")
                print(f"         Description: '{description[:60]}...'")
                print(f"         Query: '{semantic_query[:60]}...'")
                print(f"         ---")
        
        # 輸出類別匹配結果（GPT 提取的 function categories）
        print(f"\n   📊 Category Matching Results (GPT categories: {function_categories}):")
        category_sorted = sorted(category_results, key=lambda x: x['category_score'], reverse=True)
        for i, result in enumerate(category_sorted):
            path_preview = result['path_str'][:70] + "..." if len(result['path_str']) > 70 else result['path_str']
            print(f"      {i+1}. {path_preview}")
            print(f"         Category Score: {result['category_score']:.4f} | Avg Reward: {result['avg_reward']:.4f}")
            
            # 顯示閾值信息（0105_vincent 版本的簡單邏輯）
            threshold = result.get('threshold', 0.2)
            has_keyword_match = result.get('has_keyword_match', False)
            has_node_match = result.get('has_node_match', False)
            passed_threshold = result.get('passed_threshold', False)
            keyword_score = result.get('keyword_score', 0.0)
            
            # 構建閾值描述（0105_vincent 版本）
            if has_keyword_match or has_node_match:
                threshold_type = "0.15 (with keyword/node match)"
            else:
                threshold_type = "0.2 (no keyword/node match)"
            
            status = "✅ PASSED" if passed_threshold else "❌ FILTERED"
            print(f"         Keyword Score: {keyword_score:.4f} | Node Match: {has_node_match} | Threshold: {threshold_type} | {status}")
            if not passed_threshold:
                print(f"         Reason: avg_reward {result['avg_reward']:.4f} <= threshold {threshold:.2f}")
            
            # 顯示匹配的 categories
            path_lower = result['path_str'].lower()
            matched_cats = [cat for cat in function_categories if cat.lower() in path_lower]
            if matched_cats:
                print(f"         Matched Categories: {matched_cats}")
            else:
                print(f"         Matched Categories: None (path doesn't contain any GPT categories)")
        
        # 輸出算分過程
        print(f"\n   📊 Scoring Process (for each node):")
        for i, result in enumerate(all_leaf_results[:5]):  # 顯示前5個的詳細算分
            path_preview = result['path_str'][:60] + "..." if len(result['path_str']) > 60 else result['path_str']
            semantic = result.get('semantic_score', 0.0)
            category = result.get('category_score', 0.0)
            avg_reward = result.get('avg_reward', 0.0)
            visits = result.get('visits', 0)
            total_reward = result.get('total_reward', 0.0)
            
            # 計算公式：total_reward = 0.5 * semantic + 0.2 * category + 0.3 * keyword
            keyword = result.get('keyword_score', 0.0)
            calculated_reward = 0.5 * semantic + 0.2 * category + 0.3 * keyword
            
            print(f"      Node {i+1}: {path_preview}")
            print(f"         - Semantic Score: {semantic:.4f} (weight: 0.5)")
            print(f"         - Category Score: {category:.4f} (weight: 0.2)")
            print(f"         - Keyword Score: {keyword:.4f} (weight: 0.3)")
            print(f"         - Calculated Reward: {calculated_reward:.4f} = 0.5 * {semantic:.4f} + 0.2 * {category:.4f} + 0.3 * {keyword:.4f}")
            print(f"         - MCTS Visits: {visits} | Total Reward: {total_reward:.4f} | Avg Reward: {avg_reward:.4f}")
        
        # 按平均獎勵排序
        all_leaf_results.sort(key=lambda x: x.get('avg_reward', 0), reverse=True)
        
        print(f"   - Collected {len(all_leaf_results)} results from visited leaf nodes")
        if all_leaf_results:
            print(f"   - Top result: {all_leaf_results[0].get('path_str', 'N/A')} (reward: {all_leaf_results[0].get('avg_reward', 0):.3f})")
        
        # ✅ 使用 1222_vincent 的方法：只確保關鍵類別覆蓋，不主動擴展每個大類
        results = self._ensure_category_coverage(all_leaf_results, function_categories, top_n)
        
        return results
    
    def _get_path_to_root(self, node: MCTSNode) -> List[MCTSNode]:
        """獲取從節點到根的路徑"""
        path = []
        curr = node
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return path[::-1]
    
    
    def _select_results_by_category_coverage(
        self, 
        all_results: List[Dict], 
        function_categories: List[str], 
        top_n: int,
        query_embedding=None,
        extracted_keywords: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        按第一層分類分組選擇結果，確保每個大類可以返回多個子葉
        
        即使 MCTS 只找到一個節點，也要從該大類中擴展選擇多個相關的子葉。
        例如：對於 "AI, ML & Automation Intelligence" 大類，可以返回：
        - 6.1.1 LLM Providers
        - 6.2.1 Memory Stores
        - 6.4.1 Tool Augmentation
        等多個相關子葉
        """
        if not all_results:
            return []
        
        # 提取第一層分類（例如："6 AI, ML & Automation Intelligence"）
        def get_first_level_category(path_str: str) -> str:
            """從路徑中提取第一層分類"""
            parts = path_str.split(" -> ")
            if parts:
                first_part = parts[0]
                # 去掉數字前綴（例如："6 AI, ML & Automation Intelligence" -> "AI, ML & Automation Intelligence"）
                first_parts = first_part.split(' ', 1)
                if len(first_parts) == 2 and first_parts[0].isdigit():
                    return first_parts[1]
                return first_part
            return "Unknown"
        
        # 按第一層分類分組 MCTS 結果
        category_groups = {}
        for result in all_results:
            path_str = result.get('path_str', '')
            category = get_first_level_category(path_str)
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(result)
        
        print(f"   - Results grouped into {len(category_groups)} top-level categories:")
        for cat, group_results in category_groups.items():
            print(f"      - {cat}: {len(group_results)} nodes from MCTS")
        
        # ✅ 關鍵改進：對於每個 GPT category，從 node_database 中擴展選擇同一大類的其他節點
        selected_results = []
        selected_paths = set()
        
        # 先添加 MCTS 找到的結果
        for result in all_results:
            path_str = result.get('path_str', '')
            selected_results.append(result)
            selected_paths.add(path_str)
        
        # ✅ 改進：需要 query_embedding 和 extracted_keywords 來計算擴展節點的分數
        # 這些參數需要在方法簽名中傳入，或者作為類的屬性
        # 暫時從外部獲取（通過 closure 或參數傳遞）
        # 注意：這裡需要 query_embedding，但它在 search_with_categories 方法中
        # 我們需要將它作為參數傳入，或者存儲為實例變量
        
        # 對於每個 GPT category，擴展選擇同一大類的其他節點
        for gpt_category in function_categories:
            gpt_cat_lower = gpt_category.lower()
            
            # 找到匹配的第一層分類（從 MCTS 結果中）
            matching_categories = []
            for cat, group_results in category_groups.items():
                if gpt_cat_lower in cat.lower() or cat.lower() in gpt_cat_lower:
                    matching_categories.append(cat)
            
            # 對於每個匹配的分類，從 node_database 中選擇更多節點
            for matching_cat in matching_categories:
                # ✅ 優化：預先過濾，只處理屬於該大類的節點，避免重複遍歷
                # 先快速過濾出屬於該大類的節點
                category_nodes = [
                    db_node for db_node in self.node_database
                    if get_first_level_category(db_node.get('path_str', '')) == matching_cat
                    and db_node.get('path_str', '') not in selected_paths
                ]
                
                if not category_nodes:
                    continue
                
                # ✅ 優化：限制候選節點數量，只計算前 50 個的語義相似度（避免計算太多）
                # 先按 mapped_nodes 數量排序，優先考慮有更多節點的 taxonomy 節點
                category_nodes.sort(key=lambda x: len(x.get('mapped_nodes', [])), reverse=True)
                category_nodes = category_nodes[:50]  # 只處理前 50 個
                
                candidates_from_db = []
                for db_node in category_nodes:
                    db_path_str = db_node.get('path_str', '')
                    
                    # ✅ 改進：計算語義相似度，選擇更相關的節點
                    # 獲取該節點的 combined_text 和 embedding
                    combined_text = db_node.get('combined_text', f"{db_path_str}: {db_node.get('description', '')}")
                    node_embedding = self.text_embedding_map.get(combined_text)
                    
                    semantic_score = 0.0
                    if node_embedding is not None:
                        if isinstance(node_embedding, list):
                            node_embedding = torch.tensor(node_embedding)
                        # 計算與查詢的語義相似度
                        semantic_score = util.cos_sim(query_embedding, node_embedding).item()
                    
                    # ✅ 優化：只在語義分數較高時才計算關鍵字匹配（避免不必要的計算）
                    keyword_score = 0.0
                    if extracted_keywords and semantic_score > 0.15:  # 只對語義相關的節點計算關鍵字
                        use_cases = db_node.get("example_use_cases", [])
                        if use_cases:  # 只在有 use_cases 時才計算
                            match_result = self._fuzzy_match_use_cases(extracted_keywords, use_cases, semantic_threshold=0.25)
                            keyword_score = match_result["match_score"]
                    
                    # 計算綜合分數：semantic*0.5 + category*0.2 + keyword*0.3
                    category_score = 0.5  # 因為匹配了 category
                    calculated_reward = 0.5 * semantic_score + 0.2 * category_score + 0.3 * keyword_score
                    
                    candidates_from_db.append({
                        **db_node,
                        'avg_reward': calculated_reward,
                        'semantic_score': semantic_score,
                        'category_score': category_score,
                        'keyword_score': keyword_score,
                        'visits': 0,
                        'total_reward': calculated_reward
                    })
                
                # 按 reward 排序（現在有真實的分數了）
                candidates_from_db.sort(key=lambda x: x.get('avg_reward', 0), reverse=True)
                
                # 對於每個匹配的大類，選擇最多 3 個節點（包括 MCTS 找到的）
                # 如果 MCTS 已經找到了一些，再補充到 3 個
                mcts_count = len([r for r in all_results if get_first_level_category(r.get('path_str', '')) == matching_cat])
                max_per_category = 3
                needed = max(0, max_per_category - mcts_count)
                
                print(f"   - Expanding '{matching_cat}': MCTS found {mcts_count} nodes, adding {needed} more from database")
                print(f"      Found {len(candidates_from_db)} candidates, selecting top {needed} by semantic+category+keyword score")
                
                for candidate in candidates_from_db[:needed]:
                    path_str = candidate.get('path_str', '')
                    if path_str not in selected_paths:
                        selected_results.append(candidate)
                        selected_paths.add(path_str)
                        print(f"      Added: {path_str[:60]}... (reward: {candidate.get('avg_reward', 0):.3f}, semantic: {candidate.get('semantic_score', 0):.3f})")
                        if len(selected_results) >= top_n * 2:  # 允許更多結果，後續會限制
                            break
                if len(selected_results) >= top_n * 2:
                    break
        
        # 按 reward 排序
        selected_results.sort(key=lambda x: x.get('avg_reward', 0), reverse=True)
        
        # 限制最終數量
        final_results = selected_results[:top_n] if len(selected_results) > top_n else selected_results
        
        print(f"   - Selected {len(final_results)} results (expanded from {len(all_results)} MCTS results)")
        return final_results
    
    def _has_category_match(self, db_node: Dict, categories: List[str]) -> bool:
        """檢查節點是否匹配類別"""
        if not categories:
            return False
        
        path_text = f"{db_node.get('path_str', '')} {db_node.get('description', '')}".lower()
        if any(category.lower() in path_text for category in categories):
            return True
        
        mapped_nodes_text = " ".join(db_node.get('mapped_nodes', [])).lower()
        if any(category.lower() in mapped_nodes_text for category in categories):
            return True
        
        return False
    
    def _ensure_category_coverage(self, results: List[Dict], categories: List[str], top_n: int) -> List[Dict]:
        """確保類別覆蓋"""
        critical_categories = ['SMS', 'Email', 'Payment', 'Database', 'API']
        
        for category in critical_categories:
            if category not in categories:
                continue
            
            has_coverage = any(self._has_category_match(r, [category]) for r in results)
            
            if not has_coverage:
                for db_node in self.node_database:
                    if self._has_category_match(db_node, [category]):
                        if not any(r['path_str'] == db_node['path_str'] for r in results):
                            db_node_with_reward = {**db_node, 'avg_reward': 0.40}
                            results.append(db_node_with_reward)
                            print(f" - Category boost: Forcibly added '{db_node['path_str']}' for category '{category}'")
                        break
        
        # 重新排序並限制最終數量
        results.sort(key=lambda x: x.get('avg_reward', 0), reverse=True)
        return results[:top_n]
    
    def search_by_keywords(self, keywords: Set[str]) -> List[Dict]:
        """
        執行直接的關鍵字掃描（Lexical Search）
        
        Args:
            keywords: 關鍵字集合
        
        Returns:
            keyword_hits: 匹配的節點列表
        """
        if not keywords:
            return []
        
        print(" - Performing direct keyword scan...")
        keyword_hits = []
        lower_keywords = {kw.lower() for kw in keywords}
        
        for db_node in self.node_database:
            # 檢查路徑和描述
            path_str = db_node.get('path_str', '').lower()
            description = db_node.get('description', '').lower()
            mapped_nodes = " ".join(db_node.get('mapped_nodes', [])).lower()
            
            # 組合所有文本
            text_to_check = f"{path_str} {description} {mapped_nodes}"
            
            # 檢查是否有任何關鍵字匹配（部分匹配）
            found_in_text = False
            matched_keywords = []
            for kw in lower_keywords:
                # 檢查關鍵字是否在文本中
                if kw in text_to_check:
                    found_in_text = True
                    matched_keywords.append(kw)
                # 也檢查關鍵字是否在節點類型中（如 "ocr" 在 "n8n-nodes-base.ocr"）
                elif any(kw in node.lower() for node in db_node.get('mapped_nodes', [])):
                    found_in_text = True
                    matched_keywords.append(kw)
            
            # 也檢查 example_use_cases（如果有的話，像原本的程式碼）
            use_cases = db_node.get("example_use_cases", [])
            match_result = self._fuzzy_match_use_cases(keywords, use_cases)
            found_in_cases = match_result["match_score"] > 0
            matched_use_cases = match_result.get("matched_cases", [])
            
            if found_in_text or found_in_cases:
                # 根據匹配類型給不同的分數
                if found_in_cases:
                    # 如果匹配到 use cases，給更高的分數（因為更精確）
                    reward = 0.6 + (match_result["match_score"] * 0.2)  # 0.6-0.8
                    match_reason = f"Keyword in use cases (score: {match_result['match_score']:.2f})"
                else:
                    # 如果只在文本中匹配，給基礎分數
                    reward = 0.5
                    match_reason = "Keyword in path/description/mapped_nodes"
                
                db_node_with_match = {
                    **db_node,
                    'avg_reward': reward,
                    'keyword_matches': matched_keywords if matched_keywords else list(keywords),
                    'matched_use_cases': matched_use_cases,
                    'keyword_match_score': match_result["match_score"] if found_in_cases else 0.0,
                    'match_reason': match_reason
                }
                keyword_hits.append(db_node_with_match)
        
        print(f" - Direct keyword scan found {len(keyword_hits)} hits.")
        return keyword_hits
    
    def _fuzzy_match_use_cases(self, keywords: Set[str], use_cases: List[str], semantic_threshold: float = 0.3) -> Dict:
        """
        對 example_use_cases 進行語義匹配（使用 embedding 相似度）
        
        使用語義匹配而不是純文字匹配，因為 GPT 提取的 keywords 和 example_use_cases
        之間可能沒有直接的文字匹配，但語義上可能相關。
        
        Args:
            keywords: GPT 提取的關鍵字集合
            use_cases: example_use_cases 列表
            semantic_threshold: 語義相似度閾值（0-1），低於此值不認為匹配
        
        Returns: {"matched_cases": [...], "match_score": 0.0-1.0, "semantic_scores": {...}}
        """
        if not use_cases or not keywords:
            return {"matched_cases": [], "match_score": 0.0, "semantic_scores": {}}
        
        # 組合所有 keywords 為一個查詢文本
        query_text = " ".join(keywords)
        
        # ✅ 優化：緩存查詢 embedding（如果 keywords 相同，可以重用）
        query_cache_key = query_text
        if query_cache_key not in self.use_case_embedding_cache:
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            self.use_case_embedding_cache[query_cache_key] = query_embedding
        else:
            query_embedding = self.use_case_embedding_cache[query_cache_key]
        
        # ✅ 優化：緩存 use_case embeddings，避免重複計算
        use_case_embeddings_list = []
        for use_case in use_cases:
            if use_case in self.use_case_embedding_cache:
                use_case_embeddings_list.append(self.use_case_embedding_cache[use_case])
            else:
                embedding = self.model.encode(use_case, convert_to_tensor=True)
                self.use_case_embedding_cache[use_case] = embedding
                use_case_embeddings_list.append(embedding)
        
        # 將列表轉換為 tensor（stack）
        use_case_embeddings = torch.stack(use_case_embeddings_list)
        
        # 計算語義相似度（餘弦相似度）
        similarities = util.cos_sim(query_embedding, use_case_embeddings)[0]  # [0] 因為 query 只有一個
        
        # 找到匹配的 use cases（相似度 >= threshold）
        matched_cases = []
        matched_scores = {}
        semantic_scores = {}
        
        for i, (case, similarity) in enumerate(zip(use_cases, similarities)):
            similarity_value = similarity.item()
            semantic_scores[case] = similarity_value
            
            if similarity_value >= semantic_threshold:
                matched_cases.append(case)
                matched_scores[case] = similarity_value
        
        # 計算匹配分數：使用最高相似度或平均相似度
        if matched_scores:
            # 使用最高相似度作為 match_score
            max_similarity = max(matched_scores.values())
            # 或者使用平均相似度
            avg_similarity = sum(matched_scores.values()) / len(matched_scores) if matched_scores else 0.0
            # 綜合考慮：最高相似度 * 0.7 + 平均相似度 * 0.3
            match_score = max_similarity * 0.7 + avg_similarity * 0.3
        else:
            match_score = 0.0
        
        # 調試：如果找到匹配，輸出信息
        if match_score > 0:
            print(f"   - Semantic keyword match: {len(matched_cases)}/{len(use_cases)} use cases matched (max similarity: {max(matched_scores.values()):.3f})")
        
        return {
            "matched_cases": matched_cases,
            "match_score": match_score,
            "semantic_scores": semantic_scores,
            "matched_scores": matched_scores
        }

