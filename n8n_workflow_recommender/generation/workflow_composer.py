#!/usr/bin/env python3
"""
工作流程組合器（A* 路徑生成）

使用 A* 算法在知識圖中生成工作流程路徑。
"""

import networkx as nx
from typing import List, Dict, Set, Optional, Tuple
from collections import deque
import numpy as np


class DomainKnowledgeGraph:
    """
    領域知識圖
    
    使用 NetworkX 構建的有向圖，用於 A* 路徑搜索。
    """
    """
    領域知識圖
    
    使用 NetworkX 構建的有向圖，用於 A* 路徑搜索。
    """
    
    def __init__(self, triples: List[Tuple[str, str, str]], ontology: Dict, auxiliary_keywords: Optional[List[str]] = None):
        """
        初始化知識圖
        
        Args:
            triples: 三元組列表 [(head, relation, tail), ...]
            ontology: Ontology 字典 {node_type: {...}}
            auxiliary_keywords: 輔助關鍵字列表（用於擴展節點）
        """
        print("PHASE 1B: Initializing Domain Knowledge Graph (A*)...")
        self.graph = nx.DiGraph()
        self.auxiliary_keywords = auxiliary_keywords if auxiliary_keywords else []
        
        # n8n 的優先級起點和終點
        self.PRIORITY_SOURCES = [
            'n8n-nodes-base.manualTrigger',
            'n8n-nodes-base.webhook',
            'n8n-nodes-base.scheduleTrigger'
        ]
        self.PRIORITY_SINKS = [
            'n8n-nodes-base.noOp',
            'n8n-nodes-base.stopAndError'
        ]
        
        self._build_graph(triples, ontology)
        print(" - A* Graph built successfully.")
    
    def _build_graph(self, triples: List[Tuple[str, str, str]], ontology: Dict):
        """構建 NetworkX 圖"""
        all_nodes = set([h for h, _, _ in triples] + [t for _, _, t in triples])
        
        for node in all_nodes:
            node_attrs = ontology.get(node, {})
            self.graph.add_node(node, **node_attrs)
        
        for h, r, t in triples:
            self.graph.add_edge(h, t, relation=r, weight=1.0)
    
    def expand_with_dependencies(self, core_nodes: List[str], max_expansion: int = 30) -> List[str]:
        """
        精準擴展：只拉入與 core_nodes 有直接關係的節點（不擴展太深）
        強制加入起點與終點
        嚴格限制擴展以避免過度擴展
        """
        core_set = set(core_nodes)
        expanded = set(core_nodes)
        
        # 強制加入起點與終點
        for n in self.PRIORITY_SOURCES + self.PRIORITY_SINKS:
            if n in self.graph:
                expanded.add(n)
        
        # 如果核心節點太多，只取前 N 個進行擴展
        if len(core_set) > max_expansion:
            print(f"   ⚠️  Too many core nodes ({len(core_set)}), limiting to {max_expansion} for expansion")
            core_set = set(list(core_set)[:max_expansion])
        
        # 只擴展一層：直接前置和後繼（不遞歸擴展）
        to_process = list(core_set)
        expansion_count = 0
        max_expansions = 50  # 大幅減少擴展次數
        
        while to_process and expansion_count < max_expansions:
            n = to_process.pop()
            if n in self.graph:
                # 只擴展直接前置（最多1個）
                for p in list(self.graph.predecessors(n))[:1]:
                    if p not in expanded and expansion_count < max_expansions:
                        expanded.add(p)
                        expansion_count += 1
                # 只擴展直接後繼（最多1個）
                for s in list(self.graph.successors(n))[:1]:
                    if s not in expanded and expansion_count < max_expansions:
                        expanded.add(s)
                        expansion_count += 1
        
        result = list(expanded)
        # 嚴格限制結果數量
        if len(result) > 50:
            print(f"   ⚠️  Expanded to {len(result)} nodes, limiting to 50 for performance")
            # 優先保留核心節點和起點終點
            priority = set(core_nodes) | set(self.PRIORITY_SOURCES) | set(self.PRIORITY_SINKS)
            result = list(priority) + [n for n in result if n not in priority][:50-len(priority)]
        
        return result
    
    def _find_path_for_nodes(self, nodes: List[str], core_nodes: Set[str]) -> List[str]:
        """
        動態生成路徑：
        - 起點：強制優先 self.PRIORITY_SOURCES
        - 終點：強制優先 self.PRIORITY_SINKS → 再用 outdegree=0
        - 路徑：最長簡單路徑 + 過濾無關節點
        """
        if not nodes:
            return []
        if len(nodes) == 1:
            return nodes
        
        subgraph = self.graph.subgraph(nodes)
        in_degrees = dict(subgraph.in_degree())
        out_degrees = dict(subgraph.out_degree())
        
        # 1. 強制優先級起點
        sources = [n for n in self.PRIORITY_SOURCES if n in in_degrees]
        if not sources:
            min_indegree = min(in_degrees.values()) if in_degrees else 0
            sources = [n for n, d in in_degrees.items() if d == min_indegree]
        
        # 2. 強制優先級終點
        sinks = [n for n in self.PRIORITY_SINKS if n in out_degrees]
        if not sinks:
            sinks = [n for n, d in out_degrees.items() if d == 0]
        if not sinks:
            min_out = min(out_degrees.values()) if out_degrees else 0
            sinks = [n for n, d in out_degrees.items() if d == min_out]
        
        print(f" - Dynamic Source(s): {sources}")
        print(f" - Dynamic Sink(s): {sinks}")
        
        # 3. 尋找最長簡單路徑（限制搜索以避免卡住）
        longest_path = []
        best_coverage = 0.0
        
        # 如果子圖太大，先過濾節點
        if len(nodes) > 50:
            print(f"   ⚠️  Subgraph too large ({len(nodes)} nodes), filtering to essential nodes...")
            # 只保留核心節點和與它們直接相連的節點
            essential_nodes = set(core_nodes)
            # 添加與核心節點直接相連的節點
            for core_node in list(core_nodes)[:20]:  # 只處理前20個核心節點
                if core_node in subgraph:
                    # 添加直接前置和後繼
                    for pred in list(subgraph.predecessors(core_node))[:3]:
                        essential_nodes.add(pred)
                    for succ in list(subgraph.successors(core_node))[:3]:
                        essential_nodes.add(succ)
            
            # 添加起點和終點
            essential_nodes.update(sources)
            essential_nodes.update(sinks)
            
            # 限制為最多50個節點
            essential_nodes_list = list(essential_nodes)
            if len(essential_nodes_list) > 50:
                priority = set(core_nodes) | set(sources) | set(sinks)
                essential_nodes_list = list(priority) + [n for n in essential_nodes_list if n not in priority][:50-len(priority)]
            
            print(f"   - Filtered to {len(essential_nodes_list)} essential nodes")
            # 使用過濾後的節點重新構建子圖
            subgraph = self.graph.subgraph(essential_nodes_list)
            nodes = essential_nodes_list
        
        # 限制路徑搜索的深度和數量
        max_path_length = min(20, len(nodes))
        path_count = 0
        max_paths_to_check = 50
        
        for source in sources[:2]:  # 只檢查前2個起點
            for sink in sinks[:2]:  # 只檢查前2個終點
                if source == sink:
                    continue
                try:
                    # 限制路徑長度和數量
                    paths = nx.all_simple_paths(subgraph, source, sink, cutoff=max_path_length)
                    for path in paths:
                        if path_count >= max_paths_to_check:
                            break
                        path_count += 1
                        coverage = len(set(path) & core_nodes) / len(core_nodes) if core_nodes else 0
                        if len(path) > len(longest_path) or (len(path) == len(longest_path) and coverage > best_coverage):
                            longest_path = path
                            best_coverage = coverage
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
                if path_count >= max_paths_to_check:
                    break
            if path_count >= max_paths_to_check:
                break
        
        # 4. 破循環備案（但只包含核心節點）
        if not longest_path:
            print(" - No simple path. Using DAG topological sort with core nodes only...")
            # 只使用核心節點構建子圖
            if core_nodes:
                core_subgraph = subgraph.subgraph(core_nodes | set(sources) | set(sinks))
                H = nx.DiGraph(core_subgraph)
                while not nx.is_directed_acyclic_graph(H) and len(H.edges()) > 0:
                    try:
                        cycle = nx.find_cycle(H)
                        H.remove_edge(cycle[0][0], cycle[0][1])
                    except:
                        break
                try:
                    topo_nodes = list(nx.topological_sort(H))
                    # 確保起點在前，終點在後
                    if sources and sinks:
                        # 重新排序：起點 -> 中間節點 -> 終點
                        start_nodes = [n for n in topo_nodes if n in sources]
                        end_nodes = [n for n in topo_nodes if n in sinks]
                        middle_nodes = [n for n in topo_nodes if n not in sources and n not in sinks]
                        longest_path = start_nodes + middle_nodes + end_nodes
                    else:
                        longest_path = topo_nodes
                except:
                    # 如果拓撲排序失敗，只使用核心節點
                    longest_path = list(core_nodes)[:20]  # 限制為20個
            else:
                longest_path = list(nodes)[:20]  # 限制為20個
        
        # 5. 路徑過濾：嚴格過濾，只保留核心節點和必要的連接節點
        if len(longest_path) > 1:
            filtered = []
            for i, node in enumerate(longest_path):
                # 核心節點必留
                if node in core_nodes:
                    filtered.append(node)
                # 起點或終點必留
                elif node in sources or node in sinks:
                    filtered.append(node)
                # 中間節點：必須與前後節點都有連接，且是核心節點的鄰居
                elif i > 0 and i < len(longest_path) - 1:
                    prev = longest_path[i-1]
                    next_node = longest_path[i+1]
                    # 必須有連接
                    has_connection = (self.graph.has_edge(prev, node) and self.graph.has_edge(node, next_node))
                    # 必須是核心節點的鄰居（直接相連）
                    is_core_neighbor = any(
                        self.graph.has_edge(node, core) or self.graph.has_edge(core, node)
                        for core in core_nodes
                    )
                    if has_connection and is_core_neighbor:
                        filtered.append(node)
            longest_path = filtered
            
            # 進一步限制：如果路徑太長，只保留核心節點和起點終點
            if len(longest_path) > 15:
                print(f"   ⚠️  Path too long ({len(longest_path)} nodes), limiting to core nodes...")
                core_path = [n for n in longest_path if n in core_nodes]
                # 添加起點和終點
                start_nodes = [n for n in longest_path if n in sources]
                end_nodes = [n for n in longest_path if n in sinks]
                # 保持順序：起點 -> 核心節點 -> 終點
                longest_path = start_nodes + core_path + end_nodes
                # 限制總長度
                if len(longest_path) > 15:
                    longest_path = longest_path[:15]
        
        if longest_path:
            print(f" - Selected Path: {' -> '.join(longest_path)}")
            return longest_path
        
        return list(nodes)


class ModuleAwareWorkflowComposer:
    """
    模組感知的工作流程組合器
    
    使用 A* 算法在知識圖中生成多個候選工作流程。
    """
    
    def __init__(self, domain_graph: DomainKnowledgeGraph, search_agent, ontology: Dict):
        """
        初始化組合器
        
        Args:
            domain_graph: 領域知識圖
            search_agent: 搜索代理（用於獲取匹配的節點）
            ontology: Ontology 字典
        """
        self.graph = domain_graph.graph
        self.domain_graph = domain_graph
        self.search_agent = search_agent
        self.ontology = ontology
    
    def compose(
        self,
        matched_leaves: List[Dict],
        initial_concrete_nodes: List[str],
        user_query: str,
        params: Dict
    ) -> List[Dict]:
        """
        組合工作流程候選（改進版：限制版多候選生成）
        
        Args:
            matched_leaves: MCTS 搜索匹配的葉子節點
            initial_concrete_nodes: 初始具體節點列表
            user_query: 用戶查詢
            params: 參數字典
        
        Returns:
            workflow_candidates: 工作流程候選列表
        """
        print("\nPHASE 5+: Module-Aware Workflow Composition (MCTS + A*) - Multi-Chain Candidates")
        
        # Step 1: 高分葉子
        high_reward_leaves = [
            leaf for leaf in matched_leaves
            if leaf.get('avg_reward', 0) > 0.25 or leaf.get('keyword_matches')
        ]
        if not high_reward_leaves:
            high_reward_leaves = matched_leaves
        
        # Step 2: 自動偵測模組入口（恢復原本邏輯）
        module_entries = self._detect_module_entries(high_reward_leaves, list(initial_concrete_nodes))
        
        workflow_candidates = []
        
        # Step 3: 針對每個入口尋找多條路徑（但加入限制）
        for entry, nodes_in_module in module_entries.items():
            if not entry:
                continue
            
            # 限制節點數量以避免性能問題
            if len(nodes_in_module) > 40:
                print(f"   ⚠️  Module has too many nodes ({len(nodes_in_module)}), limiting to 40...")
                # 優先保留 initial_concrete_nodes
                priority_nodes = set(initial_concrete_nodes) & set(nodes_in_module)
                other_nodes = [n for n in nodes_in_module if n not in priority_nodes]
                nodes_in_module = list(priority_nodes) + other_nodes[:40-len(priority_nodes)]
            
            # 呼叫 A* 搜索，但限制 top_k=5 以避免性能問題
            possible_candidates = self._astar_in_subgraph(nodes_in_module, entry, top_k=5)
            
            for rank, cand_data in enumerate(possible_candidates):
                path = cand_data['path']
                score = cand_data['score']
                coverage_count = cand_data.get('coverage_count', 0)
                
                # 動態產生描述，區分「完整版」與「精簡版」
                desc_suffix = " (High Coverage)" if coverage_count >= 2 else " (Minimal/Efficient)"
                
                candidate = {
                    'type': f'candidate_chain_{rank+1}',
                    'description': f'自動生成路徑 #{rank+1}{desc_suffix} - Score: {score:.2f}',
                    'path': path,
                    'params': self._fill_params(path, params),
                    'score_rank': rank,
                    'metadata': cand_data  # 保留原始 metadata 供後續分析
                }
                workflow_candidates.append(candidate)
        
        if not workflow_candidates:
            print(" - Failed to generate valid workflow candidates.")
            return []
        
        print(f" - Generated {len(workflow_candidates)} candidates.")
        
        return workflow_candidates
    
    def _fill_params(self, path: List[str], params: Dict) -> Dict:
        """
        填充參數
        
        Args:
            path: 節點類型路徑
            params: 參數字典
        
        Returns:
            filled_params: 填充後的參數字典
        """
        filled_params = {}
        
        for node_type in path:
            node_params = {}
            node_ontology = self.ontology.get(node_type, {})
            required_params = node_ontology.get('required_params', [])
            
            # 嘗試從 params 中匹配參數
            for req_param in required_params:
                # 模糊匹配參數名
                matched_value = None
                for param_key, param_value in params.items():
                    if req_param.lower() in param_key.lower() or param_key.lower() in req_param.lower():
                        matched_value = param_value
                        break
                
                if matched_value is not None:
                    node_params[req_param] = matched_value
            
            if node_params:
                filled_params[node_type] = node_params
        
        return filled_params
    
    def _detect_module_entries(self, leaves, concrete_nodes_list):
        """
        自動偵測模組入口（n8n 版本：使用實際的 trigger 節點作為入口）
        
        Args:
            leaves: MCTS 匹配的葉子節點
            concrete_nodes_list: 具體節點列表
        
        Returns:
            entries: {trigger_node: [reachable_nodes]} 字典
        """
        entries = {}
        all_module_nodes = set(concrete_nodes_list)
        
        # Step 1: 找出所有 trigger 節點（以 Trigger 結尾，或包含 trigger 關鍵字）
        trigger_nodes = [
            n for n in concrete_nodes_list 
            if n.lower().endswith('trigger') or 'trigger' in n.lower()
        ]
        
        # 也包含一些常見的 trigger 節點（即使沒有 Trigger 後綴）
        common_triggers = ['webhook', 'manualTrigger', 'scheduleTrigger', 'cron']
        for node in concrete_nodes_list:
            if any(trigger in node.lower() for trigger in common_triggers):
                if node not in trigger_nodes:
                    trigger_nodes.append(node)
        
        if not trigger_nodes:
            print("   - Warning: No trigger nodes found. Using all nodes as fallback.")
            return {None: concrete_nodes_list}
        
        print(f"   - Found {len(trigger_nodes)} trigger nodes: {trigger_nodes[:5]}...")
        
        # Step 2: 找出所有出度為 0 的節點作為終點
        end_nodes = []
        for node in concrete_nodes_list:
            if node in self.graph:
                if self.graph.out_degree(node) == 0:
                    end_nodes.append(node)
        
        # 如果沒有找到出度為 0 的節點，使用常見的終點節點
        if not end_nodes:
            common_ends = ['noOp', 'stopAndError', 'emailSend', 'respondToWebhook']
            for node in concrete_nodes_list:
                if any(end in node for end in common_ends):
                    end_nodes.append(node)
        
        if not end_nodes:
            print("   - Warning: No end nodes found. Will use all nodes as potential ends.")
            end_nodes = list(concrete_nodes_list)
        
        print(f"   - Found {len(end_nodes)} potential end nodes: {end_nodes[:5]}...")
        
        # Step 3: 對每個 trigger 節點，進行雙向 BFS
        for trigger in trigger_nodes[:3]:  # 限制為前 3 個 trigger，避免太多組合
            if trigger not in self.graph:
                continue
            
            # 正向BFS：從 trigger 往後
            queue = deque([trigger])
            visited_forward = {trigger}
            
            while queue:
                curr = queue.popleft()
                for neigh in self.graph.successors(curr):
                    if neigh in all_module_nodes and neigh not in visited_forward:
                        visited_forward.add(neigh)
                        queue.append(neigh)
            
            # 反向BFS：從所有終點往前
            visited_backward = set()
            for end_node in end_nodes[:5]:  # 限制為前 5 個終點
                if end_node not in self.graph:
                    continue
                queue = deque([end_node])
                visited_backward.add(end_node)
                
                while queue:
                    curr = queue.popleft()
                    for pred in self.graph.predecessors(curr):
                        if pred in all_module_nodes and pred not in visited_backward:
                            visited_backward.add(pred)
                            queue.append(pred)
            
            # 取交集：必須同時從 trigger 可達 AND 可達終點
            reachable_nodes = visited_forward & visited_backward
            
            # 如果交集為空，至少使用正向可達的節點
            if not reachable_nodes:
                print(f"   - Warning: No nodes reachable from both {trigger} and end nodes. Using forward-reachable nodes.")
                reachable_nodes = visited_forward
            
            # 如果還是為空，使用所有節點
            if not reachable_nodes:
                print(f"   - Warning: No reachable nodes found for {trigger}. Using all module nodes.")
                reachable_nodes = all_module_nodes
            
            entries[trigger] = list(reachable_nodes)
            print(f"   - Entry [{trigger}]: {len(reachable_nodes)} reachable nodes")
        
        if not entries:
            print("   - Warning: No valid entries found. Using all nodes as fallback.")
            return {None: concrete_nodes_list}
        
        return entries
    
    def _astar_in_subgraph(self, nodes, start_node, top_k=5):
        """
        在子圖中尋找從 start_node 到潛在終點的【多條】路徑（限制版）
        
        Args:
            nodes: 候選節點列表
            start_node: 起始節點
            top_k: 返回前 k 個最佳候選（預設 5，避免性能問題）
        
        Returns:
            candidates: 候選路徑列表，每個包含 path, score, coverage_count 等
        """
        if not nodes or not start_node or start_node not in nodes:
            return []
        
        # 建立子圖 (只包含候選節點，確保路徑只走這些點)
        subgraph = self.graph.subgraph(nodes)
        
        # 確定潛在終點（n8n 版本：使用出度為 0 的節點）
        potential_ends = set()
        
        # 加入所有出度為 0 的節點作為備選終點
        for n in nodes:
            if subgraph.out_degree(n) == 0:
                potential_ends.add(n)
        
        # 如果沒有找到，使用常見的終點節點
        if not potential_ends:
            common_ends = ['noOp', 'stopAndError', 'emailSend', 'respondToWebhook', 'googleCalendar']
            for n in nodes:
                if any(end in n for end in common_ends):
                    potential_ends.add(n)
        
        if not potential_ends:
            print(" - Warning: No valid end nodes identified.")
            return []
        
        # 定義關鍵節點 (Critical Nodes) - n8n 版本
        # 使用 n8n 相關的關鍵字：function, code, if, switch, merge, openai, agent 等
        critical_keywords = [
            'function', 'code', 'if', 'switch', 'merge', 
            'openai', 'agent', 'llm', 'chat', 'embedding',
            'extract', 'transform', 'filter', 'aggregate'
        ]
        required_nodes_in_subgraph = [
            n for n in nodes 
            if any(k.lower() in n.lower() for k in critical_keywords) 
            and n != start_node 
            and n not in potential_ends
        ]
        print(f" - Target Critical Nodes ({len(required_nodes_in_subgraph)}): {required_nodes_in_subgraph[:5]}...")
        
        candidates = []
        
        # 限制：只檢查前 1 個終點，避免組合爆炸（進一步限制）
        checked_ends = list(potential_ends)[:1]
        
        # 使用 NetworkX 尋找簡單路徑（嚴格限制深度和數量）
        for end_node in checked_ends:
            if start_node == end_node:
                continue
            
            try:
                # 進一步限制路徑深度為 6（減少搜索空間）
                # 使用 generator 並立即限制，避免生成太多路徑
                raw_paths = nx.all_simple_paths(self.graph, start_node, end_node, cutoff=6)
                
                path_count = 0
                max_paths_per_end = 10  # 大幅減少：每個終點最多檢查 10 條路徑
                
                for path in raw_paths:
                    if path_count >= max_paths_per_end:
                        break
                    
                    # 過濾：確保路徑只經過候選節點
                    if not all(n in nodes or n == end_node for n in path):
                        continue
                    
                    # 限制路徑長度（避免過長的路徑）
                    if len(path) > 10:
                        continue
                    
                    # 計算分數
                    coverage = set(path) & set(required_nodes_in_subgraph)
                    score = len(coverage) * 2.0 - (len(path) * 0.1)
                    
                    candidates.append({
                        'path': path,
                        'score': score,
                        'coverage': list(coverage),
                        'coverage_count': len(coverage),
                        'length': len(path)
                    })
                    path_count += 1
                    
                    # 如果已經有足夠的候選，提前退出
                    if len(candidates) >= top_k * 2:  # 收集比需要的多一點，以便排序
                        break
                        
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                print(f"   - Error finding paths to {end_node}: {e}")
                continue
        
        # Fallback: 拓樸排序 (如果沒有連通路徑)
        if not candidates:
            print(" - No connected path found via graph traversal. Attempting topological sort fallback.")
            try:
                topo_path = list(nx.topological_sort(subgraph))
                if start_node in topo_path:
                    start_idx = topo_path.index(start_node)
                    valid_topo = topo_path[start_idx:]
                    # 限制拓撲排序路徑長度
                    if len(valid_topo) > 15:
                        valid_topo = valid_topo[:15]
                    candidates.append({
                        'path': valid_topo,
                        'score': 0,
                        'coverage': [],
                        'coverage_count': 0,
                        'length': len(valid_topo)
                    })
            except nx.NetworkXUnfeasible:
                pass
        
        # 排序：優先覆蓋率高，其次路徑短
        candidates.sort(key=lambda x: (x['score'], -x['length']), reverse=True)
        
        # 限制返回數量
        final_result = candidates[:top_k] if top_k else candidates[:5]  # 預設最多 5 個
        
        if final_result:
            print(f" - Found {len(candidates)} total valid paths. Returning {len(final_result)} candidates.")
            for i, c in enumerate(final_result):
                path_preview = ' -> '.join(c['path'][:3]) + ('...' if len(c['path']) > 3 else '')
                print(f"   Option {i+1}: {path_preview} (Score: {c['score']:.2f}, Coverage: {c['coverage_count']}, Length: {c['length']})")
        else:
            print(f" - No valid paths found from {start_node} to any end nodes.")
        
        return final_result

