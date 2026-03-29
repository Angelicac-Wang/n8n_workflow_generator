#!/usr/bin/env python3
"""
鏈推薦器

適配 ChainRecommender 類，使用預訓練的矩陣分解模型進行評分和排序。
"""

from typing import List, Dict, Optional
from .matrix_factorization_scorer import MatrixFactorizationScorer


class ChainRecommender:
    """
    鏈推薦器
    
    使用預訓練的矩陣分解模型對候選工作流程進行評分和排序。
    """
    
    def __init__(self, model_dir: str, use_type_model: bool = True):
        """
        初始化推薦器
        
        Args:
            model_dir: 預訓練模型目錄路徑
            use_type_model: 是否使用 type_to_type 模型（True）或 name_to_name 模型（False）
        """
        self.model_dir = model_dir
        self.use_type_model = use_type_model
        self.scorer = MatrixFactorizationScorer(model_dir)
    
    def score_chain(self, chain: List[str], verbose: bool = False) -> float:
        """
        對單一鏈進行評分
        
        Args:
            chain: 節點類型或名稱列表
            verbose: 是否輸出詳細資訊
        
        Returns:
            score: 鏈的平均分數
        """
        if len(chain) < 2:
            return 0.0
        
        if verbose:
            print(f"\n--- 評估 Chain: {chain} ---")
        
        # 使用 average 策略計算平均分數
        score = self.scorer.score_chain(chain, strategy='average')
        
        if verbose:
            print(f">> Chain 平均分數: {score:.4f}")
        
        return score
    
    def rank_candidates(
        self,
        candidates_list: List[Dict],
        strategy: str = 'average',
        min_score: float = 0.0,
        critical_node_weight: float = 0.5
    ) -> List[Dict]:
        """
        [IMPROVED] 混合 MF 分數與 Critical Node 覆蓋率進行排序。
        
        Args:
            candidates_list: 候選列表，每個元素是包含 'path' 鍵的字典
            strategy: 評分策略
            min_score: 最小分數閾值
            critical_node_weight: Critical Node 覆蓋率的權重（預設 0.5）
        
        Returns:
            ranked_candidates: 排序後的候選列表，每個元素添加了多個分數欄位
        """
        if not candidates_list:
            return []
        
        print(f"\n[Scoring] Running MF scoring with Critical Node awareness...")
        
        # === 定義 Critical Nodes ===
        critical_nodes = [
            'n8n-nodes-base.httpRequest',
            '@n8n/n8n-nodes-langchain.agent',
            'n8n-nodes-base.set',
            'n8n-nodes-base.code',
            'n8n-nodes-base.if'
        ]
        
        scored_candidates = []
        for candidate in candidates_list:
            chain_path = candidate.get('path', [])
            
            if not chain_path:
                continue
            
            # 1. MF 分數 (0.0 ~ 1.0)
            mf_score = self.scorer.score_chain(chain_path, strategy=strategy)
            
            # 如果分數低於閾值，跳過
            if mf_score < min_score:
                continue
            
            # 2. 計算 Critical Node 覆蓋率
            # 檢查路徑中有多少節點是 critical nodes
            critical_nodes_in_path = [
                n for n in chain_path 
                if n in critical_nodes
            ]
            
            # 從 metadata 獲取原始 coverage 資訊（如果有的話）
            metadata = candidate.get('metadata', {})
            coverage_count = metadata.get('coverage_count', len(critical_nodes_in_path))
            
            # 正規化覆蓋率（假設最多 5 個 critical nodes）
            max_critical = len(critical_nodes)
            coverage_score = min(1.0, coverage_count / max_critical)
            
            # 3. 混合分數
            # 如果沒有覆蓋任何 critical node，給予懲罰
            if coverage_count == 0:
                penalty = 0.5  # 嚴重懲罰
                combined_score = mf_score * (1 - penalty)
            else:
                combined_score = (1 - critical_node_weight) * mf_score + critical_node_weight * coverage_score
            
            candidate_copy = candidate.copy()
            candidate_copy['mf_score'] = mf_score
            candidate_copy['coverage_count'] = coverage_count
            candidate_copy['coverage_score'] = coverage_score
            candidate_copy['final_combined_score'] = combined_score
            
            scored_candidates.append(candidate_copy)
        
        # 4. 依照混合分數排序
        scored_candidates.sort(key=lambda x: x['final_combined_score'], reverse=True)
        
        # 5. 詳細輸出
        print(f"[Scoring] Ranking results:")
        for i, c in enumerate(scored_candidates):
            print(f"  #{i+1}: MF={c['mf_score']:.2f}, Coverage={c['coverage_count']}, Combined={c['final_combined_score']:.2f}")
            path_preview = ' -> '.join(c['path'][:3]) + ('...' if len(c['path']) > 3 else '')
            print(f"       Path: {path_preview}")
        
        return scored_candidates
    
    def batch_score_chains(self, list_of_chains: List[List[str]]) -> List[Dict]:
        """
        批次評分多條鏈
        
        Args:
            list_of_chains: 鏈列表
        
        Returns:
            results: 評分結果列表，每個元素是 {"chain": [...], "score": float}
        """
        print(f"\n==========================================")
        print(f"執行批次評分 (共 {len(list_of_chains)} 條)")
        print(f"==========================================")
        
        results = []
        for chain in list_of_chains:
            score = self.score_chain(chain, verbose=False)
            results.append({
                "chain": chain,
                "score": score
            })
        
        # 按分數降序排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print("評分排名 (前 5 名):")
        for i, result in enumerate(results[:5], 1):
            print(f"  {i}. {result['chain']}: {result['score']:.4f}")
        
        return results
    
    def get_transition_score(self, source: str, target: str) -> float:
        """
        獲取兩個節點之間的轉換分數
        
        Args:
            source: 源節點類型或名稱
            target: 目標節點類型或名稱
        
        Returns:
            score: 轉換分數
        """
        return self.scorer.get_transition_score(source, target)
    
    def get_top_transitions(self, source: str, top_k: int = 10) -> List[tuple]:
        """
        獲取從指定節點出發的最佳轉換
        
        Args:
            source: 源節點類型或名稱
            top_k: 返回前 k 個
        
        Returns:
            top_transitions: [(target, score), ...]
        """
        return self.scorer.get_top_transitions(source, top_k)

