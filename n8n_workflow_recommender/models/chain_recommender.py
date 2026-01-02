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
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        對候選列表進行評分和排序
        
        Args:
            candidates_list: 候選列表，每個元素是包含 'path' 鍵的字典
            strategy: 評分策略
            min_score: 最小分數閾值
        
        Returns:
            ranked_candidates: 排序後的候選列表，每個元素添加了 'mf_score' 和 'final_combined_score'
        """
        if not candidates_list:
            return []
        
        print(f"\n[Scoring] Running Matrix Factorization scoring for {len(candidates_list)} candidates...")
        
        scored_candidates = []
        for candidate in candidates_list:
            # 提取路徑
            chain_path = candidate.get('path', [])
            
            if not chain_path:
                continue
            
            # 計算分數
            mf_score = self.scorer.score_chain(chain_path, strategy=strategy)
            
            # 如果分數低於閾值，跳過
            if mf_score < min_score:
                continue
            
            # 將分數寫回物件
            candidate_copy = candidate.copy()
            candidate_copy['mf_score'] = mf_score
            candidate_copy['final_combined_score'] = mf_score  # 暫時以 MF 分數為主，未來可做加權混合
            
            scored_candidates.append(candidate_copy)
        
        # 按分數降序排序
        scored_candidates.sort(key=lambda x: x.get('final_combined_score', 0.0), reverse=True)
        
        print(f"[Scoring] Ranked {len(scored_candidates)} candidates (min_score={min_score})")
        if scored_candidates:
            print(f"[Scoring] Top score: {scored_candidates[0].get('final_combined_score', 0.0):.4f}")
        
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

