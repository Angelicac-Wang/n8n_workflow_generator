#!/usr/bin/env python3
"""
矩陣分解評分器

加載預訓練的 type_to_type 矩陣分解模型，提供鏈評分和候選排序功能。
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class MatrixFactorizationScorer:
    """
    矩陣分解評分器
    
    使用預訓練的矩陣分解模型對節點類型鏈進行評分。
    """
    
    def __init__(self, model_dir: str):
        """
        初始化評分器
        
        Args:
            model_dir: 預訓練模型目錄路徑
        """
        self.model_dir = Path(model_dir)
        self.prediction_matrix: Optional[np.ndarray] = None
        self.mapping: Optional[Dict] = None
        self.node_type_to_index: Dict[str, int] = {}
        self.index_to_node_type: Dict[int, str] = {}
        self.node_types: List[str] = []
        
        self.load_model()
    
    def load_model(self):
        """載入預訓練模型"""
        print(f"📥 載入矩陣分解模型: {self.model_dir}")
        
        # 載入映射
        mapping_path = self.model_dir / "type_type_mapping.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"映射檔案不存在: {mapping_path}")
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.mapping = json.load(f)
        
        self.node_types = self.mapping.get("node_types", [])
        self.node_type_to_index = self.mapping.get("node_type_to_index", {})
        self.index_to_node_type = {
            int(k): v for k, v in self.mapping.get("index_to_node_type", {}).items()
        }
        
        print(f"   ✅ 載入了 {len(self.node_types)} 個節點類型的映射")
        
        # 載入預測矩陣
        prediction_matrix_path = self.model_dir / "type_type_prediction_matrix.npy"
        if prediction_matrix_path.exists():
            self.prediction_matrix = np.load(prediction_matrix_path)
            print(f"   ✅ 載入了預測矩陣: {self.prediction_matrix.shape}")
        else:
            # 如果沒有預測矩陣，嘗試從 P 和 Q 矩陣重建
            p_path = self.model_dir / "type_type_P.npy"
            q_path = self.model_dir / "type_type_Q.npy"
            
            if p_path.exists() and q_path.exists():
                P = np.load(p_path)
                Q = np.load(q_path)
                self.prediction_matrix = np.dot(P, Q.T)
                print(f"   ✅ 從 P 和 Q 矩陣重建預測矩陣: {self.prediction_matrix.shape}")
            else:
                raise FileNotFoundError(
                    f"找不到預測矩陣或 P/Q 矩陣: {prediction_matrix_path}, {p_path}, {q_path}"
                )
    
    def get_transition_score(self, source_type: str, target_type: str) -> float:
        """
        獲取兩個節點類型之間的轉換分數
        
        Args:
            source_type: 源節點類型
            target_type: 目標節點類型
        
        Returns:
            score: 轉換分數 (0.0 - 1.0)
        """
        if self.prediction_matrix is None:
            return 0.0
        
        # 檢查節點類型是否存在
        if source_type not in self.node_type_to_index or target_type not in self.node_type_to_index:
            return 0.0
        
        source_idx = self.node_type_to_index[source_type]
        target_idx = self.node_type_to_index[target_type]
        
        # 獲取分數
        score = self.prediction_matrix[source_idx, target_idx]
        
        # 修剪分數到 [0, 1] 範圍
        score = max(0.0, min(1.0, score))
        
        return float(score)
    
    def score_chain(self, chain: List[str], strategy: str = 'sum') -> float:
        """
        計算節點類型鏈的總分
        
        Args:
            chain: 節點類型列表
            strategy: 評分策略
                - 'sum': 簡單相加（適合短路徑）
                - 'product': 相乘（適合長路徑，會衰減）
                - 'average': 平均值
                - 'geometric_mean': 幾何平均數
                - 'min': 取最小值（最弱連結）
        
        Returns:
            score: 鏈的總分
        """
        if len(chain) < 2:
            return 0.0
        
        # 獲取每對相鄰節點的分數
        edge_scores = []
        for i in range(len(chain) - 1):
            source = chain[i]
            target = chain[i + 1]
            score = self.get_transition_score(source, target)
            edge_scores.append(score)
        
        if not edge_scores:
            return 0.0
        
        # 根據策略計算總分
        if strategy == 'sum':
            return sum(edge_scores)
        elif strategy == 'product':
            return float(np.prod(edge_scores))
        elif strategy == 'average':
            return float(np.mean(edge_scores))
        elif strategy == 'geometric_mean':
            return float(np.prod(edge_scores) ** (1.0 / len(edge_scores)))
        elif strategy == 'min':
            return min(edge_scores)
        else:
            raise ValueError(f"未知的評分策略: {strategy}")
    
    def rank_candidates(
        self,
        candidates: List[List[str]],
        strategy: str = 'sum',
        min_score: float = 0.0
    ) -> List[Tuple[List[str], float]]:
        """
        對候選鏈進行評分和排序
        
        Args:
            candidates: 候選鏈列表
            strategy: 評分策略
            min_score: 最小分數閾值
        
        Returns:
            ranked_candidates: 排序後的候選列表，每個元素是 (chain, score) 元組
        """
        scored_candidates = []
        
        for chain in candidates:
            score = self.score_chain(chain, strategy)
            if score >= min_score:
                scored_candidates.append((chain, score))
        
        # 按分數降序排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def get_top_transitions(self, source_type: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        獲取從指定節點類型出發的最佳轉換
        
        Args:
            source_type: 源節點類型
            top_k: 返回前 k 個
        
        Returns:
            top_transitions: [(target_type, score), ...]
        """
        if self.prediction_matrix is None or source_type not in self.node_type_to_index:
            return []
        
        source_idx = self.node_type_to_index[source_type]
        scores = self.prediction_matrix[source_idx, :]
        
        # 獲取 top_k 個最高分
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        top_transitions = []
        for idx in top_indices:
            target_type = self.index_to_node_type.get(idx, f"unknown_{idx}")
            score = float(max(0.0, min(1.0, scores[idx])))
            top_transitions.append((target_type, score))
        
        return top_transitions


def main():
    """主函數：測試評分器"""
    # 設定路徑（相對於專案根目錄）
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_dir = base_dir / "scripts" / "recommendation_matrix" / "type_type_factorization" / "with_validation"
    
    print("=" * 80)
    print("Matrix Factorization Scorer Test")
    print("=" * 80)
    
    # 載入模型
    print(f"\n📥 載入模型: {model_dir}")
    scorer = MatrixFactorizationScorer(str(model_dir))
    
    # 測試轉換分數
    print("\n🧪 測試轉換分數:")
    test_pairs = [
        ("n8n-nodes-base.manualTrigger", "n8n-nodes-base.httpRequest"),
        ("n8n-nodes-base.httpRequest", "n8n-nodes-base.set"),
        ("n8n-nodes-base.set", "n8n-nodes-base.if")
    ]
    
    for source, target in test_pairs:
        score = scorer.get_transition_score(source, target)
        print(f"   - {source} -> {target}: {score:.4f}")
    
    # 測試鏈評分
    print("\n🧪 測試鏈評分:")
    test_chain = [
        "n8n-nodes-base.manualTrigger",
        "n8n-nodes-base.httpRequest",
        "n8n-nodes-base.set"
    ]
    
    for strategy in ['sum', 'average', 'min']:
        score = scorer.score_chain(test_chain, strategy)
        print(f"   - {strategy}: {score:.4f}")
    
    # 測試 top transitions
    print("\n🧪 測試 Top Transitions:")
    source = "n8n-nodes-base.manualTrigger"
    top_transitions = scorer.get_top_transitions(source, top_k=5)
    for target, score in top_transitions:
        print(f"   - {source} -> {target}: {score:.4f}")
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

