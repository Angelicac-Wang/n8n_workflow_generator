#!/usr/bin/env python3
"""
參數填充器

根據 NLU 提取的參數填充節點配置。
"""

from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher


class ParameterFiller:
    """
    參數填充器
    
    根據 NLU 提取的參數和節點的 required_params 填充節點配置。
    """
    
    def __init__(self, ontology: Dict):
        """
        初始化參數填充器
        
        Args:
            ontology: Ontology 字典 {node_type: {"required_params": [...], ...}}
        """
        self.ontology = ontology
    
    def fill_parameters(
        self,
        node_type: str,
        extracted_params: Dict[str, Any],
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        為單個節點填充參數
        
        Args:
            node_type: 節點類型
            extracted_params: NLU 提取的參數字典
            strict: 是否嚴格模式（只填充 required_params）
        
        Returns:
            filled_params: 填充後的參數字典
        """
        node_ontology = self.ontology.get(node_type, {})
        required_params = node_ontology.get("required_params", [])
        
        filled_params = {}
        
        # 如果 strict=True，只填充 required_params
        target_params = required_params if strict else list(extracted_params.keys())
        
        for param_name in target_params:
            # 嘗試精確匹配
            if param_name in extracted_params:
                filled_params[param_name] = extracted_params[param_name]
                continue
            
            # 嘗試模糊匹配
            best_match = self._fuzzy_match_param(param_name, extracted_params)
            if best_match:
                filled_params[param_name] = extracted_params[best_match]
                continue
            
            # 如果找不到匹配，使用默認值或留空
            if param_name in required_params:
                # 對於 required 參數，提供一個占位符
                filled_params[param_name] = f"<NEEDS_VALUE: {param_name}>"
        
        return filled_params
    
    def fill_chain_parameters(
        self,
        node_type_chain: List[str],
        extracted_params: Dict[str, Any],
        strict: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        為整個節點鏈填充參數
        
        Args:
            node_type_chain: 節點類型列表
            extracted_params: NLU 提取的參數字典
            strict: 是否嚴格模式
        
        Returns:
            chain_params: {node_type: {param: value}}
        """
        chain_params = {}
        
        for node_type in node_type_chain:
            node_params = self.fill_parameters(node_type, extracted_params, strict)
            if node_params:
                chain_params[node_type] = node_params
        
        return chain_params
    
    def _fuzzy_match_param(self, param_name: str, params_dict: Dict[str, Any]) -> Optional[str]:
        """
        模糊匹配參數名稱
        
        Args:
            param_name: 目標參數名稱
            params_dict: 參數字典
        
        Returns:
            matched_key: 匹配的鍵，如果沒有則返回 None
        """
        param_lower = param_name.lower()
        best_match = None
        best_score = 0.0
        
        for key in params_dict.keys():
            key_lower = key.lower()
            
            # 計算相似度
            similarity = SequenceMatcher(None, param_lower, key_lower).ratio()
            
            # 檢查是否包含
            if param_lower in key_lower or key_lower in param_lower:
                similarity += 0.3
            
            if similarity > best_score and similarity > 0.6:  # 閾值
                best_score = similarity
                best_match = key
        
        return best_match
    
    def validate_parameters(self, node_type: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        驗證參數是否完整
        
        Args:
            node_type: 節點類型
            params: 參數字典
        
        Returns:
            (is_valid, missing_params): (是否有效, 缺失的參數列表)
        """
        node_ontology = self.ontology.get(node_type, {})
        required_params = node_ontology.get("required_params", [])
        
        missing_params = []
        for req_param in required_params:
            if req_param not in params or params[req_param] is None:
                missing_params.append(req_param)
        
        return len(missing_params) == 0, missing_params

