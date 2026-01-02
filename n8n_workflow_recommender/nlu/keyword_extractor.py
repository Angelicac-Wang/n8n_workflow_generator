#!/usr/bin/env python3
"""
關鍵字提取器

從用戶查詢中提取關鍵字，用於 MCTS 搜索的關鍵字匹配。
"""

from typing import Set, List, Optional
import re


class KeywordExtractor:
    """
    關鍵字提取器
    
    從文本中提取有意義的關鍵字。
    """
    
    def __init__(self, stop_words: Optional[List[str]] = None):
        """
        初始化提取器
        
        Args:
            stop_words: 停用詞列表
        """
        self.stop_words = set(stop_words) if stop_words else self._default_stop_words()
    
    def _default_stop_words(self) -> Set[str]:
        """默認停用詞"""
        return {
            '的', '是', '在', '有', '和', '與', '或', '要', '我', '你', '他', '她', '它',
            'the', 'is', 'are', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'at',
            'for', 'with', 'from', 'by', 'as', 'this', 'that', 'these', 'those',
            'will', 'would', 'should', 'could', 'can', 'may', 'might'
        }
    
    def extract(self, text: str, min_length: int = 2) -> Set[str]:
        """
        從文本中提取關鍵字
        
        Args:
            text: 輸入文本
            min_length: 最小關鍵字長度
        
        Returns:
            keywords: 關鍵字集合
        """
        keywords = set()
        
        # 轉為小寫
        text_lower = text.lower()
        
        # 提取技術術語（如 OCR, API, JSON 等）
        tech_terms = re.findall(r'\b[A-Z]{2,}\b', text)
        keywords.update([term.lower() for term in tech_terms])
        
        # 簡單分詞（按空格和標點符號）
        words = re.findall(r'\b\w+\b', text_lower)
        
        for word in words:
            # 過濾停用詞和短詞
            if len(word) >= min_length and word not in self.stop_words:
                keywords.add(word)
        
        # 提取中文詞組（2-4 個字）
        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        for word in chinese_words:
            if word not in self.stop_words:
                keywords.add(word.lower())
        
        return keywords
    
    def extract_technical_terms(self, text: str) -> Set[str]:
        """
        提取技術術語（如 API、SMS、OCR 等）
        
        Args:
            text: 輸入文本
        
        Returns:
            technical_terms: 技術術語集合
        """
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # 大寫縮寫（如 SMS, API, OCR）
            r'\b\w+\.\w+\b',   # 點分隔的術語（如 n8n-nodes-base.httpRequest）
        ]
        
        terms = set()
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            terms.update(matches)
        
        return terms

