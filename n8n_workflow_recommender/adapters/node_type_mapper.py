#!/usr/bin/env python3
"""
節點類型映射管理

管理節點名稱與節點類型的雙向映射，提供查詢和轉換功能。
處理未知節點的 fallback 邏輯。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict, Counter


class NodeTypeMapper:
    """
    節點類型映射器
    
    提供節點名稱和節點類型之間的雙向轉換。
    """
    
    def __init__(self, mappings_path: Optional[str] = None):
        """
        初始化映射器
        
        Args:
            mappings_path: 節點映射 JSON 檔案路徑（可選）
        """
        self.name_to_type: Dict[str, str] = {}
        self.type_to_names: Dict[str, List[str]] = defaultdict(list)
        self.name_frequency: Dict[str, int] = {}
        self.type_frequency: Dict[str, int] = {}
        
        if mappings_path:
            self.load_mappings(mappings_path)
    
    def load_mappings(self, mappings_path: str):
        """
        從 JSON 檔案載入映射
        
        Args:
            mappings_path: 映射檔案路徑
        """
        mappings_path = Path(mappings_path)
        
        if not mappings_path.exists():
            raise FileNotFoundError(f"映射檔案不存在: {mappings_path}")
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.name_to_type = data.get("name_to_type", {})
        self.type_to_names = data.get("type_to_names", {})
        self.name_frequency = data.get("name_frequency", {})
        self.type_frequency = data.get("type_frequency", {})
        
        # 確保反向映射的一致性
        self._rebuild_type_to_names()
    
    def _rebuild_type_to_names(self):
        """重建反向映射以確保一致性"""
        self.type_to_names = defaultdict(list)
        for name, node_type in self.name_to_type.items():
            self.type_to_names[node_type].append(name)
        
        # 去重
        self.type_to_names = {k: list(set(v)) for k, v in self.type_to_names.items()}
    
    def get_type(self, node_name: str) -> Optional[str]:
        """
        獲取節點名稱對應的節點類型
        
        Args:
            node_name: 節點名稱
        
        Returns:
            node_type: 節點類型，如果找不到則返回 None
        """
        return self.name_to_type.get(node_name)
    
    def get_names(self, node_type: str) -> List[str]:
        """
        獲取節點類型對應的所有節點名稱
        
        Args:
            node_type: 節點類型
        
        Returns:
            node_names: 節點名稱列表
        """
        return self.type_to_names.get(node_type, [])
    
    def get_most_common_name(self, node_type: str) -> Optional[str]:
        """
        獲取節點類型最常用的節點名稱
        
        Args:
            node_type: 節點類型
        
        Returns:
            node_name: 最常用的節點名稱，如果找不到則返回 None
        """
        names = self.get_names(node_type)
        if not names:
            return None
        
        # 根據頻率排序
        name_freqs = [(name, self.name_frequency.get(name, 0)) for name in names]
        name_freqs.sort(key=lambda x: x[1], reverse=True)
        
        return name_freqs[0][0] if name_freqs else None
    
    def convert_chain_to_types(self, name_chain: List[str]) -> List[str]:
        """
        將節點名稱鏈轉換為節點類型鏈
        
        Args:
            name_chain: 節點名稱列表
        
        Returns:
            type_chain: 節點類型列表（未知節點保持原名）
        """
        type_chain = []
        for name in name_chain:
            node_type = self.get_type(name)
            type_chain.append(node_type if node_type else name)
        return type_chain
    
    def convert_chain_to_names(self, type_chain: List[str]) -> List[str]:
        """
        將節點類型鏈轉換為節點名稱鏈
        
        Args:
            type_chain: 節點類型列表
        
        Returns:
            name_chain: 節點名稱列表（使用最常用的名稱）
        """
        name_chain = []
        for node_type in type_chain:
            name = self.get_most_common_name(node_type)
            name_chain.append(name if name else node_type)
        return name_chain
    
    def add_mapping(self, node_name: str, node_type: str, frequency: int = 1):
        """
        添加映射關係
        
        Args:
            node_name: 節點名稱
            node_type: 節點類型
            frequency: 使用頻率（用於選擇最常用名稱）
        """
        # 如果已存在映射且類型不同，選擇頻率更高的
        existing_type = self.name_to_type.get(node_name)
        if existing_type and existing_type != node_type:
            existing_freq = self.name_frequency.get(node_name, 0)
            if frequency > existing_freq:
                # 移除舊映射
                if existing_type in self.type_to_names:
                    self.type_to_names[existing_type] = [
                        n for n in self.type_to_names[existing_type] if n != node_name
                    ]
                self.name_to_type[node_name] = node_type
                self.name_frequency[node_name] = frequency
        else:
            self.name_to_type[node_name] = node_type
            self.name_frequency[node_name] = frequency
        
        # 更新反向映射
        if node_type not in self.type_to_names[node_type] or node_name not in self.type_to_names[node_type]:
            self.type_to_names[node_type].append(node_name)
            self.type_to_names[node_type] = list(set(self.type_to_names[node_type]))
        
        # 更新類型頻率
        self.type_frequency[node_type] = self.type_frequency.get(node_type, 0) + frequency
    
    def get_statistics(self) -> Dict:
        """
        獲取映射統計資訊
        
        Returns:
            stats: 統計資訊字典
        """
        return {
            "total_mappings": len(self.name_to_type),
            "unique_types": len(self.type_to_names),
            "most_common_names": dict(Counter(self.name_frequency).most_common(20)),
            "most_common_types": dict(Counter(self.type_frequency).most_common(20))
        }
    
    def save_mappings(self, output_path: str):
        """
        保存映射到 JSON 檔案
        
        Args:
            output_path: 輸出檔案路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "name_to_type": self.name_to_type,
            "type_to_names": dict(self.type_to_names),
            "name_frequency": self.name_frequency,
            "type_frequency": self.type_frequency,
            "statistics": self.get_statistics()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 映射已保存到: {output_path}")


def main():
    """主函數：測試映射器"""
    # 設定路徑（相對於專案根目錄）
    base_dir = Path(__file__).resolve().parent.parent.parent
    mappings_path = base_dir / "n8n_workflow_recommender" / "data" / "node_mappings.json"
    
    print("=" * 80)
    print("Node Type Mapper Test")
    print("=" * 80)
    
    # 載入映射
    print(f"\n📥 載入映射: {mappings_path}")
    mapper = NodeTypeMapper(str(mappings_path))
    
    # 顯示統計
    stats = mapper.get_statistics()
    print(f"\n📊 映射統計:")
    print(f"   - 總映射數: {stats['total_mappings']}")
    print(f"   - 唯一類型數: {stats['unique_types']}")
    
    # 測試轉換
    print("\n🧪 測試轉換:")
    test_names = ["On clicking 'execute'", "Discord", "HTTP Request"]
    for name in test_names:
        node_type = mapper.get_type(name)
        print(f"   - {name} -> {node_type}")
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

