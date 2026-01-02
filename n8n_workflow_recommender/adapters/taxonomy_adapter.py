#!/usr/bin/env python3
"""
適配 n8n taxonomy 到 MCTS 搜索格式

將 n8n 的 taxonomy_full.json 轉換為 TaxonomySearchAgent 可用的格式。
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_taxonomy(taxonomy_path: str) -> Dict:
    """
    載入 taxonomy JSON 檔案
    
    Args:
        taxonomy_path: taxonomy 檔案路徑
    
    Returns:
        taxonomy: taxonomy 數據字典
    """
    taxonomy_path = Path(taxonomy_path)
    
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"Taxonomy 檔案不存在: {taxonomy_path}")
    
    with open(taxonomy_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def convert_taxonomy_to_mcts_format(taxonomy: Dict) -> Dict:
    """
    將 n8n taxonomy 轉換為 MCTS 搜索可用的格式
    
    Args:
        taxonomy: n8n taxonomy 數據字典
    
    Returns:
        mcts_taxonomy: MCTS 格式的 taxonomy
    """
    def process_node(node_key: str, node_content: Any, parent_path: List[str] = None) -> Dict:
        """
        遞歸處理 taxonomy 節點
        
        Args:
            node_key: 節點鍵名
            node_content: 節點內容（可能是字典或包含 Nodes/Description 的字典）
            parent_path: 父節點路徑
        
        Returns:
            processed_node: 處理後的節點字典
        """
        if parent_path is None:
            parent_path = []
        
        current_path = parent_path + [node_key]
        
        # 檢查是否為葉子節點（包含 Nodes 欄位）
        if isinstance(node_content, dict) and "Nodes" in node_content:
            # 這是葉子節點
            return {
                "name": node_key,
                "description": node_content.get("Description", ""),
                "mapped_nodes": node_content.get("Nodes", []),
                "is_leaf": True,
                "path": current_path
            }
        elif isinstance(node_content, dict):
            # 這是中間節點，遞歸處理子節點
            children = {}
            description = node_content.get("Description", "")
            
            for child_key, child_content in node_content.items():
                if child_key == "Description":
                    continue
                children[child_key] = process_node(child_key, child_content, current_path)
            
            return {
                "name": node_key,
                "description": description,
                "mapped_nodes": [],  # 中間節點沒有 mapped_nodes
                "is_leaf": False,
                "path": current_path,
                "children": children
            }
        else:
            # 未知格式
            return {
                "name": node_key,
                "description": "",
                "mapped_nodes": [],
                "is_leaf": True,
                "path": current_path
            }
    
    # 獲取 Taxonomy 根節點
    taxonomy_root = taxonomy.get("Taxonomy", {})
    
    if not taxonomy_root:
        raise ValueError("Taxonomy 中沒有找到 'Taxonomy' 根節點")
    
    # 處理所有頂層分類
    mcts_taxonomy = {}
    for top_level_key, top_level_content in taxonomy_root.items():
        mcts_taxonomy[top_level_key] = process_node(top_level_key, top_level_content)
    
    return mcts_taxonomy


def extract_leaf_nodes(mcts_taxonomy: Dict) -> List[Dict]:
    """
    提取所有葉子節點（包含 mapped_nodes 的節點）
    
    Args:
        mcts_taxonomy: MCTS 格式的 taxonomy
    
    Returns:
        leaf_nodes: 葉子節點列表
    """
    leaf_nodes = []
    
    def traverse(node: Dict):
        if node.get("is_leaf", False) and node.get("mapped_nodes"):
            leaf_nodes.append(node)
        
        for child in node.get("children", {}).values():
            traverse(child)
    
    for root_node in mcts_taxonomy.values():
        traverse(root_node)
    
    return leaf_nodes


def build_node_type_to_category_mapping(mcts_taxonomy: Dict) -> Dict[str, List[str]]:
    """
    建立節點類型到分類的映射
    
    Args:
        mcts_taxonomy: MCTS 格式的 taxonomy
    
    Returns:
        mapping: {node_type: [category_paths]}
    """
    mapping = {}
    
    def traverse(node: Dict):
        if node.get("is_leaf", False):
            mapped_nodes = node.get("mapped_nodes", [])
            category_path = " -> ".join(node.get("path", []))
            
            for node_type in mapped_nodes:
                if node_type not in mapping:
                    mapping[node_type] = []
                mapping[node_type].append(category_path)
        
        for child in node.get("children", {}).values():
            traverse(child)
    
    for root_node in mcts_taxonomy.values():
        traverse(root_node)
    
    return mapping


def get_taxonomy_statistics(mcts_taxonomy: Dict) -> Dict:
    """
    獲取 taxonomy 統計資訊
    
    Args:
        mcts_taxonomy: MCTS 格式的 taxonomy
    
    Returns:
        stats: 統計資訊字典
    """
    total_nodes = 0
    leaf_nodes = 0
    total_mapped_nodes = 0
    unique_node_types = set()
    
    def traverse(node: Dict):
        nonlocal total_nodes, leaf_nodes, total_mapped_nodes
        
        total_nodes += 1
        
        if node.get("is_leaf", False):
            leaf_nodes += 1
            mapped_nodes = node.get("mapped_nodes", [])
            total_mapped_nodes += len(mapped_nodes)
            unique_node_types.update(mapped_nodes)
        
        for child in node.get("children", {}).values():
            traverse(child)
    
    for root_node in mcts_taxonomy.values():
        traverse(root_node)
    
    return {
        "total_nodes": total_nodes,
        "leaf_nodes": leaf_nodes,
        "total_mapped_nodes": total_mapped_nodes,
        "unique_node_types": len(unique_node_types),
        "max_depth": _calculate_max_depth(mcts_taxonomy)
    }


def _calculate_max_depth(mcts_taxonomy: Dict) -> int:
    """計算 taxonomy 的最大深度"""
    def get_depth(node: Dict) -> int:
        if not node.get("children"):
            return 1
        return 1 + max([get_depth(child) for child in node.get("children", {}).values()], default=0)
    
    return max([get_depth(node) for node in mcts_taxonomy.values()], default=0)


def save_mcts_taxonomy(mcts_taxonomy: Dict, output_path: str):
    """
    保存 MCTS 格式的 taxonomy 到 JSON 檔案
    
    Args:
        mcts_taxonomy: MCTS 格式的 taxonomy
        output_path: 輸出檔案路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mcts_taxonomy, f, ensure_ascii=False, indent=2)
    
    print(f"💾 MCTS Taxonomy 已保存到: {output_path}")


def main():
    """主函數：適配 taxonomy"""
    # 設定路徑（相對於專案根目錄）
    base_dir = Path(__file__).resolve().parent.parent.parent
    taxonomy_path = base_dir / "ontology_analysis" / "taxonomy_full.json"
    output_path = base_dir / "n8n_workflow_recommender" / "data" / "mcts_taxonomy.json"
    
    print("=" * 80)
    print("n8n Taxonomy Adapter")
    print("=" * 80)
    
    # 載入 taxonomy
    print(f"\n📥 載入 taxonomy: {taxonomy_path}")
    taxonomy = load_taxonomy(str(taxonomy_path))
    
    # 轉換為 MCTS 格式
    print("\n🔄 轉換為 MCTS 格式...")
    mcts_taxonomy = convert_taxonomy_to_mcts_format(taxonomy)
    
    # 提取統計資訊
    stats = get_taxonomy_statistics(mcts_taxonomy)
    print(f"\n📊 Taxonomy 統計:")
    print(f"   - 總節點數: {stats['total_nodes']}")
    print(f"   - 葉子節點數: {stats['leaf_nodes']}")
    print(f"   - 總 mapped_nodes: {stats['total_mapped_nodes']}")
    print(f"   - 唯一節點類型數: {stats['unique_node_types']}")
    print(f"   - 最大深度: {stats['max_depth']}")
    
    # 建立節點類型到分類的映射
    print("\n🔗 建立節點類型到分類的映射...")
    node_type_mapping = build_node_type_to_category_mapping(mcts_taxonomy)
    print(f"   ✅ 建立了 {len(node_type_mapping)} 個節點類型的映射")
    
    # 保存結果
    save_mcts_taxonomy(mcts_taxonomy, str(output_path))
    
    # 保存映射
    mapping_output_path = base_dir / "n8n_workflow_recommender" / "data" / "node_type_to_category_mapping.json"
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(node_type_mapping, f, ensure_ascii=False, indent=2)
    print(f"💾 節點類型映射已保存到: {mapping_output_path}")
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

