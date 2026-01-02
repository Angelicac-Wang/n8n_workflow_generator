#!/usr/bin/env python3
"""
適配知識圖格式

將從 templates 提取的知識圖 triples 轉換為 DomainKnowledgeGraph 可用的格式。
處理節點名稱到節點類型的轉換，建立 NetworkX 圖結構。
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import networkx as nx


def load_triples(triples_path: str) -> List[Tuple[str, str, str]]:
    """
    載入知識圖三元組
    
    Args:
        triples_path: triples JSON 檔案路徑
    
    Returns:
        triples: [(head, relation, tail), ...]
    """
    triples_path = Path(triples_path)
    
    if not triples_path.exists():
        raise FileNotFoundError(f"Triples 檔案不存在: {triples_path}")
    
    with open(triples_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("triples", [])


def load_node_mappings(mappings_path: str) -> Dict[str, str]:
    """
    載入節點名稱到類型的映射
    
    Args:
        mappings_path: mappings JSON 檔案路徑
    
    Returns:
        name_to_type: {node_name: node_type}
    """
    mappings_path = Path(mappings_path)
    
    if not mappings_path.exists():
        raise FileNotFoundError(f"Mappings 檔案不存在: {mappings_path}")
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("name_to_type", {})


def load_ontology(ontology_path: str) -> Dict:
    """
    載入 Ontology
    
    Args:
        ontology_path: ontology JSON 檔案路徑
    
    Returns:
        ontology: {node_type: {"required_params": [...], ...}}
    """
    ontology_path = Path(ontology_path)
    
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology 檔案不存在: {ontology_path}")
    
    with open(ontology_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("ontology", {})


def convert_triples_to_node_types(
    triples: List[Tuple[str, str, str]],
    name_to_type: Dict[str, str]
) -> Tuple[List[Tuple[str, str, str]], Dict[str, str]]:
    """
    將節點名稱的 triples 轉換為節點類型的 triples
    
    Args:
        triples: 節點名稱的三元組列表
        name_to_type: 節點名稱到類型的映射
    
    Returns:
        (type_triples, name_to_type_mapping): 
        - type_triples: 節點類型的三元組列表
        - name_to_type_mapping: 完整的映射（包含未知節點）
    """
    type_triples = []
    unknown_nodes = set()
    
    for head, relation, tail in triples:
        head_type = name_to_type.get(head)
        tail_type = name_to_type.get(tail)
        
        # 只保留兩個節點都有類型映射的 triples
        if head_type and tail_type:
            type_triples.append((head_type, relation, tail_type))
        else:
            if not head_type:
                unknown_nodes.add(head)
            if not tail_type:
                unknown_nodes.add(tail)
    
    if unknown_nodes:
        print(f"   ⚠️  跳過了 {len(unknown_nodes)} 個未知節點的 triples")
    
    return type_triples, dict(name_to_type)


def build_ontology_for_types(
    type_ontology: Dict
) -> Dict[str, Dict]:
    """
    為節點類型建立 Ontology（直接使用類型 Ontology）
    
    Args:
        type_ontology: 節點類型的 Ontology
    
    Returns:
        type_ontology: {node_type: {"required_params": [...], ...}}
    """
    # 直接返回類型 Ontology，因為我們現在使用節點類型而不是節點名稱
    return type_ontology


def build_networkx_graph(
    triples: List[Tuple[str, str, str]],
    ontology: Optional[Dict] = None
) -> nx.DiGraph:
    """
    從 triples 建立 NetworkX 有向圖
    
    Args:
        triples: 三元組列表
        ontology: 可選的 Ontology 字典（用於添加節點屬性）
    
    Returns:
        graph: NetworkX 有向圖
    """
    graph = nx.DiGraph()
    
    # 添加所有節點
    all_nodes = set([h for h, _, _ in triples] + [t for _, _, t in triples])
    for node in all_nodes:
        node_attrs = {}
        if ontology and node in ontology:
            node_attrs = ontology[node].copy()
        graph.add_node(node, **node_attrs)
    
    # 添加所有邊
    for head, relation, tail in triples:
        graph.add_edge(head, tail, relation=relation, weight=1.0)
    
    return graph


def get_graph_statistics(graph: nx.DiGraph) -> Dict:
    """
    獲取圖的統計資訊
    
    Args:
        graph: NetworkX 圖
    
    Returns:
        stats: 統計資訊字典
    """
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "is_connected": nx.is_weakly_connected(graph),
        "num_components": nx.number_weakly_connected_components(graph),
        "density": nx.density(graph),
        "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
    }


def save_adapted_knowledge_graph(
    triples: List[Tuple[str, str, str]],
    ontology: Dict,
    output_path: str
):
    """
    保存適配後的知識圖
    
    Args:
        triples: 三元組列表
        ontology: Ontology 字典
        output_path: 輸出檔案路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "triples": triples,
        "ontology": ontology,
        "statistics": {
            "num_triples": len(triples),
            "num_nodes": len(ontology),
            "unique_relations": list(set([r for _, r, _ in triples]))
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 適配後的知識圖已保存到: {output_path}")


def main():
    """主函數：適配知識圖"""
    # 設定路徑（相對於專案根目錄）
    base_dir = Path(__file__).resolve().parent.parent.parent
    triples_path = base_dir / "n8n_workflow_recommender" / "data" / "knowledge_graph_triples.json"
    mappings_path = base_dir / "n8n_workflow_recommender" / "data" / "node_mappings.json"
    ontology_path = base_dir / "n8n_workflow_recommender" / "data" / "ontology.json"
    output_path = base_dir / "n8n_workflow_recommender" / "data" / "adapted_knowledge_graph.json"
    
    print("=" * 80)
    print("n8n Knowledge Graph Adapter")
    print("=" * 80)
    
    # 載入數據
    print(f"\n📥 載入知識圖三元組（節點名稱）: {triples_path}")
    name_triples = load_triples(str(triples_path))
    print(f"   ✅ 載入了 {len(name_triples)} 個三元組（節點名稱）")
    
    print(f"\n📥 載入節點映射: {mappings_path}")
    name_to_type = load_node_mappings(str(mappings_path))
    print(f"   ✅ 載入了 {len(name_to_type)} 個節點映射")
    
    print(f"\n📥 載入 Ontology: {ontology_path}")
    type_ontology = load_ontology(str(ontology_path))
    print(f"   ✅ 載入了 {len(type_ontology)} 個節點類型的 Ontology")
    
    # 轉換 triples 為節點類型（重要：與原始系統一致）
    print("\n🔄 將節點名稱 triples 轉換為節點類型 triples...")
    type_triples, _ = convert_triples_to_node_types(name_triples, name_to_type)
    print(f"   ✅ 轉換後得到 {len(type_triples)} 個節點類型的三元組")
    
    # 使用節點類型的 Ontology（與原始系統一致）
    print("\n🔄 使用節點類型的 Ontology...")
    ontology = build_ontology_for_types(type_ontology)
    print(f"   ✅ 使用 {len(ontology)} 個節點類型的 Ontology")
    
    # 建立 NetworkX 圖（使用節點類型）
    print("\n🔄 建立 NetworkX 圖（節點類型）...")
    graph = build_networkx_graph(type_triples, ontology)
    stats = get_graph_statistics(graph)
    print(f"   ✅ 圖統計:")
    print(f"      - 節點數: {stats['num_nodes']}")
    print(f"      - 邊數: {stats['num_edges']}")
    print(f"      - 連通分量數: {stats['num_components']}")
    print(f"      - 密度: {stats['density']:.4f}")
    
    # 保存適配後的知識圖（使用節點類型）
    save_adapted_knowledge_graph(type_triples, ontology, str(output_path))
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

