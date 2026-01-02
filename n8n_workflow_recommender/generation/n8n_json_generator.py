#!/usr/bin/env python3
"""
n8n 工作流程 JSON 生成器

將節點類型鏈轉換為完整的 n8n 工作流程 JSON。
"""

import uuid
from typing import List, Dict, Optional, Tuple
from ..adapters.node_type_mapper import NodeTypeMapper


class N8nWorkflowGenerator:
    """
    n8n 工作流程 JSON 生成器
    
    將節點類型鏈轉換為符合 n8n 格式的完整工作流程 JSON。
    """
    
    def __init__(self, node_mapper: Optional[NodeTypeMapper] = None):
        """
        初始化生成器
        
        Args:
            node_mapper: 節點類型映射器（用於將類型轉換為名稱）
        """
        self.node_mapper = node_mapper
        self.node_spacing_x = 300  # 節點間水平間距
        self.node_spacing_y = 100  # 節點間垂直間距
        self.start_x = 250  # 起始 X 座標
        self.start_y = 300  # 起始 Y 座標
    
    def generate_workflow_json(
        self,
        node_type_chain: List[str],
        workflow_name: str = "Generated Workflow",
        node_params: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        生成完整的 n8n 工作流程 JSON
        
        Args:
            node_type_chain: 節點類型列表
            workflow_name: 工作流程名稱
            node_params: 節點參數字典 {node_type: {param: value}}
        
        Returns:
            workflow_json: n8n 工作流程 JSON
        """
        if not node_type_chain:
            raise ValueError("節點鏈不能為空")
        
        # 將節點類型轉換為節點名稱（如果提供了 mapper）
        node_names = []
        if self.node_mapper:
            node_names = self.node_mapper.convert_chain_to_names(node_type_chain)
        else:
            # 如果沒有 mapper，使用類型作為名稱（簡化處理）
            node_names = [self._type_to_display_name(node_type) for node_type in node_type_chain]
        
        # 生成節點列表
        nodes = []
        node_name_to_id = {}
        
        for i, (node_type, node_name) in enumerate(zip(node_type_chain, node_names)):
            node_id = str(uuid.uuid4())
            node_name_to_id[node_name] = node_id
            
            # 計算位置
            position = [
                self.start_x + i * self.node_spacing_x,
                self.start_y
            ]
            
            # 獲取參數
            params = node_params.get(node_type, {}) if node_params else {}
            
            # 構建節點
            node = {
                "id": node_id,
                "name": node_name,
                "type": node_type,
                "typeVersion": 1,
                "position": position,
                "parameters": params
            }
            
            nodes.append(node)
        
        # 生成連接關係
        connections = {}
        for i in range(len(node_names) - 1):
            source_name = node_names[i]
            target_name = node_names[i + 1]
            
            if source_name not in connections:
                connections[source_name] = {
                    "main": [[]]
                }
            
            connections[source_name]["main"][0].append({
                "node": target_name,
                "type": "main",
                "index": 0
            })
        
        # 構建完整的工作流程 JSON
        workflow_json = {
            "name": workflow_name,
            "nodes": nodes,
            "connections": connections,
            "active": False,
            "settings": {},
            "pinData": {}
        }
        
        return workflow_json
    
    def _type_to_display_name(self, node_type: str) -> str:
        """
        將節點類型轉換為顯示名稱
        
        Args:
            node_type: 節點類型（如 'n8n-nodes-base.manualTrigger'）
        
        Returns:
            display_name: 顯示名稱（如 'Manual Trigger'）
        """
        # 提取最後一部分作為基礎名稱
        parts = node_type.split('.')
        if len(parts) > 0:
            base_name = parts[-1]
            # 將 camelCase 轉換為 Title Case
            import re
            name = re.sub(r'(?<!^)(?=[A-Z])', ' ', base_name)
            return name.title()
        return node_type
    
    def save_workflow_json(self, workflow_json: Dict, output_path: str):
        """
        保存工作流程 JSON 到檔案
        
        Args:
            workflow_json: 工作流程 JSON
            output_path: 輸出檔案路徑
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(workflow_json, f, ensure_ascii=False, indent=2)
        
        print(f"💾 工作流程 JSON 已保存到: {output_path}")

