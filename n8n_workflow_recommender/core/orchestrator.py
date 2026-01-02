#!/usr/bin/env python3
"""
主協調器

整合所有組件，提供統一的 API 處理用戶查詢並生成 n8n 工作流程。
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# 避免 huggingface tokenizers 的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .workflow_system import HybridWorkflowSystem
from ..models.chain_recommender import ChainRecommender
from ..generation.n8n_json_generator import N8nWorkflowGenerator
from ..generation.parameter_filler import ParameterFiller
from ..adapters.node_type_mapper import NodeTypeMapper
from ..utils.file_loader import load_json, load_yaml
from ..utils.logger import setup_logger


class WorkflowOrchestrator:
    """
    工作流程協調器
    
    整合生成器（Vincent）和評分器（Daniel），提供完整的推薦流程。
    """
    
    def __init__(
        self,
        openai_key: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        初始化協調器
        
        Args:
            openai_key: OpenAI API 密鑰（可選，如果為 None 或空字串，會從 config.yaml 讀取）
            config_path: 配置檔案路徑（可選）
        """
        print("\n" + "=" * 80)
        print("Initializing n8n Workflow Recommender Orchestrator")
        print("=" * 80)
        
        # 載入配置
        if config_path:
            self.config = load_yaml(config_path)
        else:
            # 使用默認路徑
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "n8n_workflow_recommender" / "config" / "config.yaml"
            self.config = load_yaml(str(config_path))
        
        # 如果 openai_key 為 None 或空字串，從 config 讀取
        if not openai_key or not openai_key.strip():
            config_key = self.config.get('api', {}).get('openai_key', '')
            if config_key and config_key != "YOUR_OPENAI_API_KEY_HERE" and config_key.strip():
                openai_key = config_key
                print("✅ 從 config.yaml 讀取 OpenAI API Key")
            else:
                openai_key = ''
                print("⚠️  Warning: OpenAI API key not found in config.yaml")
        
        # 設定路徑
        # 在打包版本中，base_dir 是打包目錄的根目錄
        base_dir = Path(__file__).resolve().parent.parent.parent
        paths = self.config.get('paths', {})
        
        # 使用帶有 example_use_cases 的 taxonomy
        # 優先使用配置文件中的路徑，否則使用默認路徑
        taxonomy_config_path = paths.get('taxonomy_path', '../data/taxonomy_full_with_examples.json')
        if taxonomy_config_path.startswith('../'):
            self.taxonomy_path = base_dir / taxonomy_config_path[3:]
        else:
            self.taxonomy_path = Path(taxonomy_config_path) if Path(taxonomy_config_path).is_absolute() else base_dir / taxonomy_config_path
        
        # 知識圖路徑（在外部 data/ 目錄中）
        self.kg_path = base_dir / "data" / "adapted_knowledge_graph.json"
        
        # 處理 Matrix Factorization 模型路徑
        matrix_model_path = paths.get('matrix_model_dir', '../models/matrix_factorization')
        # 移除 ../ 前綴
        if matrix_model_path.startswith('../'):
            matrix_model_path = matrix_model_path[3:]
        if not Path(matrix_model_path).is_absolute():
            self.matrix_model_dir = base_dir / matrix_model_path
        else:
            self.matrix_model_dir = Path(matrix_model_path)
        
        # 節點映射路徑
        self.node_mappings_path = base_dir / "data" / "node_mappings.json"
        
        # 載入數據
        print("\n📥 Loading data...")
        
        # 載入知識圖
        print(f"   - Loading knowledge graph: {self.kg_path}")
        kg_data = load_json(str(self.kg_path))
        triples = kg_data.get('triples', [])
        ontology = kg_data.get('ontology', {})
        print(f"   ✅ Loaded {len(triples)} triples and {len(ontology)} node types")
        
        # 載入節點映射
        print(f"   - Loading node mappings: {self.node_mappings_path}")
        self.node_mapper = NodeTypeMapper(str(self.node_mappings_path))
        print(f"   ✅ Loaded {len(self.node_mapper.name_to_type)} mappings")
        
        # 初始化生成器（Vincent）
        print("\n🔧 Initializing Generator (Vincent)...")
        # 使用處理後的 openai_key（已經從 config 讀取過了）
        self.generator = HybridWorkflowSystem(
            triples=triples,
            ontology=ontology,
            taxonomy_path=str(self.taxonomy_path),
            openai_api_key=openai_key
        )
        
        # 初始化評分器（Daniel）
        print("\n🔧 Initializing Scorer (Daniel)...")
        self.scorer = ChainRecommender(
            model_dir=str(self.matrix_model_dir),
            use_type_model=True
        )
        
        # 初始化參數填充器
        print("\n🔧 Initializing Parameter Filler...")
        self.parameter_filler = ParameterFiller(ontology)
        
        # 初始化 JSON 生成器
        print("\n🔧 Initializing JSON Generator...")
        self.json_generator = N8nWorkflowGenerator(self.node_mapper)
        
        print("\n✅ Orchestrator initialized successfully!")
    
    def process_user_request(self, user_query: str) -> Dict:
        """
        處理用戶請求
        
        Args:
            user_query: 用戶查詢字符串
        
        Returns:
            result: {
                "best_workflow": {...},
                "candidates": [...],
                "workflow_json": {...}
            }
        """
        print("\n" + "=" * 80)
        print("Processing User Request")
        print("=" * 80)
        
        # Phase 1: Generation (Vincent)
        print("\n📝 Phase 1: Workflow Generation")
        candidates = self.generator.generate_workflow(user_query)
        
        if not candidates:
            return {
                "error": "無法生成工作流程候選",
                "candidates": []
            }
        
        # Phase 2: Scoring (Daniel)
        print("\n📊 Phase 2: Workflow Scoring")
        ranked_candidates = self.scorer.rank_candidates(candidates)
        
        if not ranked_candidates:
            return {
                "error": "無法評分工作流程候選",
                "candidates": []
            }
        
        # Phase 3: Select Best
        print("\n🏆 Phase 3: Selecting Best Workflow")
        best_candidate = ranked_candidates[0]
        best_path = best_candidate.get('path', [])
        
        print(f"   - Best path: {' -> '.join(best_path)}")
        print(f"   - Score: {best_candidate.get('mf_score', 0.0):.4f}")
        
        # Phase 4: Fill Parameters
        print("\n🔧 Phase 4: Filling Parameters")
        extracted_params = best_candidate.get('params', {})
        chain_params = self.parameter_filler.fill_chain_parameters(
            best_path,
            extracted_params,
            strict=False
        )
        
        # Phase 5: Generate n8n JSON
        print("\n📄 Phase 5: Generating n8n Workflow JSON")
        workflow_json = self.json_generator.generate_workflow_json(
            node_type_chain=best_path,
            workflow_name="Generated Workflow",
            node_params=chain_params
        )
        
        # 構建結果
        result = {
            "best_workflow": {
                "path": best_path,
                "score": best_candidate.get('mf_score', 0.0),
                "description": best_candidate.get('description', ''),
                "params": chain_params
            },
            "candidates": [
                {
                    "path": c.get('path', []),
                    "score": c.get('mf_score', 0.0),
                    "description": c.get('description', '')
                }
                for c in ranked_candidates[:5]  # 只返回前5個
            ],
            "workflow_json": workflow_json
        }
        
        print("\n✅ Workflow generation completed!")
        return result
    
    def save_result(self, result: Dict, output_path: str):
        """
        保存結果到檔案
        
        Args:
            result: 結果字典
            output_path: 輸出檔案路徑
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Result saved to: {output_path}")


def main():
    """主函數：測試協調器"""
    import sys
    import os
    
    # 添加專案根目錄到 Python 路徑
    base_dir = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(base_dir))
    
    # 設定路徑
    config_path = base_dir / "n8n_workflow_recommender" / "config" / "config.yaml"
    
    # 從配置載入 API key
    config = load_yaml(str(config_path))
    openai_key = config.get('api', {}).get('openai_key', '')
    
    if not openai_key:
        print("❌ Error: OpenAI API key not found in config.yaml")
        return
    
    # 初始化協調器
    orchestrator = WorkflowOrchestrator(
        openai_key=openai_key,
        config_path=str(config_path)
    )
    
    # 測試查詢
    user_query = "設計智能信件處理流程，當有新gmail信件進來時自動觸發，使用 openai 理解信件內容，如果與開會相關，則提取開始時間、結束時間、地點存入google calendar。"
    
    # 處理請求
    result = orchestrator.process_user_request(user_query)
    
    # 保存結果
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "test_result.json"
    orchestrator.save_result(result, str(output_path))
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

