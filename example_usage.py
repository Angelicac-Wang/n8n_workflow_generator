#!/usr/bin/env python3
"""
n8n Workflow Generator - 使用範例

這個腳本展示如何使用 n8n Workflow Generator 生成工作流程。
"""

import json
import sys
from pathlib import Path

# 添加模組路徑
sys.path.insert(0, str(Path(__file__).parent))

from n8n_workflow_recommender.core.orchestrator import WorkflowOrchestrator


def main():
    # 配置 OpenAI API Key
    # 優先順序：環境變量 > config.yaml（由 orchestrator 自動讀取）
    import os
    
    openai_key = None
    
    # 方式 1: 從環境變量讀取（優先）
    if os.getenv("OPENAI_API_KEY"):
        openai_key = os.getenv("OPENAI_API_KEY")
        print("✅ 從環境變量讀取 OpenAI API Key")
    
    # 方式 2: 如果環境變量沒有，傳入 None 讓 orchestrator 從 config.yaml 讀取
    # orchestrator 會自動從 config.yaml 讀取 API key
    
    print("=" * 80)
    print("n8n Workflow Generator - 使用範例")
    print("=" * 80)
    
    # 初始化 Orchestrator
    print("\n🔧 初始化系統...")
    try:
        # 如果 openai_key 為 None，傳入 None 讓 orchestrator 從 config.yaml 讀取
        # 如果 openai_key 有值，直接傳入
        orchestrator = WorkflowOrchestrator(openai_key=openai_key if openai_key else None)
    except Exception as e:
        print(f"❌ 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 範例查詢
    user_queries = [
        "設計智能信件處理流程，當有新gmail信件進來時自動觸發，使用 openai 理解信件內容，如果與開會相關，則提取開始時間、結束時間、地點存入google calendar。",
        "設計一個 OCR 流程，讀取圖片並識別文字",
        "創建一個自動發送郵件的流程"
    ]
    
    # 處理第一個查詢
    query = user_queries[0]
    print(f"\n📝 用戶查詢: {query}")
    print("\n" + "-" * 80)
    
    try:
        result = orchestrator.process_user_request(query)
        
        if "error" in result:
            print(f"\n❌ 錯誤: {result['error']}")
            return
        
        # 顯示結果
        print("\n✅ 生成成功！")
        print("\n📊 結果摘要:")
        print(f"   - 最佳路徑: {' -> '.join(result['best_workflow']['path'][:5])}...")
        print(f"   - MF 分數: {result['best_workflow'].get('mf_score', 'N/A')}")
        
        # 保存工作流程 JSON
        output_path = Path("output") / "generated_workflow.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result['best_workflow']['workflow_json'], f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 工作流程已保存到: {output_path}")
        print(f"\n📄 工作流程 JSON (前 500 字符):")
        workflow_str = json.dumps(result['best_workflow']['workflow_json'], indent=2, ensure_ascii=False)
        print(workflow_str[:500] + "..." if len(workflow_str) > 500 else workflow_str)
        
    except Exception as e:
        print(f"\n❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
