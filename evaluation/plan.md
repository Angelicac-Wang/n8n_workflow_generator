# LLM Workflow Evaluation System - Implementation Plan

## Overview

建立一個完整的評估系統，用於測試 LLM 從 n8n workflow descriptions 生成 workflow 的能力，並與 ground truth 進行多維度比較。

## Project Structure

新增以下模組結構（保持與現有項目一致的組織方式）：

```
n8n_workflow_generator/
├── evaluation/                           # 新增：評估子系統
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── evaluation_config.yaml       # 評估配置
│   │   └── workflow_generation_prompt.txt # LLM prompt 模板
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── llm_workflow_generator.py    # 使用 GPT-4o 生成 workflows
│   │   └── prompt_builder.py            # 構建 prompts
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── node_accuracy_evaluator.py   # 節點類型與連接評估
│   │   ├── parameter_evaluator.py       # 參數語意相似度評估
│   │   └── cost_tracker.py              # Token 與成本追蹤
│   ├── comparison/
│   │   ├── __init__.py
│   │   ├── workflow_normalizer.py       # 標準化 LLM 輸出與 GT
│   │   ├── node_matcher.py              # 匹配 LLM 節點與 GT 節點
│   │   └── semantic_comparator.py       # 語意相似度比較
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── report_generator.py          # 生成圖表與視覺化
│   ├── orchestration/
│   │   ├── __init__.py
│   │   ├── evaluation_pipeline.py       # 主要編排器
│   │   └── progress_tracker.py          # 進度追蹤
│   └── utils/
│       ├── __init__.py
│       ├── template_loader.py           # 載入與解析 templates
│       └── result_saver.py              # 保存評估結果
├── outputs/                              # 新增：輸出目錄
│   ├── llm_generated_workflows/         # LLM 生成的 workflows
│   ├── evaluation_results/              # 評估指標與報告
│   │   ├── summary_statistics.json
│   │   ├── detailed_per_template.json
│   │   ├── detailed_per_template.csv
│   │   └── cost_report.json
│   └── visualizations/                   # 圖表與視覺化
├── scripts/                              # 新增：執行腳本
│   ├── run_generation.py                # 僅生成 workflows
│   ├── run_evaluation.py                # 僅評估現有 workflows
│   └── run_full_pipeline.py             # 完整流程
```

## Implementation Details

### 1. LLM Workflow Generation

**檔案**: `evaluation/generators/llm_workflow_generator.py`

**功能**:
- 使用 GPT-4o 從 description 生成 n8n workflows
- 追蹤每個 template 的 token usage (prompt_tokens, completion_tokens, total_tokens)
- 錯誤處理：JSON 解析失敗時記錄錯誤並繼續
- Resume 功能：檢查已生成的檔案，跳過已完成的 templates
- Rate limiting：API 調用間隔 0.5 秒（可配置）

**Prompt 模板** (`evaluation/config/workflow_generation_prompt.txt`):
```
You are an assistant for building n8n workflows from the user request:
{{ $json.output }}

[使用用戶提供的完整 prompt...]
```

**輸出格式** (保存至 `outputs/llm_generated_workflows/generated_{template_id}.json`):
```json
{
  "template_id": "1862",
  "llm_response": {
    "mode": "create_workflow",
    "workflowPlan": {
      "nodes": [...],
      "connections": [...]
    }
  },
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  },
  "error": null,
  "generated_at": "2026-01-11T12:00:00"
}
```

### 2. Workflow Normalization

**檔案**: `evaluation/comparison/workflow_normalizer.py`

**功能**:
- 將 ground truth templates 標準化為統一格式
- 將 LLM 輸出（支援 "steps" 或 "nodes+connections" 兩種格式）標準化
- Node type 正規化：移除 namespace prefix（例如 "n8n-nodes-base.httpRequest" → "httpRequest"）

**標準化輸出格式**:
```python
{
  "nodes": [
    {
      "id": "node-id",
      "name": "Node Name",
      "type": "httpRequest",           # 正規化後的 type
      "full_type": "n8n-nodes-base.httpRequest",
      "parameters": {                   # 僅 top-level parameters
        "url": "https://api.com",
        "method": "GET"
      }
    }
  ],
  "connections": [
    {
      "from": "Node A",
      "to": "Node B",
      "from_output": 0,
      "to_input": 0
    }
  ]
}
```

### 3. Node Matching

**檔案**: `evaluation/comparison/node_matcher.py`

**策略**:
- Greedy matching：按 node type 分組，同類型節點按順序配對
- 僅匹配相同 type 的節點
- 未匹配的節點分為：unmatched_gt (false negatives) 和 unmatched_llm (false positives)

### 4. Evaluation Metrics

#### 4.1 Node Type Accuracy

**檔案**: `evaluation/evaluators/node_accuracy_evaluator.py`

**計算**:
- TP (True Positive): 成功匹配且 type 相同的節點數
- FP (False Positive): LLM 生成但無 GT 匹配的節點數
- FN (False Negative): GT 中存在但 LLM 未生成的節點數
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × (Precision × Recall) / (Precision + Recall)

#### 4.2 Connection Accuracy

**計算**:
- 將 connections 轉換為 (from, to) 元組集合
- Correct connections = GT connections ∩ LLM connections
- Precision = Correct / LLM connection count
- Recall = Correct / GT connection count
- F1 score

#### 4.3 Parameter Filling Accuracy

**檔案**: `evaluation/evaluators/parameter_evaluator.py`

**方法**:
- 使用 `paraphrase-multilingual-mpnet-base-v2` embedding model（與項目現有配置一致）
- 僅比較 top-level scalar parameters (str, int, float, bool)
- 語意相似度閾值：0.8（cosine similarity）
- 對於每對匹配的節點：
  1. 提取 scalar parameters
  2. 計算 parameter key 和 value 的 embedding similarity
  3. Similarity ≥ 0.8 視為匹配
- 輸出：每個節點的 match_ratio 與平均 parameter accuracy

#### 4.4 Cost Tracking

**檔案**: `evaluation/evaluators/cost_tracker.py`

**定價** (GPT-4o, 2026-01):
- Input: $5 / 1M tokens
- Output: $15 / 1M tokens

**追蹤**:
- Per template: prompt_tokens, completion_tokens, cost (USD)
- Aggregate: total tokens, total cost, average cost per template

### 5. Output Files

#### 5.1 Generated Workflows
- 路徑: `outputs/llm_generated_workflows/generated_{template_id}.json`
- 格式: 保留 LLM 原始輸出 + token usage + 時間戳

#### 5.2 Evaluation Results

**Summary Statistics** (`outputs/evaluation_results/summary_statistics.json`):
```json
{
  "total_templates": 1000,
  "successful_evaluations": 987,
  "failed_evaluations": 13,
  "node_accuracy": {
    "mean_f1": 0.78,
    "median_f1": 0.82,
    "std_f1": 0.15,
    "min_f1": 0.23,
    "max_f1": 1.0
  },
  "connection_accuracy": { ... },
  "parameter_accuracy": { ... }
}
```

**Detailed Per-Template** (`outputs/evaluation_results/detailed_per_template.json` + `.csv`):
```json
[
  {
    "template_id": "1862",
    "template_name": "OpenAI GPT-3: Company enrichment...",
    "error": null,
    "metrics": {
      "node_type_precision": 0.85,
      "node_type_recall": 0.92,
      "node_type_f1": 0.88,
      "connection_precision": 0.80,
      "connection_recall": 0.75,
      "connection_f1": 0.77,
      "avg_parameter_accuracy": 0.65,
      "total_cost": 0.0123,
      "usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 567
      }
    }
  }
]
```

**Cost Report** (`outputs/evaluation_results/cost_report.json`):
```json
{
  "total_input_tokens": 1234567,
  "total_output_tokens": 567890,
  "total_tokens": 1802457,
  "total_cost": 15.23,
  "avg_cost_per_template": 0.015,
  "currency": "USD"
}
```

### 6. Visualization

**檔案**: `evaluation/visualization/report_generator.py`

**生成的圖表** (保存至 `outputs/visualizations/`):
1. **node_accuracy_distribution.png**: Node F1 score 分佈（histogram + KDE）
2. **connection_accuracy_distribution.png**: Connection F1 score 分佈
3. **parameter_accuracy_distribution.png**: Parameter accuracy 分佈
4. **cost_analysis.png**: 4-panel 成本分析
   - Cost distribution
   - Token usage scatter plot
   - Cumulative cost curve
   - Cost vs Accuracy correlation
5. **comparison_heatmap.png**: 前 50 個 templates 的指標熱圖
6. **metric_correlations.png**: Metrics 之間的相關性矩陣

**使用庫**:
- matplotlib, seaborn (與現有項目依賴一致)
- 配置: 300 DPI, figure size [12, 8]

### 7. Configuration

**檔案**: `evaluation/config/evaluation_config.yaml`

```yaml
# Paths
templates_dir: "n8n_templates/testing_data"
output_dir: "outputs"

# OpenAI Configuration
openai_key: "${OPENAI_API_KEY}"  # 從環境變數或主配置讀取
model: "gpt-4o"
api_delay: 0.5  # API 調用間隔（秒）

# Prompt Template
prompt_template_path: "evaluation/config/workflow_generation_prompt.txt"

# Evaluation Settings
embedding_model: "paraphrase-multilingual-mpnet-base-v2"
param_similarity_threshold: 0.8

# Visualization
visualization:
  figure_size: [12, 8]
  dpi: 300
  style: "seaborn-v0_8-darkgrid"

# Processing
batch_size: 100
resume_enabled: true
```

### 8. Execution Scripts

#### Script 1: Generate Workflows Only
**檔案**: `scripts/run_generation.py`

```bash
python scripts/run_generation.py [--resume] [--limit N]
```

功能：
- 讀取 1000 個 templates
- 調用 GPT-4o 生成 workflows
- 保存至 `outputs/llm_generated_workflows/`
- Resume: 跳過已生成的 templates

#### Script 2: Evaluate Existing Workflows
**檔案**: `scripts/run_evaluation.py`

```bash
python scripts/run_evaluation.py
```

功能：
- 讀取已生成的 workflows
- 執行評估（獨立於生成）
- 保存結果與視覺化

#### Script 3: Full Pipeline
**檔案**: `scripts/run_full_pipeline.py`

```bash
python scripts/run_full_pipeline.py [--resume]
```

功能：
- 生成 → 評估 → 視覺化
- 端到端執行

## Implementation Sequence

### Phase 1: Foundation (優先)
1. 建立模組結構與 `__init__.py`
2. 實作 `evaluation_config.yaml` 與 config loader
3. 實作 `template_loader.py`（重用 `file_loader.py` 模式）
4. 實作 `prompt_builder.py`

### Phase 2: LLM Generation
1. 實作 `llm_workflow_generator.py`（重用 `intent_analyzer.py` 的 OpenAI 客戶端模式）
2. 實作 `progress_tracker.py` with resume capability
3. 實作 `result_saver.py`
4. 測試：生成 5-10 個樣本

### Phase 3: Normalization & Matching
1. 實作 `workflow_normalizer.py`（處理 GT 和 LLM 兩種格式）
2. 實作 `node_matcher.py`（greedy matching）
3. 單元測試

### Phase 4: Evaluation Metrics
1. 實作 `node_accuracy_evaluator.py`（節點與連接）
2. 實作 `parameter_evaluator.py`（語意相似度）
3. 實作 `cost_tracker.py`
4. 驗證指標計算

### Phase 5: Orchestration
1. 實作 `evaluation_pipeline.py`（主編排器）
2. 整合所有組件
3. 實作結果聚合與保存（JSON + CSV）
4. 端到端測試

### Phase 6: Visualization
1. 實作 `report_generator.py`（所有圖表類型）
2. 測試視覺化生成
3. 自定義樣式與佈局

### Phase 7: Scripts & Documentation
1. 建立執行腳本（`run_generation.py`, `run_evaluation.py`, `run_full_pipeline.py`）
2. 添加命令列參數解析
3. 撰寫 README
4. 在 50-100 個 templates 上測試

### Phase 8: Full Run
1. 在全部 1000 個 templates 上執行生成
2. 執行評估
3. 驗證輸出檔案與視覺化
4. 分析結果

## Critical Files to Modify/Create

### 新增檔案 (按優先級):
1. `evaluation/config/evaluation_config.yaml` - 配置檔案
2. `evaluation/config/workflow_generation_prompt.txt` - Prompt 模板
3. `evaluation/generators/llm_workflow_generator.py` - LLM 生成核心
4. `evaluation/comparison/workflow_normalizer.py` - 標準化關鍵邏輯
5. `evaluation/evaluators/node_accuracy_evaluator.py` - 節點評估
6. `evaluation/evaluators/parameter_evaluator.py` - 參數評估（最複雜）
7. `evaluation/orchestration/evaluation_pipeline.py` - 主編排器
8. `evaluation/visualization/report_generator.py` - 視覺化
9. `scripts/run_full_pipeline.py` - 執行入口

### 重用現有模式:
- OpenAI 客戶端設置：參考 `n8n_workflow_recommender/nlu/intent_analyzer.py`
- 檔案載入：重用 `n8n_workflow_recommender/utils/file_loader.py`
- Embedding model：使用 `config.yaml` 中的 `paraphrase-multilingual-mpnet-base-v2`
- 配置管理：與現有 `config.yaml` 格式一致

## Key Design Decisions

1. **Node Type Matching**: 僅比較 type，忽略 parameters（用戶需求）
2. **Connection Matching**: Set-based intersection，計算 precision/recall
3. **Parameter Comparison**:
   - 僅 top-level scalar parameters
   - 語意相似度 ≥ 0.8
   - 使用 sentence-transformers
4. **LLM Output Format**: 支援 "steps" 和 "nodes+connections" 雙格式
5. **Cost Tracking**: 基於實際 API response 的 token usage
6. **Evaluation Independence**: 評估可獨立執行，不依賴生成步驟
7. **Output Preservation**: 保留 LLM 原始輸出，供後續分析

## Error Handling

1. **Empty descriptions**: 記錄錯誤，跳過該 template
2. **Malformed LLM JSON**: 嘗試修復，失敗則記錄錯誤
3. **Missing GT connections**: 推斷為線性連接
4. **API errors**: 重試最多 3 次，指數退避
5. **Rate limiting**: 配置延遲 + 錯誤處理

## Expected Outputs

執行完成後，用戶將獲得：

1. **1000 個 LLM 生成的 workflows** (`outputs/llm_generated_workflows/`)
2. **詳細評估報告**:
   - Summary statistics (JSON)
   - Per-template metrics (JSON + CSV)
   - Cost report (JSON)
3. **6 個視覺化圖表** (`outputs/visualizations/`)
4. **可獨立執行的評估流程**（可重新運行評估而無需重新生成）

## Estimated Costs

假設每個 template 平均：
- Prompt tokens: ~1500 (description + prompt template)
- Completion tokens: ~800 (workflow JSON)

總成本估算：
- Input: 1000 × 1500 tokens × $5/1M = **$7.50**
- Output: 1000 × 800 tokens × $15/1M = **$12.00**
- **總計: ~$20 USD**

## Dependencies

新增至 `requirements.txt`:
```
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
```

現有依賴（已滿足）:
- openai>=1.0.0
- sentence-transformers>=2.2.0
- numpy, networkx, pyyaml, scikit-learn

---

**準備開始實作！** 此計劃提供完整的架構設計、資料流程、評估指標演算法、檔案結構與執行步驟。
