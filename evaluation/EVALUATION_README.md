# LLM Workflow Evaluation System

這個評估系統用於測試 LLM 從 n8n workflow descriptions 生成 workflow 的能力，並與 ground truth 進行多維度比較。

## 功能特色

- **LLM 生成**: 使用 GPT-4o 從 template descriptions 生成 n8n workflows
- **多維度評估**:
  - Node Type Accuracy (precision, recall, F1)
  - Connection Accuracy (precision, recall, F1)
  - Parameter Filling Accuracy (語意相似度)
  - Cost Tracking (token usage 與成本)
- **獨立評估**: 可單獨運行評估而無需重新生成
- **詳細報告**: JSON, CSV, 與視覺化圖表
- **Resume 功能**: 可中斷並繼續處理

## 安裝依賴

```bash
pip install matplotlib seaborn pandas
```

其他依賴已在主專案的 `requirements.txt` 中。

## 配置

### 1. 設定 OpenAI API Key

在環境變數或 `n8n_workflow_recommender/config/config.yaml` 中設定:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 修改評估配置 (可選)

編輯 `evaluation/config/evaluation_config.yaml`:

```yaml
# OpenAI Configuration
model: "gpt-4o"
api_delay: 0.5  # 調整 API 調用間隔

# Evaluation Settings
param_similarity_threshold: 0.8  # 調整參數匹配閾值
```

## 使用方法

### 完整流程 (推薦)

生成 → 評估 → 視覺化：

```bash
python scripts/run_full_pipeline.py
```

選項：
- `--resume`: 跳過已生成的 templates
- `--limit N`: 僅處理前 N 個 templates (用於測試)

範例：

```bash
# 測試：處理前 10 個 templates
python scripts/run_full_pipeline.py --limit 10

# 完整運行：處理全部 1000 個 templates (可中斷並 resume)
python scripts/run_full_pipeline.py --resume
```

### 僅生成 Workflows

```bash
python scripts/run_generation.py --resume
```

### 僅評估現有 Workflows

```bash
python scripts/run_evaluation.py
```

## 輸出結構

```
outputs/
├── llm_generated_workflows/          # LLM 生成的 workflows
│   ├── generated_1862.json
│   ├── generated_1863.json
│   └── ...
├── evaluation_results/               # 評估結果
│   ├── summary_statistics.json       # 彙總統計
│   ├── detailed_per_template.json    # 每個 template 的詳細指標
│   ├── detailed_per_template.csv     # CSV 格式
│   └── cost_report.json              # 成本報告
└── visualizations/                    # 視覺化圖表
    ├── node_accuracy_distribution.png
    ├── connection_accuracy_distribution.png
    ├── parameter_accuracy_distribution.png
    ├── cost_analysis.png
    ├── comparison_heatmap.png
    └── metric_correlations.png
```

## 評估指標說明

### 1. Node Type Accuracy

- **Precision**: TP / (TP + FP) - LLM 生成的節點中有多少是正確的
- **Recall**: TP / (TP + FN) - Ground truth 中的節點有多少被 LLM 正確生成
- **F1 Score**: 調和平均數

### 2. Connection Accuracy

- 比較 workflow 中的連接關係 (A → B)
- 使用集合交集計算 precision, recall, F1

### 3. Parameter Filling Accuracy

- 使用語意相似度 (embedding cosine similarity)
- 閾值: 0.8 (可配置)
- 僅比較 top-level scalar parameters

### 4. Cost Tracking

- 基於實際 API response 的 token usage
- GPT-4o 定價:
  - Input: $5 / 1M tokens
  - Output: $15 / 1M tokens

## 預估成本

處理 1000 個 templates:
- 預估: ~$20 USD
- 實際成本取決於 description 長度和 LLM 生成的 workflow 複雜度

## 故障排除

### ImportError: No module named 'matplotlib'

```bash
pip install matplotlib seaborn pandas
```

### OpenAI API Key not found

確保設定環境變數或在 config.yaml 中配置:

```bash
export OPENAI_API_KEY="sk-..."
```

### Rate limit errors

調整 `api_delay` 在 `evaluation_config.yaml`:

```yaml
api_delay: 1.0  # 增加延遲至 1 秒
```

## 進階使用

### 自定義 Prompt

編輯 `evaluation/config/workflow_generation_prompt.txt` 來修改 LLM prompt。

### 調整評估閾值

修改 `evaluation_config.yaml`:

```yaml
param_similarity_threshold: 0.7  # 降低匹配閾值
```

### 分析特定 Template

```python
from evaluation.utils.result_saver import ResultSaver

saver = ResultSaver("outputs")
result = saver.load_generated_workflow("1862")
print(result['llm_response'])
```

## 架構概覽

- **evaluation/generators/**: LLM workflow 生成
- **evaluation/comparison/**: Workflow 正規化與節點匹配
- **evaluation/evaluators/**: 評估指標計算
- **evaluation/orchestration/**: 主要流程編排
- **evaluation/visualization/**: 圖表生成
- **scripts/**: 執行腳本

## 貢獻

如有問題或建議，請提交 issue 或 pull request。
