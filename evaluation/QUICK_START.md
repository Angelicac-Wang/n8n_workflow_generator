# Quick Start Guide - LLM Workflow Evaluation

## 安裝步驟

### 1. 安裝依賴

由於您的環境使用外部管理，建議使用虛擬環境：

```bash
# 進入專案目錄
cd /Users/yu/Desktop/projects/gss_cai/n8n_workflow_generator

# 建立虛擬環境（如果還沒有）
python3 -m venv venv

# 啟動虛擬環境
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 設定 OpenAI API Key

```bash
# 方法 1: 環境變數
export OPENAI_API_KEY="sk-your-api-key-here"

# 方法 2: 修改配置檔案
# 編輯 n8n_workflow_recommender/config/config.yaml
# 將 openai_key 改為您的 API key
```

## 使用方法

### 快速測試（2 個 templates）

```bash
# 啟動虛擬環境
source venv/bin/activate

# 測試完整流程
python scripts/run_full_pipeline.py --limit 2
```

### 完整運行（1000 個 templates）

```bash
# 啟動虛擬環境
source venv/bin/activate

# 運行完整評估（預估 $20 USD，約 30-60 分鐘）
python scripts/run_full_pipeline.py --resume
```

### 分步執行

```bash
# 1. 僅生成 workflows
python scripts/run_generation.py --resume --limit 10

# 2. 僅評估（使用已生成的 workflows）
python scripts/run_evaluation.py
```

## 查看結果

完成後，檢查以下目錄：

```bash
# 1. LLM 生成的 workflows
ls outputs/llm_generated_workflows/

# 2. 評估結果
cat outputs/evaluation_results/summary_statistics.json
open outputs/evaluation_results/detailed_per_template.csv

# 3. 視覺化圖表
open outputs/visualizations/node_accuracy_distribution.png
open outputs/visualizations/cost_analysis.png
```

## 輸出範例

### Summary Statistics
```json
{
  "total_templates": 1000,
  "successful_evaluations": 987,
  "node_accuracy": {
    "mean_f1": 0.78,
    "median_f1": 0.82
  },
  "connection_accuracy": {
    "mean_f1": 0.65
  },
  "parameter_accuracy": {
    "mean": 0.52
  }
}
```

### Cost Report
```json
{
  "total_tokens": 1802457,
  "total_cost": 19.45,
  "avg_cost_per_template": 0.019,
  "currency": "USD"
}
```

## 常見問題

### Q: ModuleNotFoundError

**A**: 確保已安裝所有依賴：
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Q: OpenAI API Key not found

**A**: 設定環境變數：
```bash
export OPENAI_API_KEY="sk-..."
```

或編輯 `n8n_workflow_recommender/config/config.yaml`

### Q: Rate limit error

**A**: 調整 API 延遲：

編輯 `evaluation/config/evaluation_config.yaml`:
```yaml
api_delay: 1.0  # 增加至 1 秒
```

### Q: 想要中斷後繼續

**A**: 使用 `--resume` 參數：
```bash
python scripts/run_full_pipeline.py --resume
```

## 檔案結構

```
evaluation/
├── config/
│   ├── evaluation_config.yaml       # 主要配置
│   └── workflow_generation_prompt.txt  # LLM prompt
├── generators/                      # LLM 生成
├── evaluators/                      # 評估指標
├── comparison/                      # 正規化與匹配
├── visualization/                   # 視覺化
└── orchestration/                   # 流程編排

scripts/
├── run_generation.py               # 僅生成
├── run_evaluation.py               # 僅評估
└── run_full_pipeline.py            # 完整流程

outputs/
├── llm_generated_workflows/        # 生成結果
├── evaluation_results/             # 評估報告
└── visualizations/                 # 圖表
```

## 下一步

1. 運行測試（2 個 templates）確認系統正常
2. 運行完整評估（1000 個 templates）
3. 分析結果並調整 prompt 或閾值
4. 重新評估並比較結果

詳細說明請參考 [EVALUATION_README.md](EVALUATION_README.md)
