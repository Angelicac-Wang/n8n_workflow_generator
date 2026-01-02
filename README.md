# n8n Workflow Generator

智能 n8n 工作流程自動生成系統 - 根據自然語言描述自動生成完整的 n8n 工作流程 JSON

## 📋 簡介

本系統可以根據自然語言描述自動生成完整的 n8n 工作流程 JSON。系統使用：
- **NLU (自然語言理解)**: 使用 GPT-4o 理解用戶意圖
- **MCTS 搜索**: 在 taxonomy 中搜索相關節點
- **A* 路徑生成**: 在知識圖中生成工作流程路徑
- **Matrix Factorization**: 對候選工作流程進行評分和排序
- **參數填充**: 自動填充節點參數

## 📁 目錄結構

```
n8n_workflow_generator_package/
├── n8n_workflow_recommender/     # 主程式碼
│   ├── core/                    # 核心協調器
│   ├── nlu/                     # 自然語言理解
│   ├── search/                  # MCTS 搜索
│   ├── generation/              # 工作流程生成
│   ├── models/                  # 評分模型
│   ├── adapters/                # 數據適配器
│   ├── utils/                   # 工具函數
│   └── config/                  # 配置文件
├── data/                        # 所有數據文件
│   ├── adapted_knowledge_graph.json
│   ├── node_mappings.json
│   ├── ontology.json            # 先隨便抓的，需要再替換成正確的 Ontology list
│   └── taxonomy_full_with_examples.json
├── models/                      # 預訓練模型
│   └── matrix_factorization/    # MF 模型文件
├── n8n_templates/                   # 所有的 templates
│   └── templates_?_Template_?.son   # templates 範例檔名
├── requirements.txt              # Python 依賴
├── example_usage.py              # 使用範例
└── README.md                     # 本文件
```

## 🚀 快速開始

### 步驟 1: 安裝依賴

```bash
pip install -r requirements.txt
```

**注意**: 首次安裝時會下載 SentenceTransformer 模型（約 400MB），請確保網絡連接正常。

### 步驟 2: 配置 OpenAI API Key

編輯 `n8n_workflow_recommender/config/config.yaml`，填入您的 OpenAI API Key:

```yaml
api:
  openai_key: "sk-..."  # 替換為您的 API Key
```

或者設置環境變量：

```bash
export OPENAI_API_KEY="sk-..."
```

### 步驟 3: 運行範例

```bash
python example_usage.py
```

## 💻 使用方式

### 方式 1: 使用範例腳本（推薦新手）

`example_usage.py` 是一個完整的示例腳本，展示了如何使用系統：

```bash
python example_usage.py
```

這個腳本會：
- 自動初始化 `WorkflowOrchestrator`
- 使用預設的查詢範例
- 生成工作流程並保存到 `output/generated_workflow.json`

**適合**: 快速測試系統是否正常工作

### 方式 2: 作為模組運行（快速測試）

`orchestrator.py` 包含 `main()` 函數，可以直接作為模組運行：

```bash
python -m n8n_workflow_recommender.core.orchestrator
```

這個方式會：
- 自動從 `config.yaml` 讀取 API Key
- 使用預設的測試查詢
- 生成工作流程並保存結果

**適合**: 快速測試，無需編寫代碼


# 初始化
orchestrator = WorkflowOrchestrator(openai_key="your-api-key")


# 生成工作流程
result = orchestrator.process_user_request(
    "設計智能信件處理流程，當有新gmail信件進來時自動觸發，使用 openai 理解信件內容，如果與開會相關，則提取開始時間、結束時間、地點存入google calendar。"
)

# 獲取結果
if "error" not in result:
    workflow_json = result["best_workflow"]["workflow_json"]
    # 保存或使用 workflow_json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(workflow_json, f, indent=2, ensure_ascii=False)
else:
    print(f"錯誤: {result['error']}")
```

**適合**: 整合到自己的應用中，處理自定義查詢

### 方式 4: 處理多個查詢

```python
orchestrator = WorkflowOrchestrator(openai_key="your-api-key")

queries = [
    "設計一個 OCR 流程，讀取圖片並識別文字",
    "創建一個自動發送郵件的流程",
    "設計一個數據同步流程"
]

for query in queries:
    result = orchestrator.process_user_request(query)
    if "error" not in result:
        # 處理結果...
        print(f"✅ 成功生成: {query}")
```

## ⚙️ 配置說明

主要配置文件: `n8n_workflow_recommender/config/config.yaml`

```yaml
api:
  openai_key: "your-api-key"  # OpenAI API Key

search:
  mcts:
    iterations: 2000          # MCTS 迭代次數
    top_n: 5                  # 返回前 N 個結果
  taxonomy:
    semantic_model: "paraphrase-multilingual-mpnet-base-v2"

generation:
  max_path_length: 12         # 最大路徑長度
  min_score_threshold: 0.2   # 最小分數閾值
```

## 📊 系統流程

1. **NLU 分析**: 使用 GPT-4o 分析用戶查詢，提取目標、參數和功能類別
2. **MCTS 搜索**: 在 taxonomy 中搜索相關節點類型
3. **A* 路徑生成**: 在知識圖中生成多個候選工作流程路徑
4. **MF 評分**: 使用 Matrix Factorization 模型對候選進行評分
5. **參數填充**: 自動填充節點參數
6. **JSON 生成**: 生成完整的 n8n 工作流程 JSON

## 🔧 依賴要求

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- networkx >= 2.6.0
- sentence-transformers >= 2.2.0
- torch >= 1.9.0
- openai >= 1.0.0
- pyyaml >= 6.0
- scikit-learn >= 1.0.0

## 📦 打包內容

本打包包含運行所需的所有文件：

- ✅ 所有必要的 Python 程式碼
- ✅ 預訓練的 Matrix Factorization 模型
- ✅ 知識圖數據 (adapted_knowledge_graph.json)
- ✅ 節點映射數據 (node_mappings.json)
- ✅ Taxonomy 數據 (taxonomy_full_with_examples.json)
- ✅ 配置文件和使用範例

## ⚠️ 注意事項

1. **OpenAI API Key**: 必須配置有效的 OpenAI API Key 才能使用 NLU 功能
2. **模型文件**: Matrix Factorization 模型文件已包含在 `models/` 目錄中
3. **數據文件**: 所有必要的數據文件已包含在 `data/` 目錄中
4. **首次運行**: 首次運行時會下載 SentenceTransformer 模型（約 400MB），請確保網絡連接正常
5. **路徑配置**: 所有路徑都是相對於打包目錄根目錄的，請不要移動或重命名子目錄

## 🐛 問題排查

### 問題 1: 找不到數據文件

**解決方案**: 確保以下文件存在：

```bash
# 檢查數據文件
ls data/adapted_knowledge_graph.json
ls data/node_mappings.json
ls data/taxonomy_full_with_examples.json

# 檢查模型文件
ls models/matrix_factorization/type_type_mapping.json
ls models/matrix_factorization/*.npy
```

**確保**: 在打包目錄的根目錄運行腳本，不要移動或重命名子目錄

### 問題 2: OpenAI API 錯誤

**檢查清單**:
- [ ] API Key 是否正確配置（在 `config.yaml` 或環境變量中）
- [ ] API Key 是否有足夠的額度
- [ ] 網絡連接是否正常

### 問題 3: 模型加載失敗

**解決方案**: 確保 `models/matrix_factorization/` 目錄中包含所有必要的 `.npy` 和 `.json` 文件

### 問題 4: 首次運行很慢

**原因**: 首次運行時會下載 SentenceTransformer 模型（約 400MB）

**解決方案**: 這是正常現象，請耐心等待。下載完成後會緩存，後續運行會更快

### 問題 5: Import 錯誤

**解決方案**: 
- 確保在打包目錄的根目錄運行腳本
- 確保已安裝所有依賴：`pip install -r requirements.txt`
- 確保 Python 版本 >= 3.8

## 📊 系統要求

- **Python**: 3.8 或更高版本
- **磁盤空間**: 至少 2GB（用於模型和數據）
- **內存**: 建議 4GB 或更多
- **網絡**: 首次運行需要下載模型（約 400MB）

## 📝 文件說明

### example_usage.py vs orchestrator.py

- **`orchestrator.py`**: 核心類 `WorkflowOrchestrator`，提供 `process_user_request()` 方法
  - 這是系統的核心功能類
  - 負責協調 NLU、MCTS、A*、MF 評分等所有組件
  - 包含 `main()` 函數，可以作為模組運行：`python -m n8n_workflow_recommender.core.orchestrator`
  - 適合在代碼中導入使用，或直接作為模組運行

- **`example_usage.py`**: 使用範例腳本，展示如何使用 `WorkflowOrchestrator`
  - 包含完整的初始化、調用、結果處理流程
  - 包含預設的查詢範例
  - 適合快速測試和學習如何使用系統

**簡單來說**: 
- `orchestrator.py` 是工具類，可以作為模組運行（`python -m`）或在代碼中導入使用
- `example_usage.py` 是使用範例腳本，展示完整的使用流程

