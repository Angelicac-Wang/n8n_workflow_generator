# Evaluation Metrics Documentation

This document describes the three evaluation metrics used to assess LLM-generated n8n workflows against ground truth templates.

## Overview

All metrics are based on **node types** (not node names) to ensure fair comparison, as LLMs may use different naming conventions.

---

## 1. Node Type Accuracy

**What it measures**: Whether the LLM generates the correct types of nodes.

### Implementation
- **Location**: [node_accuracy_evaluator.py:16-49](evaluation/evaluators/node_accuracy_evaluator.py#L16-L49)
- **Matching strategy**: Greedy matching by node type
  - Groups nodes by type
  - Matches same-type nodes sequentially

### Node Type Normalization
- Strips namespace prefix: `n8n-nodes-base.httpRequest` → `httprequest`
- Converts to lowercase: `openAi` → `openai`
- Applies aliases (e.g., `httpRequest`, `http` both map to `httprequest`)

### Metrics Calculation
```
TP (True Positive)  = Successfully matched nodes with same type
FP (False Positive) = LLM nodes with no GT match
FN (False Negative) = GT nodes with no LLM match

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

### What is NOT evaluated
- Node names
- Node parameters
- Node positions
- Node IDs

---

## 2. Connection Accuracy

**What it measures**: Whether the LLM correctly connects node types.

### Implementation
- **Location**: [node_accuracy_evaluator.py:51-113](evaluation/evaluators/node_accuracy_evaluator.py#L51-L113)
- **Comparison basis**: Node types (not node names)

### Process
1. Convert each connection to a `(from_type, to_type)` tuple
   - Example: `(httprequest, openai)` means httpRequest node connects to openai node
2. Use lowercase normalized types (same as Node Type Accuracy)
3. Calculate set intersection to find matching connections

### Metrics Calculation
```
Correct Connections = GT connection set ∩ LLM connection set

Precision = Correct / Total LLM connections
Recall    = Correct / Total GT connections
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

### Key Features
- **ID to Name Resolution**: LLM connections use node IDs ("1", "2"), which are mapped to node names
- **Name to Type Resolution**: Node names are then mapped to node types for comparison
- **Case-insensitive**: All types are lowercase

### Example
```
LLM:  telegramtrigger → httprequest → openai
GT:   telegramtrigger → httprequest → openai → googlesheets

Matching: (telegramtrigger, httprequest), (httprequest, openai)
Correct: 2/2 LLM connections = 100% Precision
         2/3 GT connections = 66.7% Recall
```

---

## 3. Parameter Accuracy (Slots Filling)

**What it measures**: Whether the LLM fills node parameters with semantically correct values.

### Implementation
- **Location**: [parameter_evaluator.py](evaluation/evaluators/parameter_evaluator.py)
- **Model**: `paraphrase-multilingual-mpnet-base-v2` (Sentence Transformer)

### Comparison Scope
- Only **top-level scalar parameters** (string, number, boolean)
- Excludes nested objects and arrays

### Process
1. Match nodes between GT and LLM (same matching as Node Type Accuracy)
2. For each matched node pair:
   - Extract scalar parameters
   - Compute embedding similarity for parameter keys and values
   - Parameters with similarity ≥ 0.8 are considered correct

### Metrics Calculation
```
For each matched node:
  Parameter Match Ratio = Correct params / Total params

Average Parameter Accuracy = Mean of all node match ratios
```

### Similarity Threshold
- **0.8 (cosine similarity)** is the threshold for parameter match
- Allows semantic equivalence (e.g., "https://api.openai.com/v1" ≈ "https://api.openai.com/v1/chat")

---

## Special Handling: StickyNote Nodes

**Problem**: 96.1% of ground truth templates contain `stickyNote` nodes (total: 6,148 nodes across 1,000 templates). These are comments/annotations that LLMs should not generate.

**Solution**: Automatically filter out all `stickyNote` nodes and their connections during normalization.

### Implementation
- **Location**: [workflow_normalizer.py:47-48](evaluation/comparison/workflow_normalizer.py#L47-L48)
- Filters nodes: `if 'stickynote' in full_type.lower(): continue`
- Filters connections: Only includes connections between valid (non-stickyNote) nodes

### Impact
Without filtering:
```
Template 10000:
  Total nodes: 35 (including 7 stickyNote)
  Node Type F1: 0.279
```

With filtering:
```
Template 10000:
  Total nodes: 28 (stickyNote removed)
  Node Type F1: 0.389 (+39% improvement)
```

---

## Data Flow

```
Ground Truth Template          LLM Generated Workflow
        ↓                              ↓
   Normalizer                     Normalizer
   - Remove stickyNote            - Parse nodes/connections
   - Extract nodes                - ID → Name resolution
   - Extract connections          - Type normalization
        ↓                              ↓
  Standardized Format            Standardized Format
  - nodes (type lowercase)       - nodes (type lowercase)
  - connections (by name)        - connections (by name)
        ↓                              ↓
        └──────────┬──────────────────┘
                   ↓
           Node Matcher (Greedy)
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
   Node Type   Connection  Parameter
   Evaluator   Evaluator   Evaluator
        ↓          ↓          ↓
    Precision  Precision   Semantic
    Recall     Recall      Similarity
    F1         F1          Average
```

---

## Example Results

Template 10000 (with stickyNote filtering):

| Metric | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| **Node Type** | 0.875 | 0.250 | 0.389 |
| **Connection** | 0.333 | 0.053 | 0.091 |
| **Parameter** | - | - | 0.161 |

**Interpretation**:
- LLM generated 8 nodes vs 28 GT nodes (28.6% coverage)
- 87.5% of LLM nodes are correct types (high precision)
- Only 25% of GT nodes were generated (low recall)
- Connection patterns partially match (1/3 correct)
- Parameter filling needs improvement

---

## Cost Tracking

In addition to accuracy metrics, the system tracks:
- **Prompt tokens**: Input to GPT-4o
- **Completion tokens**: Output from GPT-4o
- **Total cost**: Based on GPT-4o pricing ($2.50/1M input, $10.00/1M output)

**Location**: [cost_tracker.py](evaluation/evaluators/cost_tracker.py)

---

## Configuration

Key settings in [evaluation_config.yaml](evaluation/config/evaluation_config.yaml):

```yaml
# Model settings
model: "gpt-4o"
temperature: 0.3

# Evaluation settings
embedding_model: "paraphrase-multilingual-mpnet-base-v2"
param_similarity_threshold: 0.8

# Processing
resume_enabled: true  # Skip already-generated templates
```

---

## Output Files

### Per-template results
- `outputs/llm_generated_workflows/generated_{template_id}.json`
- `outputs/evaluation_results/detailed_per_template.json`
- `outputs/evaluation_results/detailed_per_template.csv`

### Aggregate statistics
- `outputs/evaluation_results/summary_statistics.json`
- `outputs/evaluation_results/cost_report.json`

### Visualizations
- `outputs/visualizations/node_accuracy_distribution.png`
- `outputs/visualizations/connection_accuracy_distribution.png`
- `outputs/visualizations/parameter_accuracy_distribution.png`
- `outputs/visualizations/cost_analysis.png`
- `outputs/visualizations/comparison_heatmap.png`
- `outputs/visualizations/metric_correlations.png`

---

## Design Decisions Summary

1. **Type-based comparison** (not name-based): Allows LLMs to use different naming
2. **Lowercase normalization**: Handles `openAi` vs `openai` inconsistencies
3. **StickyNote filtering**: Removes annotation nodes that shouldn't be evaluated
4. **Greedy matching**: Simple, fast node matching strategy
5. **Semantic similarity for parameters**: Uses embeddings instead of exact string match
6. **Independent evaluation**: Can re-run evaluation without re-generating workflows
7. **Resume capability**: Skip already-processed templates to save API costs
