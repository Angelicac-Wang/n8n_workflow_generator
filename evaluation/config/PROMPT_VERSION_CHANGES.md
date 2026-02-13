# Workflow Generation Prompt - Version History

## Version 4 (v4) - Refined Detection + httpRequest Fix (Current)

**File**: `workflow_generation_prompt_improved.txt`
**Backup**: `workflow_generation_prompt_improved_v3.txt` (previous version)
**Date**: 2026-02-05
**Goal**: Fix v3 issues - tighten detection, reduce httpRequest overuse, reach Node F1 = 0.47+

### Critical Problems in v3

Analysis of v3 results revealed three major issues:

1. **AI Workflow Detection Too Broad**
   - Detected 53 workflows as "AI workflows" (should be ~10)
   - Non-AI workflows performed better (0.481) than "AI workflows" (0.412)
   - Many regressions due to mis-classification

2. **httpRequest Explosion**
   - v2: 7 unnecessary uses
   - v3: 94 unnecessary uses (13x increase!)
   - LLM defaulting to httpRequest instead of specific integration nodes

3. **Still Missing Key Nodes**
   - Agent: 37 missing (v2: 26) - worse!
   - Set: 35 missing (v2: 20) - worse!
   - Code: 32 missing (v2: 50) - improved

### Major Changes in v4

#### 1. **Tightened AI Workflow Detection** (CRITICAL)

**v3 Detection** (Too broad):
```
Any of: "chat", "AI", "agent", "conversational", "RAG", etc.
Result: 53 detected
```

**v4 Detection** (Much stricter):
```
BOTH required:
1. Conversational pattern: "chat with", "chatbot that", "Q&A about"
2. AND memory pattern: "memory", "context", "history", "RAG"

Result: Expected ~8-12 detected
```

**Specific changes**:
- Removed generic keywords: "AI", "intelligent", "smart", "optimization"
- Require BOTH conversational AND memory indicators
- Added explicit "NOT LangChain" examples
- Updated reasoning field to check both conditions

#### 2. **httpRequest Overuse Prevention** (NEW)

Added comprehensive section **#2 in Common Mistakes**:

```markdown
### 2. ❌ CRITICAL: Overusing httpRequest

**WRONG**: httpRequest → Slack, Google Sheets, GitHub, Database
**CORRECT**: Use dedicated nodes (slack, googleSheets, github, postgres)

**When to use httpRequest**:
✅ Custom/uncommon APIs without n8n integration
❌ NOT for any service with a dedicated node
```

**Added to Checklist**:
- Check if specific integration nodes exist first
- Avoid httpRequest overuse
- Verify node types are real

**Updated Common Node Types**:
- Listed specific nodes for Storage, Communication, Project Management
- Emphasized "use these, NOT httpRequest"
- httpRequest moved to end with "ONLY when no dedicated node exists"

#### 3. **Strengthened Data Transformation Nodes**

Updated **#4 in Common Mistakes**:
- Added statistics: "Set: 35 missing, Code: 32 missing"
- Added pattern example: `API → Set/Code (transform) → Next step`
- Emphasized NOT to skip transformation steps

**Added to Checklist**:
- Explicit item: "Include Set/Code nodes" between steps
- Don't skip data transformation

#### 4. **Updated Examples**

All three base examples now include `workflow_type_check` in reasoning:
- Example 1: "NO - simple automation"
- Example 2: "NO - email automation with conditional logic"
- Example 3: "NO - automation with AI analysis (single API call)"
- Example 4 (AI Agent): "YES - conversational + memory + context"

### Test Script Changes

**`test_method_c_v4.py`** - Updated detection logic:

```python
# v3: Single condition check
ai_agent_patterns = ['chat with', 'chatbot', 'rag', ...]
is_ai = any(pattern in text for pattern in patterns)

# v4: BOTH conditions required
conversational = ['chat with', 'chatbot that', 'q&a about', ...]
memory = ['memory', 'context', 'history', 'rag', ...]
is_ai = has_conversational AND has_memory
```

### Expected Impact

**Based on v3 analysis**:

```
v3 Results:
- Overall Node F1: 0.445
- AI workflows (53): 0.412
- Non-AI workflows (47): 0.481

Expected v4:
- True AI workflows (~10): 0.60+ (maintained improvements)
- Falsely detected (43): 0.412 → 0.48 (recovered)
- Non-AI workflows (47): 0.481 (maintained)

Weighted average:
(10 × 0.60 + 43 × 0.48 + 47 × 0.481) / 100 = 0.471

Target: 0.47-0.48 (94% of 0.5 goal)
```

**httpRequest reduction**:
- v3: 94 unnecessary
- v4 target: <20 unnecessary (80% reduction)

### Files Modified
- `evaluation/config/workflow_generation_prompt_improved.txt`
- Backup: `evaluation/config/workflow_generation_prompt_improved_v3.txt`

### Files Created
- `scripts/test_method_c_v4.py`
- `scripts/analyze_method_c_v3_gaps.py`
- `0205_enhance_with_checking_differences/V2_VS_V3_ANALYSIS.md`

### Testing

Run comparison test:
```bash
cd n8n_workflow_generator
python3 scripts/test_method_c_v4.py
```

---

## Version 3 (v3) - LangChain Architecture Focus

**File**: `workflow_generation_prompt_improved.txt`
**Backup**: `workflow_generation_prompt_improved_v2.txt` (previous version)
**Date**: 2026-02-05
**Goal**: Fix AI agent workflow generation to reach Node F1 = 0.50

### Critical Problem Identified

Analysis of Method C v2 results revealed:
- **4 workflows with F1 = 0.000** → ALL were AI agent workflows
- **15 workflows with F1 < 0.2** → ALL contained LangChain nodes
- **87 LangChain nodes missed** across these workflows
- **Root cause**: LLM uses wrong node namespace (`n8n-nodes-base.*` instead of `@n8n/n8n-nodes-langchain.*`)

### Major Changes in v3

#### 1. AI Agent Workflow Detection Section (NEW)
**Location**: After "Step-by-Step Process", before "Few-Shot Examples"

Added comprehensive decision tree:
- **Detection Keywords**: List of AI/chat/agent/RAG keywords
- **Architecture Diagram**: Visual representation of LangChain node structure
- **Critical Node Mappings Table**: Side-by-side comparison of WRONG vs CORRECT nodes
- **Available LLM Models**: Complete list of LangChain LLM options
- **Required Components**: Checklist for AI agent workflows

**Key mapping taught**:
```
❌ n8n-nodes-base.webhook → ✅ @n8n/n8n-nodes-langchain.chatTrigger
❌ n8n-nodes-base.openAi   → ✅ @n8n/n8n-nodes-langchain.lmChatOpenAi
Missing entirely           → ✅ @n8n/n8n-nodes-langchain.agent
```

#### 2. AI Agent Few-Shot Example (NEW)
**Location**: Added as Example 4 after existing examples

Complete working example of:
- Database chat assistant using LangChain architecture
- Proper use of chatTrigger, agent, lmChatOpenAi, memory, postgresTool
- Detailed reasoning showing WHY each node type is chosen
- Key points section highlighting correct vs incorrect approaches

#### 3. Reorganized "Common Mistakes" Section
**Location**: Updated priority order

**NEW #1 Mistake**: Using wrong nodes for AI agent workflows
- Moved from #3/#4 to #1 (highest priority)
- Added specific failure statistics (87 missing nodes, F1=0 cases)
- Visual comparison of wrong vs correct approaches
- Detection keywords and patterns

Original mistakes moved to positions #2-#6

#### 4. Enhanced Pre-Generation Checklist
**Location**: Added AI workflows as FIRST checklist section

New section: "🤖 AI Agent Workflows (CHECK FIRST!)"
- 6 critical checks for AI workflows
- Explicit reminders about using LangChain nodes
- Warning against using base nodes for agents

#### 5. Expanded "Commonly Needed Node Types"
**Location**: Enhanced AI & LangChain section

Added comprehensive LangChain node reference:
- **Core LangChain Nodes**: All agent components
- **LangChain Tools**: Tool versions of base nodes
- **RAG & Document Processing**: Vector stores, embeddings, text splitters
- **Output Parsers**: Structured output handling
- Complete list of LLM model options (OpenAI, Anthropic, Google, Ollama, DeepSeek)

#### 6. Updated Key Guidelines
Added AI workflow check as #1 guideline before all others

### Specific Node Types Added to Prompt

**Core Nodes**:
- `@n8n/n8n-nodes-langchain.chatTrigger`
- `@n8n/n8n-nodes-langchain.agent`
- `@n8n/n8n-nodes-langchain.lmChatOpenAi`
- `@n8n/n8n-nodes-langchain.lmChatAnthropic`
- `@n8n/n8n-nodes-langchain.lmChatGoogleGemini`
- `@n8n/n8n-nodes-langchain.lmChatOllama`
- `@n8n/n8n-nodes-langchain.lmChatDeepSeek`
- `@n8n/n8n-nodes-langchain.memoryBufferWindow`
- `@n8n/n8n-nodes-langchain.memoryBufferMemory`

**Tools**:
- `n8n-nodes-base.postgresTool`
- `n8n-nodes-base.httpRequestTool`
- `n8n-nodes-base.calculatorTool`
- `@n8n/n8n-nodes-langchain.toolCode`
- `@n8n/n8n-nodes-langchain.toolWorkflow`

**RAG Components**:
- `@n8n/n8n-nodes-langchain.documentDefaultDataLoader`
- `@n8n/n8n-nodes-langchain.embeddingsOpenAi`
- `@n8n/n8n-nodes-langchain.vectorStoreQdrant`
- `@n8n/n8n-nodes-langchain.textSplitterRecursiveCharacterTextSplitter`
- `@n8n/n8n-nodes-langchain.chainLlm`
- `@n8n/n8n-nodes-langchain.chainSummarization`
- `@n8n/n8n-nodes-langchain.outputParserStructured`

### Expected Impact

**Baseline (Method C v2)**:
- Node F1: 0.415
- Connection F1: 0.189
- Parameter Accuracy: 0.139
- 4 workflows with F1 = 0.000 (all AI agent workflows)
- 15 workflows with F1 < 0.2

**Target (Method C v3)**:
- **Node F1: 0.50 - 0.53** (+0.085 to +0.115) ← REACHES GOAL!
- Connection F1: 0.22+ (+0.03)
- Workflows with F1 = 0.000: 0 (eliminate all)
- Workflows with F1 < 0.2: 5-8 (reduce by 50%)

### Calculation
- 15 AI workflows improve from avg 0.15 → 0.50 = +0.35 each
- Impact: (15 × 0.35) / 100 = +0.0525
- Spillover to other AI workflows: +0.03
- **Total improvement: +0.08 to +0.12**

### Testing

Run comparison test:
```bash
cd n8n_workflow_generator
python3 scripts/test_method_c_v3.py
```

This will:
- Load existing v2 results for comparison
- Generate v3 results with new prompt
- Compare overall performance
- Analyze AI workflow improvement specifically
- Output to `outputs/method_c_v3_comparison/`

### Files Modified
- `evaluation/config/workflow_generation_prompt_improved.txt` (main prompt)
- Backup created: `evaluation/config/workflow_generation_prompt_improved_v2.txt`

### Files Created
- `scripts/test_method_c_v3.py` (test script)
- `0205_enhance_with_checking_differences/CRITICAL_FINDINGS.md` (analysis)
- `0205_enhance_with_checking_differences/PROMPT_V3_IMPROVEMENTS.md` (plan)

---

## Version 2 (v2) - Enhanced with Gap Analysis

**File**: `workflow_generation_prompt_improved.txt`
**Date**: 2026-02-05
**Goal**: Improve Method C Node F1 from 0.371 to 0.5

### Changes Based on Analysis of 100 Method C Results

#### Analysis Findings
- Analyzed 100 workflows generated by Method C
- Key issues identified:
  - LLM generates ~80% fewer nodes than needed (avg: 7 nodes vs 19 in ground truth)
  - Missing 50 Code nodes, 20 Set nodes across workflows
  - Missing 26 AI Agent nodes in AI workflows
  - Generates 48% fewer connections than needed
  - Low parameter accuracy (0.146 avg)

#### New Sections Added

1. **⚠️ CRITICAL: Common Mistakes to Avoid**
   - Section explaining the 6 most common errors found in analysis
   - Specific guidance on avoiding each mistake
   - Based on actual data from 100 workflow comparisons

2. **📋 Pre-Generation Checklist**
   - Comprehensive checklist before generating workflow
   - Organized by categories: Structure, Data Flow, Error Handling, AI Workflows, Connections
   - Forces LLM to verify completeness before generating

3. **🔧 Commonly Needed Node Types**
   - Quick reference for frequently missing node types
   - Categorized by function: Data Processing, Control Flow, AI & LangChain, Response Handling
   - Includes specific LangChain node types that were commonly missed

#### Specific Improvements

**1. Addresses "Too Few Nodes" Problem**
- Added explicit warning that workflows need intermediate steps
- Guidance on when to use Set vs Code nodes
- Examples of what not to skip (data transformation, error handling, validation)

**2. AI/LangChain Workflow Support**
- Added dedicated section for AI agent workflows
- Clarified @n8n/n8n-nodes-langchain.* vs n8n-nodes-base.* distinction
- Listed commonly missing LangChain nodes:
  - agent (missing 26 times)
  - lmChatOpenAi (missing 18 times)
  - chatTrigger, memoryBufferWindow, toolCode

**3. Node Type Disambiguation**
- Corrected common node type confusions:
  - openAi vs lmChatOpenAi
  - function vs code (function doesn't exist)
  - schedule vs scheduleTrigger
  - webhook vs webhookTrigger

**4. Connection Completeness**
- Added guidance that workflows need ~11 connections on average
- Emphasis on connecting all branches (success, error, alternative)
- Warning about orphaned nodes

**5. Error Handling Emphasis**
- Specific checklist items for error handling
- Examples of what to handle: API failures, data validation, fallbacks

#### Key Metrics We're Targeting

**Current Performance (Method C v1)**:
- Node F1: 0.371
- Connection F1: 0.172
- Parameter Accuracy: 0.146

**Target Performance (Method C v2)**:
- Node F1: 0.50+ (35% improvement)
- Connection F1: 0.25+ (45% improvement)
- Parameter Accuracy: 0.20+ (37% improvement)

#### Top 5 Most Commonly Missing Node Types (from analysis)

1. `n8n-nodes-base.code` - Missing 50 times
2. `@n8n/n8n-nodes-langchain.agent` - Missing 26 times
3. `n8n-nodes-base.set` - Missing 20 times
4. `@n8n/n8n-nodes-langchain.lmChatOpenAi` - Missing 18 times
5. `n8n-nodes-base.httpRequest` - Missing 14 times (even though also added unnecessarily in other cases)

#### Implementation Details

**Placement of New Content**:
- Common Mistakes section: Before Few-Shot Examples (lines ~397-443)
- Pre-Generation Checklist: After Examples, before Key Guidelines (lines ~446-475)
- Commonly Needed Node Types: After Checklist (lines ~478-500)

**No Changes to**:
- Few-shot examples (kept intact - they're already good)
- Response format specification
- Chain-of-thought reasoning structure
- Output rules

---

## Version 1 (v1) - Initial Improved Prompt

**File**: `workflow_generation_prompt_improved_v1.txt` (backup)
**Date**: 2026-01-25 (original implementation)

### Features
- Chain-of-thought reasoning field
- Three complexity-level few-shot examples
- Structured output format
- Basic guidelines

### Performance
- Node F1: 0.371
- Connection F1: 0.172
- Parameter Accuracy: 0.146
- Total Cost: $0.0240

### Strengths
- Good few-shot examples
- Clear reasoning structure
- Proper JSON format enforcement

### Limitations (Addressed in v2)
- No guidance on common mistakes
- No checklist to prevent oversimplification
- Missing LangChain node type guidance
- No emphasis on workflow completeness

---

## Base Version (v0) - Original Base Prompt

**File**: `workflow_generation_prompt.txt`
**Date**: Earlier

### Performance (Method A)
- Node F1: 0.354
- Connection F1: 0.165
- Parameter Accuracy: 0.111
- Total Cost: $0.0132

### Features
- Basic workflow generation instructions
- Minimal examples
- Simple format requirements

---

## Usage

### For Evaluation/Testing
```python
from evaluation.generators.prompt_builder import PromptBuilder

# Use v2 (current improved version)
prompt_builder = PromptBuilder('evaluation/config/workflow_generation_prompt_improved.txt', use_improved=True)

# Use v1 (for comparison)
prompt_builder = PromptBuilder('evaluation/config/workflow_generation_prompt_improved_v1.txt', use_improved=True)

# Use base version
prompt_builder = PromptBuilder('evaluation/config/workflow_generation_prompt.txt', use_improved=False)
```

### For Production
Always use the latest version (`workflow_generation_prompt_improved.txt`) which includes all improvements.

---

## Future Improvements (Ideas)

1. **Add more complex examples**: Include 20+ node workflow examples
2. **Domain-specific guidance**: Special sections for e-commerce, data pipelines, etc.
3. **Parameter templates**: Common parameter patterns for frequently used nodes
4. **Visual flow patterns**: Describe common workflow patterns (ETL, event-driven, etc.)
5. **Validation rules**: Built-in validation logic to check generated workflows

---

## Analysis References

- Gap analysis report: `/Users/yu/Desktop/projects/gss_cai/n8n_workflow_generator/0205_enhance_with_checking_differences/analyze_method_C_gap.txt`
- Method C results: `outputs/four_methods_comparison/method_c_results.json`
- Analysis script: `analysis/analyze_method_c_patterns.py`
