# Method C v3 Prompt Improvements - LangChain Focus

## Goal

Improve Node F1 from **0.415** → **0.50+** by fixing LangChain node recognition

## Root Cause Analysis

**Problem**: LLM uses wrong node namespace for AI agent workflows

**Evidence**:
- 15/15 low-performing cases (F1 < 0.2) contain LangChain nodes
- 87 LangChain nodes missed across these workflows
- LLM defaults to `n8n-nodes-base.*` instead of `@n8n/n8n-nodes-langchain.*`

## Proposed Changes to Prompt

### 1. Add "AI Agent Workflow Decision Tree" (HIGH PRIORITY)

**Location**: Add RIGHT AFTER "Step-by-Step Process" section (before Few-Shot Examples)

```markdown
---

## 🤖 CRITICAL: AI Agent Workflow Detection

**BEFORE designing any workflow, check if it's an AI Agent workflow:**

### Detection Keywords:
If the user request contains ANY of these keywords:
- "AI agent", "chat", "chatbot", "conversational"
- "RAG", "retrieval", "vector store", "embeddings"
- "LLM", "language model", "OpenAI", "Claude", "GPT"
- "AI assistant", "intelligent", "natural language"
- "memory", "context", "conversation history"

### Then → Use LangChain Architecture

**Architecture Pattern for AI Agent Workflows:**

```
┌─────────────────┐
│  Chat Trigger   │ @n8n/n8n-nodes-langchain.chatTrigger
│  (User Input)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Agent       │ @n8n/n8n-nodes-langchain.agent
│  (Orchestrator) │ ← Core node for AI decision-making
└────────┬────────┘
         │
         ├─→ LM Chat Model      (@n8n/n8n-nodes-langchain.lmChatOpenAi)
         ├─→ Memory             (@n8n/n8n-nodes-langchain.memoryBufferWindow)
         ├─→ Tools              (@n8n/n8n-nodes-langchain.tool*)
         └─→ Output Parser      (@n8n/n8n-nodes-langchain.outputParserStructured)
```

### Required Components for AI Agent Workflows:

1. **Trigger**: Use `@n8n/n8n-nodes-langchain.chatTrigger` (NOT n8n-nodes-base.webhook)
2. **Agent**: Use `@n8n/n8n-nodes-langchain.agent` (NOT n8n-nodes-base.openAi)
3. **LLM**: Use `@n8n/n8n-nodes-langchain.lmChatOpenAi` (NOT n8n-nodes-base.openAi)
4. **Tools**: Use `@n8n/n8n-nodes-langchain.tool*` versions (NOT base nodes)

---
```

### 2. Add LangChain Node Mapping Table (HIGH PRIORITY)

**Location**: Add in "Commonly Needed Node Types" section

```markdown
### 🔄 LangChain vs Base Nodes - Critical Differences

**IMPORTANT**: For AI agent workflows, you MUST use `@n8n/n8n-nodes-langchain.*` nodes.

| Use Case | ❌ WRONG (Base Node) | ✅ CORRECT (LangChain Node) |
|----------|---------------------|----------------------------|
| Chat interface | `n8n-nodes-base.webhook` | `@n8n/n8n-nodes-langchain.chatTrigger` |
| AI chat model | `n8n-nodes-base.openAi` | `@n8n/n8n-nodes-langchain.lmChatOpenAi` |
| Agent orchestration | (missing) | `@n8n/n8n-nodes-langchain.agent` |
| Database tool | `n8n-nodes-base.postgres` | `n8n-nodes-base.postgresTool` |
| API tool | `n8n-nodes-base.httpRequest` | `@n8n/n8n-nodes-langchain.toolCode` or `toolWorkflow` |
| Memory/Context | (missing) | `@n8n/n8n-nodes-langchain.memoryBufferWindow` |
| Document loading | (missing) | `@n8n/n8n-nodes-langchain.documentDefaultDataLoader` |
| Embeddings | (missing) | `@n8n/n8n-nodes-langchain.embeddingsOpenAi` |
| Vector store | (missing) | `@n8n/n8n-nodes-langchain.vectorStore*` |

### Available LangChain LLM Models:
- `@n8n/n8n-nodes-langchain.lmChatOpenAi` - For GPT-4, GPT-3.5
- `@n8n/n8n-nodes-langchain.lmChatAnthropic` - For Claude
- `@n8n/n8n-nodes-langchain.lmChatGoogleGemini` - For Gemini
- `@n8n/n8n-nodes-langchain.lmChatOllama` - For local LLMs
```

### 3. Add AI Agent Few-Shot Example (HIGH PRIORITY)

**Location**: Add as "Example 4" after existing examples

```markdown
### Example 4: AI Agent Workflow with Database (LangChain Architecture)

**User Request:**
"Create an AI chatbot that can query a PostgreSQL database using natural language. The bot should remember conversation history and respond intelligently."

**Expected Response:**
{
  "mode": "create_workflow",
  "reasoning": {
    "trigger_analysis": "AI agent workflow detected (keywords: AI chatbot, natural language). Must use @n8n/n8n-nodes-langchain.chatTrigger for interactive chat interface.",
    "data_sources": "PostgreSQL database accessed via LangChain tool",
    "operations": "1. Receive chat input, 2. Agent processes query with LLM, 3. Tool accesses database, 4. Return formatted response",
    "output_actions": "Chat response back to user",
    "nodes_plan": "chatTrigger (input), agent (orchestrator), lmChatOpenAi (reasoning), memoryBufferWindow (context), postgresTool (database access)",
    "connections_plan": "chatTrigger → agent, agent connects to: lmChatOpenAi, memoryBufferWindow, postgresTool"
  },
  "workflowPlan": {
    "name": "AI Database Chat Assistant",
    "nodes": [
      {
        "id": "1",
        "label": "Chat Trigger",
        "nodeType": "@n8n/n8n-nodes-langchain.chatTrigger",
        "typeVersion": 1,
        "params": {
          "mode": "chat"
        },
        "position": { "x": 100, "y": 300 }
      },
      {
        "id": "2",
        "label": "AI Agent",
        "nodeType": "@n8n/n8n-nodes-langchain.agent",
        "typeVersion": 1,
        "params": {
          "promptType": "define",
          "text": "You are a helpful database assistant. Answer user questions about the database."
        },
        "position": { "x": 300, "y": 300 }
      },
      {
        "id": "3",
        "label": "OpenAI Chat Model",
        "nodeType": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
        "typeVersion": 1,
        "params": {
          "model": "gpt-4",
          "temperature": 0.7
        },
        "position": { "x": 500, "y": 200 }
      },
      {
        "id": "4",
        "label": "Conversation Memory",
        "nodeType": "@n8n/n8n-nodes-langchain.memoryBufferWindow",
        "typeVersion": 1,
        "params": {
          "windowSize": 10
        },
        "position": { "x": 500, "y": 300 }
      },
      {
        "id": "5",
        "label": "Postgres Tool",
        "nodeType": "n8n-nodes-base.postgresTool",
        "typeVersion": 1,
        "params": {
          "database": "my_database",
          "host": "localhost"
        },
        "position": { "x": 500, "y": 400 }
      }
    ],
    "connections": [
      { "from": "1", "to": "2", "outputIndex": 0, "inputIndex": 0 },
      { "from": "2", "to": "3", "outputIndex": 0, "inputIndex": 0 },
      { "from": "2", "to": "4", "outputIndex": 0, "inputIndex": 0 },
      { "from": "2", "to": "5", "outputIndex": 0, "inputIndex": 0 }
    ]
  }
}
```

### 4. Update "Common Mistakes" Section (MEDIUM PRIORITY)

Add this as **#1 mistake** (move others down):

```markdown
## ⚠️ CRITICAL: Common Mistakes to Avoid

### 1. ❌ MOST CRITICAL: Using Base Nodes for AI Agent Workflows (87% of F1=0 cases)

**Mistake**: Using `n8n-nodes-base.openAi` and `n8n-nodes-base.webhook` for AI chatbots

**Why it fails**: AI agent workflows require LangChain architecture with:
- Agent node for orchestration
- Memory nodes for conversation history
- Tool nodes for capabilities
- Proper LLM chat models

**How to fix**:
- IF workflow involves "AI agent", "chat", "RAG", "conversational" → Use `@n8n/n8n-nodes-langchain.*`
- Always include: `chatTrigger`, `agent`, `lmChatOpenAi`, `memory*`
- Use tool versions: `postgresTool`, `toolWorkflow`, etc.

**Example Detection**:
- ✅ "Build a chatbot to query database" → LangChain architecture
- ✅ "AI assistant with memory" → LangChain architecture
- ✅ "RAG workflow for documents" → LangChain architecture
- ❌ "Call OpenAI API to generate text" → Can use base openAi node

[Keep existing mistakes as #2, #3, etc.]
```

### 5. Update Pre-Generation Checklist (MEDIUM PRIORITY)

Add to existing checklist:

```markdown
## 📋 Pre-Generation Checklist

Before generating the workflow JSON, verify your plan includes:

**🤖 For AI Agent Workflows** (check first!):
- [ ] Using `@n8n/n8n-nodes-langchain.chatTrigger` for chat interface
- [ ] Including `@n8n/n8n-nodes-langchain.agent` as orchestrator
- [ ] Using `@n8n/n8n-nodes-langchain.lmChatOpenAi` (or other LLM) for reasoning
- [ ] Adding memory node if conversation context needed
- [ ] Using tool versions of nodes (e.g., `postgresTool`, not `postgres`)

[Keep existing checklist items...]
```

### 6. Add RAG Workflow Pattern (MEDIUM PRIORITY)

**Location**: Add after AI Agent example

```markdown
### Common AI Agent Patterns:

#### Pattern 1: Simple Chat Agent
```
chatTrigger → agent → lmChatOpenAi
```

#### Pattern 2: Agent with Memory
```
chatTrigger → agent → [lmChatOpenAi, memoryBufferWindow]
```

#### Pattern 3: Agent with Tools
```
chatTrigger → agent → [lmChatOpenAi, memoryBufferWindow, tool1, tool2, ...]
```

#### Pattern 4: RAG (Retrieval-Augmented Generation)
```
documentDefaultDataLoader → textSplitter → embeddingsOpenAi → vectorStore
                                                                      ↓
chatTrigger → agent → [lmChatOpenAi, memoryBufferWindow, vectorStoreRetriever]
```
```

## Implementation Plan

### Phase 1: Quick Wins (Implement First)
1. ✅ Add AI Agent Decision Tree section
2. ✅ Add LangChain vs Base Nodes mapping table
3. ✅ Add AI Agent few-shot example (#4)

### Phase 2: Reinforcement (Implement Second)
4. ✅ Update Common Mistakes section (make LangChain #1)
5. ✅ Update Pre-Generation Checklist
6. ✅ Add RAG workflow pattern examples

### Phase 3: Testing
7. Create `test_method_c_v3.py`
8. Run on same 100 templates
9. Verify Node F1 >= 0.5

## Expected Impact

### Baseline (Method C v2):
- Node F1: 0.415
- 15 workflows with F1 < 0.2 (all LangChain-related)

### Expected (Method C v3):
- **Node F1: 0.50 - 0.53** (+0.085 to +0.115)
- Workflows with F1 < 0.2: **5-8** (reduce by ~50%)
- 4 workflows with F1=0 should improve to 0.4-0.6

### Calculation:
- Fix 15 workflows from avg 0.15 → 0.50 = +0.35 each
- Impact on overall: (15 × 0.35) / 100 = +0.0525
- Plus spillover effect on other AI workflows: +0.03
- **Total expected improvement: +0.08 to +0.12**

## Files to Modify

1. **Main**: `evaluation/config/workflow_generation_prompt_improved.txt`
   - Add sections as described above
   - Keep existing examples and guidelines

2. **Test**: Create `scripts/test_method_c_v3.py`
   - Copy from `test_method_c_v2.py`
   - Update to use new prompt
   - Compare v2 vs v3

3. **Docs**: Update `evaluation/config/PROMPT_VERSION_CHANGES.md`
   - Document v3 changes
   - Include analysis that led to changes

## Success Criteria

- ✅ Node F1 >= 0.50 (target reached!)
- ✅ Workflows with F1 < 0.2 reduced by 50%
- ✅ Zero workflows with F1 = 0.000
- ✅ Connection F1 > 0.20 (should improve with correct nodes)
- ⚠️  Parameter Accuracy: Focus on nodes first, tackle parameters in v4

## Risk Mitigation

**Risk**: Adding too much complexity to prompt
**Mitigation**: Keep examples concrete, use visual patterns

**Risk**: Breaking existing good performance
**Mitigation**: Keep all existing examples and guidelines, only ADD new content

**Risk**: Cost increase
**Mitigation**: Acceptable for accuracy gain. Monitor per-workflow cost.

## Next Version Ideas (v4)

Once v3 reaches 0.5 Node F1:
1. Focus on Parameter Accuracy (currently 0.138, target 0.20)
2. Add parameter validation examples
3. Include common parameter configurations
4. Two-stage generation (structure → parameters)
