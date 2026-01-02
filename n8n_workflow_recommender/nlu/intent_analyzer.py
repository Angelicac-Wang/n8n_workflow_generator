#!/usr/bin/env python3
"""
NLU 意圖分析器

使用 GPT 進行自然語言理解，提取目標描述、參數和功能類別。
"""

import json
from typing import Dict, List, Optional, Set
from openai import OpenAI


class IntentAnalyzer:
    """
    意圖分析器
    
    使用 GPT 分析用戶查詢，提取結構化信息。
    """
    
    def __init__(self, openai_api_key: str, ontology: Optional[Dict] = None, function_categories: Optional[Dict] = None):
        """
        初始化意圖分析器
        
        Args:
            openai_api_key: OpenAI API 密鑰
            ontology: Ontology 字典（用於提供參數上下文）
            function_categories: 功能類別字典（用於提供類別上下文）
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.ontology = ontology or {}
        self.function_categories = function_categories or {}
    
    def _build_nlu_prompt(self, user_query: str) -> str:
        """構建 NLU 分析 prompt（動態包含 ontology 和 function_categories）"""
        # 構建 ontology 描述
        ontology_description_parts = []
        for node_name, details in list(self.ontology.items())[:20]:  # 只取前20個避免過長
            params_list = ", ".join(details.get("required_params", [])[:5])  # 每個節點只取前5個參數
            ontology_description_parts.append(
                f"- Node '{node_name}' needs parameters: [{params_list}]"
            )
        ontology_description = "\n".join(ontology_description_parts)
        
        # 構建功能類別描述
        function_categories_str = json.dumps(self.function_categories, ensure_ascii=False, indent=2) if self.function_categories else "{}"
        
        # 構建可選的類別列表（只取類別名稱）
        available_categories_list = list(self.function_categories.keys()) if self.function_categories else []
        categories_list_str = json.dumps(available_categories_list, ensure_ascii=False, indent=2) if available_categories_list else "[]"
        
        # 調試：輸出傳入的 categories 數量
        print(f"   - Passing {len(available_categories_list)} function categories to GPT")
        if available_categories_list:
            print(f"   - Sample categories: {available_categories_list[:5]}")
        
        return f"""You are an expert NLU engine for a workflow automation system. Analyze the user's query and provide:

1. A clear "goal_description" summarizing the overall intent (in English, concise)
2. A "parameters" object with extracted entities mapped to our ontology
3. A "function_categories" list - **MUST select from the available categories list below**

--- AVAILABLE FUNCTION CATEGORIES (YOU MUST SELECT FROM THIS LIST) ---
{categories_list_str}

--- FUNCTION CATEGORIES WITH DESCRIPTIONS ---
{function_categories_str}

--- AVAILABLE ONTOLOGY SCHEMA (sample) ---
{ontology_description}

**IMPORTANT**: For "function_categories", you MUST select one or more categories from the available list above. 
Do NOT create new category names. Only use the exact names from the list.

**Example Output:**
{{
  "goal_description": "Design an intelligent email processing workflow",
  "parameters": {{
    "email_provider": "gmail",
    "ai_model": "openai",
    "calendar_service": "google_calendar"
  }},
  "function_categories": ["Customer Engagement & Marketing", "Productivity & Collaboration", "AI, ML & Automation Intelligence"]
}}

**Note**: The function_categories should be top-level categories from the taxonomy (e.g., "Commerce & Revenue Operations", "Customer Engagement & Marketing", "AI, ML & Automation Intelligence").

Now analyze: "{user_query}"

Return ONLY valid JSON, no other text."""
    
    def analyze(self, user_query: str) -> Dict:
        """
        分析用戶查詢
        
        Args:
            user_query: 用戶查詢字符串
        
        Returns:
            analysis: {
                "goal_description": str,
                "parameters": dict,
                "function_categories": list
            }
        """
        print("\nSTAGE 0: NLU Analysis (GPT)")
        print(f" - User Query: {user_query}")
        
        try:
            prompt = self._build_nlu_prompt(user_query)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # 使用更強的模型，像原本的程式碼
                messages=[
                    {"role": "system", "content": "You are an expert NLU engine for workflow automation. Extract structured information accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 嘗試清理 JSON（移除可能的 markdown 代碼塊）
            if result_text.startswith('```'):
                # 移除 markdown 代碼塊標記
                lines = result_text.split('\n')
                result_text = '\n'.join([line for line in lines if not line.strip().startswith('```')])
            
            # 嘗試解析 JSON
            try:
                analysis = json.loads(result_text)
            except json.JSONDecodeError:
                # 如果失敗，嘗試提取 JSON 部分
                import re
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    raise
            
            print(f" - Goal: {analysis.get('goal_description', 'N/A')}")
            print(f" - Parameters: {list(analysis.get('parameters', {}).keys())}")
            print(f" - Function Categories: {analysis.get('function_categories', [])}")
            
            return analysis
        
        except Exception as e:
            print(f" - Error in NLU analysis: {e}")
            # 返回默認值
            return {
                "goal_description": user_query,
                "parameters": {},
                "function_categories": []
            }
    
    def extract_keywords(self, user_query: str, analysis: Optional[Dict] = None) -> Set[str]:
        """
        使用 LLM 從用戶查詢和 NLU 分析中提取高質量的關鍵字（像原本的程式碼）
        
        Args:
            user_query: 用戶查詢字符串
            analysis: NLU 分析結果
        
        Returns:
            keywords: 關鍵字集合
        """
        print(" - Extracting keywords with LLM...")
        
        prompt = f"""
You are an expert NLU (Natural Language Understanding) assistant. 
Your task is to extract critical technical keywords, entities, and actions from the user's query. These keywords will be used to search a technical function database (taxonomy).

**Input Context:**
1.  **User Query:** "{user_query}"
2.  **NLU Goal:** "{analysis.get('goal_description', 'N/A') if analysis else 'N/A'}"
3.  **NLU Parameters:** {json.dumps(analysis.get('parameters', {}) if analysis else {}, ensure_ascii=False)}

**Instructions:**
1.  Focus on *specific* technical terms, nouns, and actions (e.g., "OCR", "變數", "API呼叫", "JSON", "flow.begin").
2.  Include key entities from the parameters if they are technical (e.g., "OCR_result", "post_node_execution").
3.  Avoid generic, conversational words (e.g., "我要設計", "一個流程", "最後有", "結果").
4.  Keep keywords concise and relevant.
5.  Include both English and Chinese keywords if applicable.

**Output Format:**
Return a JSON object with a single key "keywords", which contains a list of unique keyword strings.
Example: {{"keywords": ["OCR", "變數", "節點執行後", "JSON", "OCR結果"]}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 關鍵字提取用 mini 就夠了
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            keywords = result.get("keywords", [])
            
            # 轉為 set 以確保唯一性，並過濾掉空字串
            keywords_set = {kw for kw in keywords if kw}
            
            # 也從分析結果中添加功能類別作為關鍵字
            if analysis:
                categories = analysis.get('function_categories', [])
                keywords_set.update([cat for cat in categories if cat])
            
            print(f"   - Extracted {len(keywords_set)} keywords: {keywords_set}")
            return keywords_set
            
        except Exception as e:
            print(f" - LLM Keyword extraction failed: {e}. Falling back to simple extraction.")
            # 備用方案：簡單提取
            keywords = set()
            words = user_query.lower().split()
            stop_words = {'的', '是', '在', '有', '和', '與', '或', '要', '我', '你', '他', '她', '它', 
                         'the', 'is', 'are', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'at'}
            for word in words:
                word = word.strip('.,!?;:()[]{}"\'-')
                if len(word) > 2 and word not in stop_words:
                    keywords.add(word)
            return keywords

