"""
文檔分析器
使用 LLM 進行文檔分析、摘要、實體提取、問答等
"""

import logging
from typing import Dict, List, Optional, Any
import os
from openai import AsyncOpenAI
import json
import re
from collections import Counter

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """文檔分析器類"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        """
        初始化文檔分析器

        Args:
            model_name: OpenAI 模型名稱
            api_key: OpenAI API 密鑰
        """
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

        # 分析結果緩存
        self.analysis_cache = {}

    def get_supported_analyses(self) -> List[str]:
        """獲取支持的分析類型"""
        return [
            "summary",
            "entities",
            "keywords",
            "topics",
            "sentiment",
            "structure"
        ]

    async def analyze(
        self,
        document_id: str,
        analysis_type: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        分析文檔

        Args:
            document_id: 文檔 ID
            analysis_type: 分析類型
            options: 分析選項

        Returns:
            分析結果字典
        """
        # 檢查緩存
        cache_key = f"{document_id}_{analysis_type}"
        if cache_key in self.analysis_cache:
            logger.info(f"Using cached analysis for {cache_key}")
            return self.analysis_cache[cache_key]

        # 獲取文檔文本（這裡需要從 document_processor 獲取）
        # 簡化處理，實際應該注入 document_processor
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        text = processor.get_document_text(document_id)

        if not text:
            raise ValueError(f"Document {document_id} not found")

        # 根據類型執行不同分析
        if analysis_type == "summary":
            result = await self._generate_summary(text, options)
        elif analysis_type == "entities":
            result = await self._extract_entities(text, options)
        elif analysis_type == "keywords":
            result = await self._extract_keywords(text, options)
        elif analysis_type == "topics":
            result = await self._extract_topics(text, options)
        elif analysis_type == "sentiment":
            result = await self._analyze_sentiment(text, options)
        elif analysis_type == "structure":
            result = await self._analyze_structure(text, options)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")

        # 緩存結果
        self.analysis_cache[cache_key] = result

        return result

    async def _generate_summary(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        生成文檔摘要

        Args:
            text: 文檔文本
            options: 選項（如摘要長度）

        Returns:
            摘要結果
        """
        options = options or {}
        summary_length = options.get('length', 'medium')  # short, medium, long

        # 根據長度設置 prompt
        length_prompts = {
            'short': "一段簡短的摘要（2-3句話）",
            'medium': "一個中等長度的摘要（100-150字）",
            'long': "一個詳細的摘要（200-300字）"
        }

        prompt = f"""
請為以下文檔生成{length_prompts.get(summary_length, length_prompts['medium'])}。
摘要應該：
1. 抓住文檔的核心主題
2. 包含關鍵信息和要點
3. 使用清晰、簡潔的語言

文檔內容：
{text[:4000]}

請直接生成摘要，不要添加額外說明。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )

        summary = response.choices[0].message.content.strip()

        return {
            "summary": summary,
            "length": summary_length,
            "word_count": len(summary.split()),
            "original_length": len(text.split())
        }

    async def _extract_entities(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        提取命名實體

        Args:
            text: 文檔文本
            options: 選項

        Returns:
            實體提取結果
        """
        prompt = f"""
請從以下文檔中提取所有重要的命名實體，包括：
- 人名（PERSON）
- 地名（LOCATION）
- 組織名（ORGANIZATION）
- 日期時間（DATE）
- 其他重要實體

以 JSON 格式返回結果，格式如下：
{{
    "entities": [
        {{"text": "實體文本", "type": "類型", "context": "上下文"}},
        ...
    ]
}}

文檔內容：
{text[:4000]}

請只返回 JSON，不要添加其他內容。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )

        # 解析 JSON 結果
        result_text = response.choices[0].message.content.strip()

        try:
            # 提取 JSON 部分
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                entities_data = json.loads(json_match.group())
            else:
                entities_data = {"entities": []}
        except json.JSONDecodeError:
            logger.error("Failed to parse entities JSON")
            entities_data = {"entities": []}

        # 統計實體類型
        entity_types = Counter([e['type'] for e in entities_data.get('entities', [])])

        return {
            "entities": entities_data.get('entities', []),
            "total_entities": len(entities_data.get('entities', [])),
            "entity_types": dict(entity_types)
        }

    async def _extract_keywords(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        提取關鍵詞

        Args:
            text: 文檔文本
            options: 選項

        Returns:
            關鍵詞提取結果
        """
        options = options or {}
        max_keywords = options.get('max_keywords', 10)

        prompt = f"""
請從以下文檔中提取最重要的 {max_keywords} 個關鍵詞或關鍵短語。

要求：
1. 關鍵詞應該反映文檔的核心主題
2. 優先選擇專業術語和重要概念
3. 每個關鍵詞附帶重要性評分（1-10）

以 JSON 格式返回：
{{
    "keywords": [
        {{"keyword": "關鍵詞", "score": 9, "category": "類別"}},
        ...
    ]
}}

文檔內容：
{text[:4000]}

請只返回 JSON。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        result_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                keywords_data = json.loads(json_match.group())
            else:
                keywords_data = {"keywords": []}
        except json.JSONDecodeError:
            logger.error("Failed to parse keywords JSON")
            keywords_data = {"keywords": []}

        return {
            "keywords": keywords_data.get('keywords', []),
            "total_keywords": len(keywords_data.get('keywords', [])),
        }

    async def _extract_topics(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        主題提取

        Args:
            text: 文檔文本
            options: 選項

        Returns:
            主題提取結果
        """
        prompt = f"""
請分析以下文檔，識別主要主題和子主題。

以 JSON 格式返回：
{{
    "topics": [
        {{
            "name": "主題名稱",
            "description": "主題描述",
            "relevance": 0.9,
            "subtopics": ["子主題1", "子主題2"]
        }},
        ...
    ]
}}

文檔內容：
{text[:4000]}

請只返回 JSON。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                topics_data = json.loads(json_match.group())
            else:
                topics_data = {"topics": []}
        except json.JSONDecodeError:
            logger.error("Failed to parse topics JSON")
            topics_data = {"topics": []}

        return topics_data

    async def _analyze_sentiment(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        情感分析

        Args:
            text: 文檔文本
            options: 選項

        Returns:
            情感分析結果
        """
        prompt = f"""
請分析以下文檔的情感傾向。

以 JSON 格式返回：
{{
    "overall_sentiment": "positive/neutral/negative",
    "confidence": 0.85,
    "sentiment_score": 0.7,
    "key_sentiments": [
        {{"aspect": "方面", "sentiment": "positive", "confidence": 0.9}},
        ...
    ],
    "explanation": "簡要解釋"
}}

文檔內容：
{text[:4000]}

請只返回 JSON。
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        result_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                sentiment_data = json.loads(json_match.group())
            else:
                sentiment_data = {
                    "overall_sentiment": "neutral",
                    "confidence": 0.0
                }
        except json.JSONDecodeError:
            logger.error("Failed to parse sentiment JSON")
            sentiment_data = {"overall_sentiment": "neutral", "confidence": 0.0}

        return sentiment_data

    async def _analyze_structure(
        self,
        text: str,
        options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        文檔結構分析

        Args:
            text: 文檔文本
            options: 選項

        Returns:
            結構分析結果
        """
        # 簡單的結構分析
        lines = text.split('\n')
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        # 識別標題（簡單規則）
        headings = []
        for line in lines:
            if line.strip() and (
                line.strip().isupper() or
                re.match(r'^#{1,6}\s+', line) or
                re.match(r'^\d+\.', line)
            ):
                headings.append(line.strip())

        return {
            "total_lines": len(lines),
            "total_paragraphs": len(paragraphs),
            "headings": headings[:20],  # 前20個標題
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            "structure_type": "structured" if len(headings) > 5 else "unstructured"
        }

    async def answer_question(
        self,
        document_id: str,
        question: str,
        context_window: int = 1000
    ) -> Dict[str, Any]:
        """
        基於文檔回答問題

        Args:
            document_id: 文檔 ID
            question: 問題
            context_window: 上下文窗口大小

        Returns:
            問答結果
        """
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        text = processor.get_document_text(document_id)

        if not text:
            raise ValueError(f"Document {document_id} not found")

        # 截斷文本以適應上下文窗口
        text_excerpt = text[:context_window * 4]  # 約4字符 = 1 token

        prompt = f"""
基於以下文檔內容，請回答問題。

要求：
1. 答案應該基於文檔內容
2. 如果文檔中沒有相關信息，請明確說明
3. 提供置信度評分（0-1）
4. 引用文檔中的相關部分

文檔內容：
{text_excerpt}

問題：{question}

請以 JSON 格式回答：
{{
    "answer": "答案內容",
    "confidence": 0.9,
    "sources": ["引用1", "引用2"],
    "explanation": "解釋"
}}
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        result_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                answer_data = json.loads(json_match.group())
            else:
                answer_data = {
                    "answer": result_text,
                    "confidence": 0.5,
                    "sources": []
                }
        except json.JSONDecodeError:
            logger.error("Failed to parse answer JSON")
            answer_data = {
                "answer": result_text,
                "confidence": 0.5,
                "sources": []
            }

        return answer_data

    async def compare_documents(
        self,
        document_ids: List[str],
        aspects: List[str]
    ) -> Dict[str, Any]:
        """
        比較多個文檔

        Args:
            document_ids: 文檔 ID 列表
            aspects: 比較方面

        Returns:
            比較結果
        """
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()

        # 獲取所有文檔的摘要
        summaries = {}
        for doc_id in document_ids:
            text = processor.get_document_text(doc_id)
            if text:
                summary_result = await self._generate_summary(text, {'length': 'short'})
                summaries[doc_id] = summary_result['summary']

        # 生成比較
        summaries_text = "\n\n".join([
            f"文檔 {i+1} ({doc_id}):\n{summary}"
            for i, (doc_id, summary) in enumerate(summaries.items())
        ])

        prompt = f"""
請比較以下文檔，重點關注：{', '.join(aspects)}

{summaries_text}

以 JSON 格式返回比較結果：
{{
    "similarities": ["相似點1", "相似點2"],
    "differences": ["差異點1", "差異點2"],
    "comparison_matrix": {{}},
    "conclusion": "總結"
}}
"""

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        try:
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                comparison_data = json.loads(json_match.group())
            else:
                comparison_data = {"similarities": [], "differences": []}
        except json.JSONDecodeError:
            comparison_data = {"similarities": [], "differences": []}

        return comparison_data

    async def batch_analyze(
        self,
        document_ids: List[str],
        analysis_types: List[str]
    ) -> Dict[str, Any]:
        """
        批量分析多個文檔

        Args:
            document_ids: 文檔 ID 列表
            analysis_types: 分析類型列表

        Returns:
            批量分析結果
        """
        results = {}

        for doc_id in document_ids:
            doc_results = {}
            for analysis_type in analysis_types:
                try:
                    result = await self.analyze(doc_id, analysis_type)
                    doc_results[analysis_type] = result
                except Exception as e:
                    logger.error(f"Error analyzing {doc_id} with {analysis_type}: {str(e)}")
                    doc_results[analysis_type] = {"error": str(e)}

            results[doc_id] = doc_results

        return results

    async def export_analysis(
        self,
        document_id: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        導出分析結果

        Args:
            document_id: 文檔 ID
            format: 導出格式

        Returns:
            導出數據
        """
        # 獲取所有分析結果
        all_analyses = {}
        for analysis_type in self.get_supported_analyses():
            cache_key = f"{document_id}_{analysis_type}"
            if cache_key in self.analysis_cache:
                all_analyses[analysis_type] = self.analysis_cache[cache_key]

        if format == "json":
            return all_analyses
        elif format == "markdown":
            # 轉換為 Markdown
            md_content = f"# 文檔分析報告\n\n文檔 ID: {document_id}\n\n"
            for analysis_type, result in all_analyses.items():
                md_content += f"## {analysis_type.upper()}\n\n"
                md_content += json.dumps(result, ensure_ascii=False, indent=2)
                md_content += "\n\n"
            return {"content": md_content, "format": "markdown"}
        else:
            return {"error": f"Unsupported format: {format}"}

    async def analyze_background(self, document_id: str):
        """後台分析任務"""
        logger.info(f"Starting background analysis for {document_id}")

        # 執行基本分析
        for analysis_type in ["summary", "keywords", "topics"]:
            try:
                await self.analyze(document_id, analysis_type)
            except Exception as e:
                logger.error(f"Background analysis error: {str(e)}")
