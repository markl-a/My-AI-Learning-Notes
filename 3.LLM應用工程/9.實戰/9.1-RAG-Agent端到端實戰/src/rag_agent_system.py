"""RAG + Agent 核心系統"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import json

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from src.vector_store import VectorStoreManager, HybridSearcher
from src.agent_tools import (
    ToolRegistry, RAGSearchTool, CalculatorTool,
    WebSearchTool, CodeInterpreterTool
)
from src.models import QueryRequest, QueryResponse, Source

logger = logging.getLogger(__name__)


class RAGAgentSystem:
    """RAG + Agent 集成系統"""

    def __init__(
        self,
        llm,
        vector_store: VectorStoreManager,
        config: Dict[str, Any]
    ):
        """初始化 RAG Agent 系統

        Args:
            llm: 語言模型
            vector_store: 向量存儲
            config: 配置字典
        """
        self.llm = llm
        self.vector_store = vector_store
        self.config = config

        # 初始化混合搜索器
        self.hybrid_searcher = HybridSearcher(vector_store)

        # 初始化工具註冊表
        self.tool_registry = ToolRegistry()
        self._register_tools()

        # 對話歷史（按 session_id 存儲）
        self.conversation_history: Dict[str, List] = {}

        # 統計信息
        self.stats = {
            "total_queries": 0,
            "tool_usage": {},
            "avg_response_time": 0.0
        }

    def _register_tools(self):
        """註冊 Agent 工具"""
        agent_config = self.config.get('agent', {})
        enabled_tools = agent_config.get('enable_tools', [])

        # RAG 搜索工具
        if 'rag_search' in enabled_tools:
            self.tool_registry.register(RAGSearchTool(self.vector_store))

        # 計算器工具
        if 'calculator' in enabled_tools:
            self.tool_registry.register(CalculatorTool())

        # 網路搜索工具
        if 'web_search' in enabled_tools:
            api_key = self.config.get('web_search', {}).get('api_key')
            self.tool_registry.register(WebSearchTool(api_key))

        # 代碼解釋器工具
        if 'code_interpreter' in enabled_tools:
            self.tool_registry.register(CodeInterpreterTool())

        logger.info(f"Registered {len(self.tool_registry.tools)} tools")

    def query(self, request: QueryRequest) -> QueryResponse:
        """處理查詢請求

        Args:
            request: 查詢請求

        Returns:
            查詢響應
        """
        start_time = time.time()

        try:
            # 更新統計
            self.stats["total_queries"] += 1

            # 獲取會話歷史
            session_history = self.conversation_history.get(
                request.session_id or "default", []
            )

            if request.use_agent:
                # 使用 Agent 模式
                response = self._agent_query(request, session_history)
            else:
                # 簡單 RAG 模式
                response = self._simple_rag_query(request)

            # 添加 AI 輔助功能
            response = self._enhance_response(response, request)

            # 更新會話歷史
            session_history.append({
                "question": request.question,
                "answer": response.answer
            })
            self.conversation_history[request.session_id or "default"] = session_history[-10:]  # 保留最近10輪

            # 計算處理時間
            response.processing_time = time.time() - start_time

            # 更新平均響應時間
            self.stats["avg_response_time"] = (
                (self.stats["avg_response_time"] * (self.stats["total_queries"] - 1) +
                 response.processing_time) / self.stats["total_queries"]
            )

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return QueryResponse(
                answer=f"抱歉，處理查詢時出現錯誤: {str(e)}",
                processing_time=time.time() - start_time
            )

    def _simple_rag_query(self, request: QueryRequest) -> QueryResponse:
        """簡單 RAG 查詢（不使用 Agent）

        Args:
            request: 查詢請求

        Returns:
            查詢響應
        """
        # 檢索相關文檔
        rag_config = self.config.get('rag', {})
        use_hybrid = rag_config.get('use_hybrid_search', False)

        if use_hybrid:
            results = self.hybrid_searcher.hybrid_search(
                request.question,
                top_k=request.top_k
            )
        else:
            results = self.vector_store.similarity_search(
                request.question,
                top_k=request.top_k
            )

        # 構建上下文
        context_parts = []
        sources = []

        for i, (doc, metadata, score) in enumerate(results, 1):
            context_parts.append(f"[文檔 {i}]\n{doc}")
            sources.append(Source(
                content=doc[:500],  # 截斷以節省空間
                document=metadata.get('source', 'Unknown'),
                page=metadata.get('page'),
                score=score,
                metadata=metadata
            ))

        context = "\n\n".join(context_parts)

        # 生成回答
        prompt = f"""基於以下文檔內容回答問題。如果文檔中沒有相關信息，請明確說明。

文檔內容:
{context}

問題: {request.question}

請提供準確、詳細的回答，並在適當的地方引用文檔編號。"""

        response_text = self._call_llm(prompt)

        return QueryResponse(
            answer=response_text,
            sources=sources,
            tools_used=["rag_search"],
            confidence=self._estimate_confidence(results)
        )

    def _agent_query(
        self,
        request: QueryRequest,
        session_history: List[Dict]
    ) -> QueryResponse:
        """Agent 查詢（帶工具使用）

        Args:
            request: 查詢請求
            session_history: 會話歷史

        Returns:
            查詢響應
        """
        # 分析問題類型並決定使用哪些工具
        tools_used = []
        all_sources = []
        intermediate_steps = []

        # 系統提示
        tools_description = self._format_tools_description()
        system_prompt = f"""你是一個智能助手，可以使用多種工具來回答問題。

可用工具:
{tools_description}

請根據問題選擇合適的工具。你可以多次調用工具，並綜合所有信息給出最終答案。

工具調用格式:
使用 JSON 格式: {{"tool": "工具名稱", "parameters": {{參數}}}}

回答格式:
1. 先分析問題
2. 決定使用哪些工具
3. 調用工具並獲取結果
4. 綜合結果給出最終答案"""

        # 構建對話歷史
        messages = [SystemMessage(content=system_prompt)]

        # 添加歷史對話
        for turn in session_history[-3:]:  # 只保留最近3輪
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(AIMessage(content=turn["answer"]))

        # 添加當前問題
        messages.append(HumanMessage(content=request.question))

        # Agent 推理循環
        max_iterations = self.config.get('agent', {}).get('max_iterations', 5)

        for iteration in range(max_iterations):
            # 調用 LLM
            response = self._call_llm_with_messages(messages)

            # 檢查是否需要調用工具
            tool_call = self._parse_tool_call(response)

            if not tool_call:
                # 沒有工具調用，返回最終答案
                return QueryResponse(
                    answer=response,
                    sources=all_sources,
                    tools_used=tools_used,
                    confidence=self._estimate_confidence_from_sources(all_sources)
                )

            # 執行工具
            tool_name = tool_call["tool"]
            tool_params = tool_call["parameters"]

            logger.info(f"Executing tool: {tool_name} with params: {tool_params}")

            tool_result = self.tool_registry.execute_tool(tool_name, **tool_params)
            tools_used.append(tool_name)

            # 更新工具使用統計
            self.stats["tool_usage"][tool_name] = self.stats["tool_usage"].get(tool_name, 0) + 1

            # 如果是 RAG 搜索，提取來源
            if tool_name == "rag_search":
                sources = self._extract_sources_from_rag_result(tool_result)
                all_sources.extend(sources)

            # 添加工具結果到對話
            intermediate_steps.append({
                "tool": tool_name,
                "result": tool_result
            })

            messages.append(AIMessage(content=f"工具調用: {json.dumps(tool_call, ensure_ascii=False)}"))
            messages.append(HumanMessage(content=f"工具結果:\n{tool_result}\n\n請繼續處理或給出最終答案。"))

        # 達到最大迭代次數，生成最終答案
        final_prompt = "請基於以上所有工具調用的結果，給出最終的完整答案。"
        messages.append(HumanMessage(content=final_prompt))

        final_answer = self._call_llm_with_messages(messages)

        return QueryResponse(
            answer=final_answer,
            sources=all_sources,
            tools_used=tools_used,
            confidence=self._estimate_confidence_from_sources(all_sources)
        )

    def _call_llm(self, prompt: str) -> str:
        """調用 LLM

        Args:
            prompt: 提示詞

        Returns:
            LLM 響應
        """
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "抱歉，生成回答時出現錯誤。"

    def _call_llm_with_messages(self, messages: List) -> str:
        """使用消息列表調用 LLM

        Args:
            messages: 消息列表

        Returns:
            LLM 響應
        """
        try:
            response = self.llm.invoke(messages)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "抱歉，生成回答時出現錯誤。"

    def _format_tools_description(self) -> str:
        """格式化工具描述"""
        tools = self.tool_registry.get_all_tools()
        descriptions = []

        for tool in tools:
            schema = tool.get_schema()
            descriptions.append(f"- {schema['name']}: {schema['description']}")

        return "\n".join(descriptions)

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 響應中的工具調用

        Args:
            response: LLM 響應

        Returns:
            工具調用字典或 None
        """
        try:
            # 嘗試找到 JSON 格式的工具調用
            import re
            json_match = re.search(r'\{[^}]*"tool"[^}]*\}', response)

            if json_match:
                tool_call = json.loads(json_match.group())
                if "tool" in tool_call:
                    return tool_call

            return None

        except Exception as e:
            logger.debug(f"Failed to parse tool call: {e}")
            return None

    def _extract_sources_from_rag_result(self, result: str) -> List[Source]:
        """從 RAG 結果中提取來源

        Args:
            result: RAG 工具結果

        Returns:
            來源列表
        """
        sources = []

        # 解析結果格式
        import re
        pattern = r'\[文檔 \d+\] \(來源: ([^,]+), 相關度: ([\d.]+)\)\n(.+?)(?=\n\[文檔|\Z)'

        matches = re.finditer(pattern, result, re.DOTALL)

        for match in matches:
            source_name = match.group(1)
            score = float(match.group(2))
            content = match.group(3).strip()

            sources.append(Source(
                content=content[:500],
                document=source_name,
                score=score
            ))

        return sources

    def _enhance_response(
        self,
        response: QueryResponse,
        request: QueryRequest
    ) -> QueryResponse:
        """AI 輔助增強響應

        Args:
            response: 原始響應
            request: 查詢請求

        Returns:
            增強後的響應
        """
        # 生成追問建議
        if response.sources:
            suggestions = self._generate_followup_questions(
                request.question,
                response.answer
            )
            response.suggestions = suggestions

        return response

    def _generate_followup_questions(
        self,
        question: str,
        answer: str
    ) -> List[str]:
        """生成追問建議

        Args:
            question: 原始問題
            answer: 回答

        Returns:
            追問問題列表
        """
        try:
            prompt = f"""基於以下問答，生成3個相關的追問問題。

問題: {question}
回答: {answer[:500]}

請生成3個自然的追問問題，每行一個，不要編號："""

            response = self._call_llm(prompt)

            # 解析追問問題
            questions = [
                q.strip().lstrip('123456789.-）)').strip()
                for q in response.split('\n')
                if q.strip()
            ]

            return questions[:3]

        except Exception as e:
            logger.error(f"Failed to generate follow-up questions: {e}")
            return []

    def _estimate_confidence(self, results: List[Tuple]) -> float:
        """估計回答的置信度

        Args:
            results: 檢索結果

        Returns:
            置信度分數 (0-1)
        """
        if not results:
            return 0.0

        # 基於最高相似度分數
        top_score = results[0][2] if len(results) > 0 else 0.0

        # 基於結果數量
        count_factor = min(len(results) / 5, 1.0)

        # 綜合置信度
        confidence = (top_score * 0.7 + count_factor * 0.3)

        return round(confidence, 2)

    def _estimate_confidence_from_sources(self, sources: List[Source]) -> float:
        """從來源估計置信度

        Args:
            sources: 來源列表

        Returns:
            置信度分數
        """
        if not sources:
            return 0.5  # 默認中等置信度

        avg_score = sum(s.score for s in sources) / len(sources)
        return round(avg_score, 2)

    def get_stats(self) -> Dict[str, Any]:
        """獲取系統統計"""
        return {
            **self.stats,
            "vector_store": self.vector_store.get_stats()
        }
