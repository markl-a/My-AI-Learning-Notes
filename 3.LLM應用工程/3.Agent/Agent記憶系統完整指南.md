# Agent 記憶系統 (Agent Memory Systems)

## 概述

記憶系統是構建可靠 AI Agent 的關鍵組件。有效的記憶架構讓 Agent 能夠在長對話中保持上下文、學習用戶偏好，並從過去的經驗中改進。

## 記憶類型架構

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent 記憶架構                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  短期記憶   │  │  長期記憶   │  │  情景記憶   │         │
│  │ Short-term │  │  Long-term  │  │  Episodic   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 對話上下文  │  │ 向量資料庫  │  │  經驗回放   │         │
│  │  Buffer     │  │ Vector DB  │  │  Replay     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  工作記憶   │  │  語義記憶   │  │  程序記憶   │         │
│  │  Working   │  │  Semantic   │  │ Procedural  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. 短期記憶 (Short-term Memory)

### 對話緩衝區

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json

@dataclass
class Message:
    """對話訊息"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

class ConversationBuffer:
    """對話緩衝區 - 短期記憶"""

    def __init__(
        self,
        max_messages: int = 50,
        max_tokens: int = 4000
    ):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: list[Message] = []

    def add(self, role: str, content: str, **metadata):
        """新增訊息"""
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)

        # 維持大小限制
        self._trim()

    def _trim(self):
        """修剪緩衝區"""
        # 訊息數量限制
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)

        # Token 限制（簡化估算）
        while self._estimate_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def _estimate_tokens(self) -> int:
        """估算 token 數量"""
        total = 0
        for msg in self.messages:
            # 粗略估算: 1 token ≈ 4 字元（英文）或 1.5 字元（中文）
            total += len(msg.content) // 2
        return total

    def get_messages(self) -> list[dict]:
        """取得格式化的訊息列表"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def get_context_window(self, n: int = 10) -> list[dict]:
        """取得最近 n 則訊息"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages[-n:]
        ]

    def clear(self):
        """清空緩衝區"""
        self.messages = []

    def search(self, keyword: str) -> list[Message]:
        """搜尋包含關鍵字的訊息"""
        return [
            msg for msg in self.messages
            if keyword.lower() in msg.content.lower()
        ]

# 使用範例
buffer = ConversationBuffer(max_messages=100, max_tokens=8000)

buffer.add("user", "你好，我想了解機器學習")
buffer.add("assistant", "好的，機器學習是人工智慧的一個分支...")
buffer.add("user", "可以舉個實際例子嗎？")

messages = buffer.get_messages()
```

### 滑動視窗記憶

```python
class SlidingWindowMemory:
    """滑動視窗記憶"""

    def __init__(
        self,
        window_size: int = 10,
        overlap: int = 2
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.all_messages: list[Message] = []
        self.summaries: list[str] = []

    def add(self, role: str, content: str):
        """新增訊息"""
        self.all_messages.append(Message(role=role, content=content))

        # 當超過視窗大小時，總結舊訊息
        if len(self.all_messages) > self.window_size:
            self._summarize_and_slide()

    def _summarize_and_slide(self):
        """總結並滑動視窗"""
        # 取出要總結的訊息（保留 overlap）
        to_summarize = self.all_messages[:self.window_size - self.overlap]
        self.all_messages = self.all_messages[self.window_size - self.overlap:]

        # 生成總結（這裡簡化處理，實際應使用 LLM）
        summary = self._generate_summary(to_summarize)
        self.summaries.append(summary)

    def _generate_summary(self, messages: list[Message]) -> str:
        """生成訊息總結（應使用 LLM）"""
        # 簡化版本，實際應該呼叫 LLM
        contents = [f"{m.role}: {m.content[:50]}..." for m in messages]
        return f"[總結] 討論了: {'; '.join(contents)}"

    def get_context(self) -> str:
        """取得完整上下文"""
        context_parts = []

        # 加入歷史總結
        if self.summaries:
            context_parts.append("=== 歷史總結 ===")
            for i, summary in enumerate(self.summaries):
                context_parts.append(f"{i+1}. {summary}")

        # 加入當前視窗
        context_parts.append("\n=== 當前對話 ===")
        for msg in self.all_messages:
            context_parts.append(f"{msg.role}: {msg.content}")

        return "\n".join(context_parts)
```

## 2. 長期記憶 (Long-term Memory)

### 向量儲存記憶

```python
from openai import OpenAI
import chromadb
from datetime import datetime
import hashlib
from typing import Optional
import json

class VectorMemory:
    """向量儲存長期記憶"""

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_dir: str = "./memory_store"
    ):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> list[float]:
        """取得文字嵌入"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _generate_id(self, content: str) -> str:
        """生成唯一 ID"""
        return hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()

    def store(
        self,
        content: str,
        memory_type: str = "conversation",
        importance: float = 0.5,
        metadata: Optional[dict] = None
    ) -> str:
        """儲存記憶"""
        memory_id = self._generate_id(content)
        embedding = self._get_embedding(content)

        doc_metadata = {
            "type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            **(metadata or {})
        }

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[doc_metadata]
        )

        return memory_id

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> list[dict]:
        """檢索相關記憶"""
        query_embedding = self._get_embedding(query)

        # 構建過濾條件
        where_filter = {}
        if memory_type:
            where_filter["type"] = memory_type
        if min_importance > 0:
            where_filter["importance"] = {"$gte": min_importance}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        memories = []
        for i in range(len(results['ids'][0])):
            memory_id = results['ids'][0][i]

            # 更新訪問計數
            self._update_access_count(memory_id)

            memories.append({
                "id": memory_id,
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "relevance": 1 - results['distances'][0][i]
            })

        return memories

    def _update_access_count(self, memory_id: str):
        """更新訪問計數"""
        existing = self.collection.get(ids=[memory_id])
        if existing['metadatas']:
            metadata = existing['metadatas'][0]
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            metadata['last_accessed'] = datetime.now().isoformat()

            self.collection.update(
                ids=[memory_id],
                metadatas=[metadata]
            )

    def forget(
        self,
        memory_id: Optional[str] = None,
        older_than_days: Optional[int] = None,
        min_access_count: Optional[int] = None
    ):
        """遺忘記憶"""
        if memory_id:
            self.collection.delete(ids=[memory_id])
            return

        # 根據條件刪除
        all_data = self.collection.get(include=["metadatas"])

        ids_to_delete = []
        for i, metadata in enumerate(all_data['metadatas']):
            should_delete = False

            if older_than_days:
                created = datetime.fromisoformat(metadata['timestamp'])
                age = (datetime.now() - created).days
                if age > older_than_days:
                    should_delete = True

            if min_access_count is not None:
                if metadata.get('access_count', 0) < min_access_count:
                    should_delete = True

            if should_delete:
                ids_to_delete.append(all_data['ids'][i])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

    def consolidate(self, memory_type: str = "conversation"):
        """整合記憶（合併相似記憶）"""
        # 取得所有該類型的記憶
        all_data = self.collection.get(
            where={"type": memory_type},
            include=["documents", "metadatas", "embeddings"]
        )

        # 簡化版：這裡應該用 clustering 來找相似記憶並合併
        # 實際實作需要更複雜的邏輯
        pass

# 使用範例
memory = VectorMemory()

# 儲存記憶
memory.store(
    "用戶喜歡簡短的回答",
    memory_type="preference",
    importance=0.8
)

memory.store(
    "討論了 Python 程式設計的基礎",
    memory_type="conversation",
    importance=0.5
)

# 檢索相關記憶
relevant = memory.retrieve(
    "用戶偏好什麼樣的回答風格？",
    n_results=3,
    memory_type="preference"
)
```

### SQL 結構化記憶

```python
import sqlite3
from datetime import datetime
from typing import Optional
import json

class SQLMemory:
    """SQL 結構化記憶"""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化資料庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 對話記憶表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # 用戶偏好表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, key)
            )
        """)

        # 事實記憶表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 任務歷史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT,
                success BOOLEAN,
                duration_ms INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def store_conversation(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ):
        """儲存對話"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO conversations (session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(metadata) if metadata else None)
        )

        conn.commit()
        conn.close()

    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> list[dict]:
        """取得對話歷史"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT role, content, timestamp, metadata
               FROM conversations
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, limit)
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "metadata": json.loads(row[3]) if row[3] else None
            }
            for row in reversed(rows)
        ]

    def set_preference(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 0.5
    ):
        """設定用戶偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO preferences (user_id, key, value, confidence, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_id, key) DO UPDATE SET
               value = excluded.value,
               confidence = excluded.confidence,
               updated_at = CURRENT_TIMESTAMP""",
            (user_id, key, value, confidence)
        )

        conn.commit()
        conn.close()

    def get_preference(
        self,
        user_id: str,
        key: str
    ) -> Optional[dict]:
        """取得用戶偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT value, confidence FROM preferences WHERE user_id = ? AND key = ?",
            (user_id, key)
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {"value": row[0], "confidence": row[1]}
        return None

    def get_all_preferences(self, user_id: str) -> dict:
        """取得所有用戶偏好"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT key, value, confidence FROM preferences WHERE user_id = ?",
            (user_id,)
        )

        rows = cursor.fetchall()
        conn.close()

        return {
            row[0]: {"value": row[1], "confidence": row[2]}
            for row in rows
        }

    def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        source: str = None,
        confidence: float = 1.0
    ):
        """儲存事實"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO facts (subject, predicate, object, source, confidence) VALUES (?, ?, ?, ?, ?)",
            (subject, predicate, obj, source, confidence)
        )

        conn.commit()
        conn.close()

    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None
    ) -> list[dict]:
        """查詢事實"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT subject, predicate, object, confidence FROM facts WHERE 1=1"
        params = []

        if subject:
            query += " AND subject LIKE ?"
            params.append(f"%{subject}%")

        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "subject": row[0],
                "predicate": row[1],
                "object": row[2],
                "confidence": row[3]
            }
            for row in rows
        ]

    def log_task(
        self,
        task_type: str,
        input_data: str,
        output_data: str = None,
        success: bool = True,
        duration_ms: int = 0
    ):
        """記錄任務執行"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """INSERT INTO task_history
               (task_type, input, output, success, duration_ms)
               VALUES (?, ?, ?, ?, ?)""",
            (task_type, input_data, output_data, success, duration_ms)
        )

        conn.commit()
        conn.close()

    def get_task_success_rate(self, task_type: str) -> float:
        """取得任務成功率"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
               FROM task_history
               WHERE task_type = ?""",
            (task_type,)
        )

        row = cursor.fetchone()
        conn.close()

        if row[0] > 0:
            return row[1] / row[0]
        return 0.0

# 使用範例
sql_memory = SQLMemory()

# 儲存對話
sql_memory.store_conversation(
    session_id="session_001",
    role="user",
    content="幫我寫一個 Python 函式"
)

# 設定偏好
sql_memory.set_preference(
    user_id="user_001",
    key="language",
    value="繁體中文",
    confidence=0.9
)

# 儲存事實
sql_memory.store_fact(
    subject="用戶",
    predicate="職業",
    obj="軟體工程師",
    confidence=0.8
)
```

## 3. 情景記憶 (Episodic Memory)

### 經驗回放系統

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
from openai import OpenAI

@dataclass
class Episode:
    """情景/經驗"""
    id: str
    context: str  # 情境描述
    action: str   # 採取的行動
    result: str   # 結果
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)

class EpisodicMemory:
    """情景記憶系統"""

    def __init__(self, max_episodes: int = 1000):
        self.episodes: list[Episode] = []
        self.max_episodes = max_episodes
        self.client = OpenAI()

    def _get_embedding(self, text: str) -> list[float]:
        """取得文字嵌入"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float]
    ) -> float:
        """計算餘弦相似度"""
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def record(
        self,
        context: str,
        action: str,
        result: str,
        success: bool,
        metadata: Optional[dict] = None
    ) -> Episode:
        """記錄經驗"""
        # 生成嵌入
        episode_text = f"情境: {context}\n行動: {action}\n結果: {result}"
        embedding = self._get_embedding(episode_text)

        episode = Episode(
            id=f"ep_{len(self.episodes)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            context=context,
            action=action,
            result=result,
            success=success,
            embedding=embedding,
            metadata=metadata or {}
        )

        self.episodes.append(episode)

        # 限制大小
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()

        return episode

    def _prune_episodes(self):
        """修剪經驗（保留重要的）"""
        # 優先保留成功的經驗
        successful = [e for e in self.episodes if e.success]
        failed = [e for e in self.episodes if not e.success]

        # 失敗的只保留最近的 20%
        keep_failed = failed[-(self.max_episodes // 5):]

        self.episodes = successful + keep_failed
        self.episodes.sort(key=lambda e: e.timestamp)

    def recall(
        self,
        current_context: str,
        n_results: int = 5,
        success_only: bool = False
    ) -> list[Episode]:
        """回憶相關經驗"""
        context_embedding = self._get_embedding(current_context)

        # 計算相似度
        similarities = []
        for episode in self.episodes:
            if success_only and not episode.success:
                continue

            if episode.embedding:
                sim = self._cosine_similarity(
                    context_embedding,
                    episode.embedding
                )
                similarities.append((episode, sim))

        # 排序並返回最相關的
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in similarities[:n_results]]

    def get_lessons_learned(
        self,
        context: str,
        n_results: int = 3
    ) -> str:
        """從經驗中學習"""
        # 取得相關的成功和失敗經驗
        successful = self.recall(context, n_results, success_only=True)
        failed = self.recall(context, n_results, success_only=False)
        failed = [e for e in failed if not e.success][:n_results]

        lessons = []

        if successful:
            lessons.append("=== 成功經驗 ===")
            for ep in successful:
                lessons.append(f"情境: {ep.context}")
                lessons.append(f"行動: {ep.action}")
                lessons.append(f"結果: {ep.result}")
                lessons.append("")

        if failed:
            lessons.append("=== 失敗經驗（避免） ===")
            for ep in failed:
                lessons.append(f"情境: {ep.context}")
                lessons.append(f"錯誤行動: {ep.action}")
                lessons.append(f"負面結果: {ep.result}")
                lessons.append("")

        return "\n".join(lessons)

    def analyze_patterns(self) -> dict:
        """分析經驗模式"""
        total = len(self.episodes)
        successful = sum(1 for e in self.episodes if e.success)

        # 按行動類型分組
        action_stats = {}
        for ep in self.episodes:
            action_type = ep.metadata.get("action_type", "unknown")
            if action_type not in action_stats:
                action_stats[action_type] = {"total": 0, "success": 0}
            action_stats[action_type]["total"] += 1
            if ep.success:
                action_stats[action_type]["success"] += 1

        return {
            "total_episodes": total,
            "success_rate": successful / total if total > 0 else 0,
            "action_stats": {
                k: {
                    **v,
                    "success_rate": v["success"] / v["total"] if v["total"] > 0 else 0
                }
                for k, v in action_stats.items()
            }
        }

# 使用範例
episodic = EpisodicMemory()

# 記錄經驗
episodic.record(
    context="用戶詢問如何排序列表",
    action="使用 sorted() 函數並解釋其參數",
    result="用戶成功理解並實作",
    success=True,
    metadata={"action_type": "code_explanation"}
)

episodic.record(
    context="用戶詢問複雜的演算法",
    action="直接給出完整程式碼",
    result="用戶表示無法理解",
    success=False,
    metadata={"action_type": "code_generation"}
)

# 回憶相關經驗
relevant = episodic.recall("用戶想學習列表操作", success_only=True)

# 獲取教訓
lessons = episodic.get_lessons_learned("用戶詢問程式問題")
print(lessons)
```

## 4. 整合記憶系統

### 統一記憶管理器

```python
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI

@dataclass
class MemoryContext:
    """記憶上下文"""
    short_term: list[dict]
    long_term: list[dict]
    episodic: list[dict]
    preferences: dict
    facts: list[dict]

class UnifiedMemoryManager:
    """統一記憶管理器"""

    def __init__(
        self,
        user_id: str,
        persist_dir: str = "./unified_memory"
    ):
        self.user_id = user_id
        self.client = OpenAI()

        # 初始化各種記憶系統
        self.short_term = ConversationBuffer(max_messages=50)
        self.long_term = VectorMemory(
            collection_name=f"memory_{user_id}",
            persist_dir=persist_dir
        )
        self.sql_memory = SQLMemory(f"{persist_dir}/memory.db")
        self.episodic = EpisodicMemory()

    def add_interaction(
        self,
        role: str,
        content: str,
        session_id: str = "default"
    ):
        """新增互動"""
        # 短期記憶
        self.short_term.add(role, content)

        # SQL 記錄
        self.sql_memory.store_conversation(
            session_id=session_id,
            role=role,
            content=content
        )

        # 判斷是否值得存入長期記憶
        if self._is_worth_remembering(content):
            self.long_term.store(
                content=content,
                memory_type="conversation",
                importance=self._calculate_importance(content)
            )

    def _is_worth_remembering(self, content: str) -> bool:
        """判斷是否值得記住"""
        # 簡單啟發式：超過一定長度或包含關鍵詞
        if len(content) > 100:
            return True

        important_keywords = [
            "記住", "重要", "偏好", "喜歡", "不要",
            "總是", "永遠", "remember", "important"
        ]
        return any(kw in content.lower() for kw in important_keywords)

    def _calculate_importance(self, content: str) -> float:
        """計算重要性分數"""
        score = 0.5  # 基礎分數

        # 長度加分
        if len(content) > 200:
            score += 0.1

        # 關鍵詞加分
        high_importance = ["非常重要", "必須", "關鍵", "critical"]
        if any(kw in content for kw in high_importance):
            score += 0.3

        return min(score, 1.0)

    def update_preference(
        self,
        key: str,
        value: str,
        confidence: float = 0.5
    ):
        """更新偏好"""
        self.sql_memory.set_preference(
            user_id=self.user_id,
            key=key,
            value=value,
            confidence=confidence
        )

    def record_experience(
        self,
        context: str,
        action: str,
        result: str,
        success: bool
    ):
        """記錄經驗"""
        self.episodic.record(context, action, result, success)

    def get_context(
        self,
        query: str,
        include_short_term: bool = True,
        include_long_term: bool = True,
        include_episodic: bool = True,
        include_preferences: bool = True
    ) -> MemoryContext:
        """取得完整記憶上下文"""
        context = MemoryContext(
            short_term=[],
            long_term=[],
            episodic=[],
            preferences={},
            facts=[]
        )

        if include_short_term:
            context.short_term = self.short_term.get_messages()

        if include_long_term:
            memories = self.long_term.retrieve(query, n_results=5)
            context.long_term = [
                {"content": m["content"], "relevance": m["relevance"]}
                for m in memories
            ]

        if include_episodic:
            episodes = self.episodic.recall(query, n_results=3)
            context.episodic = [
                {
                    "context": e.context,
                    "action": e.action,
                    "result": e.result,
                    "success": e.success
                }
                for e in episodes
            ]

        if include_preferences:
            context.preferences = self.sql_memory.get_all_preferences(
                self.user_id
            )

        return context

    def build_system_prompt(self, base_prompt: str, query: str) -> str:
        """建構包含記憶的系統提示"""
        context = self.get_context(query)

        memory_section = []

        # 用戶偏好
        if context.preferences:
            memory_section.append("## 用戶偏好")
            for key, value in context.preferences.items():
                memory_section.append(f"- {key}: {value['value']}")

        # 相關長期記憶
        if context.long_term:
            memory_section.append("\n## 相關記憶")
            for mem in context.long_term[:3]:
                memory_section.append(f"- {mem['content'][:100]}...")

        # 相關經驗
        if context.episodic:
            successful = [e for e in context.episodic if e['success']]
            if successful:
                memory_section.append("\n## 成功經驗參考")
                for exp in successful[:2]:
                    memory_section.append(
                        f"- 類似情境: {exp['context'][:50]}... "
                        f"→ 行動: {exp['action'][:50]}..."
                    )

        if memory_section:
            return f"{base_prompt}\n\n# 記憶上下文\n{''.join(memory_section)}"

        return base_prompt

    def generate_response(
        self,
        user_message: str,
        system_prompt: str = "你是一個有記憶的 AI 助手。"
    ) -> str:
        """生成回應（整合記憶）"""
        # 建構包含記憶的提示
        enhanced_prompt = self.build_system_prompt(system_prompt, user_message)

        # 取得對話歷史
        messages = [
            {"role": "system", "content": enhanced_prompt},
            *self.short_term.get_messages(),
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )

        assistant_message = response.choices[0].message.content

        # 記錄這次互動
        self.add_interaction("user", user_message)
        self.add_interaction("assistant", assistant_message)

        return assistant_message

# 使用範例
memory_manager = UnifiedMemoryManager(user_id="user_001")

# 設定偏好
memory_manager.update_preference("language", "繁體中文", confidence=0.9)
memory_manager.update_preference("expertise", "中級程式設計師", confidence=0.7)

# 對話
response = memory_manager.generate_response(
    "幫我解釋什麼是裝飾器？"
)
print(response)

# 記錄經驗
memory_manager.record_experience(
    context="用戶詢問 Python 裝飾器",
    action="提供概念解釋和簡單範例",
    result="用戶表示理解",
    success=True
)
```

## 5. 記憶優化策略

### 記憶壓縮與總結

```python
class MemoryCompressor:
    """記憶壓縮器"""

    def __init__(self):
        self.client = OpenAI()

    def summarize_conversation(
        self,
        messages: list[dict],
        max_length: int = 200
    ) -> str:
        """總結對話"""
        conversation = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in messages
        ])

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"請將以下對話總結為不超過 {max_length} 字的摘要，保留關鍵資訊。"
                },
                {
                    "role": "user",
                    "content": conversation
                }
            ],
            max_tokens=300
        )

        return response.choices[0].message.content

    def extract_key_facts(
        self,
        text: str
    ) -> list[dict]:
        """擷取關鍵事實"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """從文字中擷取關鍵事實，以 JSON 格式輸出：
[{"subject": "主詞", "predicate": "謂詞", "object": "受詞"}]"""
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=500
        )

        try:
            result = response.choices[0].message.content
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            return json.loads(result.strip())
        except:
            return []

    def merge_similar_memories(
        self,
        memories: list[str],
        similarity_threshold: float = 0.8
    ) -> list[str]:
        """合併相似記憶"""
        # 簡化版：使用 LLM 判斷和合併
        if len(memories) <= 1:
            return memories

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "合併以下相似的記憶條目，移除重複資訊，保留獨特內容。每個合併後的記憶用換行分隔。"
                },
                {
                    "role": "user",
                    "content": "\n---\n".join(memories)
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content.split("\n")
```

## 最佳實踐

### 1. 記憶分層策略

```
高頻訪問 → 短期記憶（記憶體）
    ↓
中頻訪問 → 向量記憶（快速檢索）
    ↓
低頻訪問 → SQL 記憶（結構化查詢）
    ↓
歸檔 → 壓縮儲存
```

### 2. 記憶生命週期

```python
# 記憶衰減策略
def calculate_memory_score(
    importance: float,
    recency_days: int,
    access_count: int
) -> float:
    """計算記憶保留分數"""
    # 時間衰減
    recency_score = 1.0 / (1 + recency_days * 0.1)

    # 訪問頻率加權
    access_score = min(access_count / 10, 1.0)

    # 綜合分數
    return importance * 0.4 + recency_score * 0.3 + access_score * 0.3
```

### 3. 隱私考量

```python
def sanitize_memory(content: str) -> str:
    """清理敏感資訊"""
    import re

    # 移除電子郵件
    content = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', '[EMAIL]', content)

    # 移除電話號碼
    content = re.sub(r'\b\d{10,}\b', '[PHONE]', content)

    # 移除信用卡號
    content = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', content)

    return content
```

## 延伸閱讀

- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [MemGPT](https://memgpt.ai/)
- [Cognitive Architectures for AI](https://arxiv.org/abs/2309.02427)
- [Long-term Memory in AI Systems](https://lilianweng.github.io/posts/2023-06-23-agent/)
