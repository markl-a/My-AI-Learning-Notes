# 即時 ML 系統 (Real-time ML Systems)

## 概述

即時 ML 系統能夠在毫秒級延遲內處理請求並返回結果，是現代 AI 應用的關鍵基礎設施。

## 系統架構

```
┌─────────────────────────────────────────────────────────────┐
│                   即時 ML 系統架構                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  客戶端                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                       │
│  │   Web   │ │  Mobile │ │   API   │                       │
│  └────┬────┘ └────┬────┘ └────┬────┘                       │
│       │           │           │                             │
│       └───────────┴───────────┘                             │
│                   │                                         │
│                   ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              API Gateway / Load Balancer             │   │
│  └─────────────────────────┬───────────────────────────┘   │
│                            │                               │
│          ┌─────────────────┼─────────────────┐             │
│          │                 │                 │             │
│          ▼                 ▼                 ▼             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │  Inference  │   │  Inference  │   │  Inference  │      │
│  │  Server 1   │   │  Server 2   │   │  Server N   │      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                │
│                           ▼                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Feature Store                      │   │
│  │         (Redis / DynamoDB / Feature Server)          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Model Registry                     │   │
│  │              (MLflow / Weights & Biases)             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 1. 低延遲推論服務

### FastAPI 高效能服務

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import time
from contextlib import asynccontextmanager
import numpy as np

# 模型載入
class ModelManager:
    """模型管理器"""

    def __init__(self):
        self.models = {}
        self.loading = False

    async def load_model(self, model_name: str):
        """異步載入模型"""
        # 模擬模型載入
        await asyncio.sleep(0.1)
        self.models[model_name] = {"loaded": True, "version": "1.0"}

    def predict(self, model_name: str, inputs: List[float]) -> List[float]:
        """同步預測"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        # 模擬預測
        return [x * 2 for x in inputs]

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期"""
    # 啟動時載入模型
    await model_manager.load_model("default")
    yield
    # 關閉時清理
    model_manager.models.clear()

app = FastAPI(
    title="Real-time ML Service",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 請求/回應模型
class PredictRequest(BaseModel):
    inputs: List[float]
    model_name: str = "default"

class PredictResponse(BaseModel):
    predictions: List[float]
    latency_ms: float
    model_version: str

# 健康檢查
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": list(model_manager.models.keys())}

# 預測端點
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.perf_counter()

    try:
        predictions = model_manager.predict(
            request.model_name,
            request.inputs
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - start_time) * 1000

    return PredictResponse(
        predictions=predictions,
        latency_ms=latency_ms,
        model_version=model_manager.models[request.model_name]["version"]
    )

# 批次預測
class BatchPredictRequest(BaseModel):
    batch: List[PredictRequest]

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictRequest):
    start_time = time.perf_counter()

    results = []
    for item in request.batch:
        predictions = model_manager.predict(item.model_name, item.inputs)
        results.append(predictions)

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "results": results,
        "total_latency_ms": latency_ms,
        "batch_size": len(request.batch)
    }
```

### 串流推論

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

@app.post("/stream")
async def stream_inference(request: dict):
    """串流推論端點"""

    async def generate():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=request.get("messages", []),
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                data = {
                    "content": chunk.choices[0].delta.content,
                    "finish_reason": chunk.choices[0].finish_reason
                }
                yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
```

## 2. 特徵服務 (Feature Store)

### Redis 特徵快取

```python
import redis
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

class FeatureStore:
    """Redis 特徵儲存"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600
    ):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = default_ttl

    def _feature_key(self, entity_type: str, entity_id: str) -> str:
        """生成特徵鍵"""
        return f"features:{entity_type}:{entity_id}"

    def set_features(
        self,
        entity_type: str,
        entity_id: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """設定特徵"""
        key = self._feature_key(entity_type, entity_id)

        data = {
            "features": features,
            "updated_at": datetime.now().isoformat()
        }

        self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(data)
        )

    def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """取得特徵"""
        key = self._feature_key(entity_type, entity_id)
        data = self.redis.get(key)

        if not data:
            return None

        parsed = json.loads(data)
        features = parsed["features"]

        if feature_names:
            return {k: features.get(k) for k in feature_names}

        return features

    def get_features_batch(
        self,
        entity_type: str,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """批次取得特徵"""
        keys = [self._feature_key(entity_type, eid) for eid in entity_ids]
        values = self.redis.mget(keys)

        results = {}
        for eid, value in zip(entity_ids, values):
            if value:
                parsed = json.loads(value)
                results[eid] = parsed["features"]

        return results

    def delete_features(self, entity_type: str, entity_id: str):
        """刪除特徵"""
        key = self._feature_key(entity_type, entity_id)
        self.redis.delete(key)

# 使用範例
feature_store = FeatureStore()

# 設定用戶特徵
feature_store.set_features(
    entity_type="user",
    entity_id="user_123",
    features={
        "age": 25,
        "purchase_count": 10,
        "avg_order_value": 150.0,
        "last_login_days": 2
    }
)

# 取得特徵
features = feature_store.get_features("user", "user_123")
```

### 即時特徵計算

```python
from typing import Dict, Any, Callable
from dataclasses import dataclass
import time

@dataclass
class FeatureDefinition:
    """特徵定義"""
    name: str
    compute_fn: Callable
    dependencies: list[str]
    cache_ttl: int = 300

class RealtimeFeatureEngine:
    """即時特徵引擎"""

    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.feature_defs: Dict[str, FeatureDefinition] = {}

    def register_feature(self, definition: FeatureDefinition):
        """註冊特徵"""
        self.feature_defs[definition.name] = definition

    def compute_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_names: list[str],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """計算特徵"""
        context = context or {}
        results = {}

        # 先嘗試從快取取得
        cached = self.feature_store.get_features(
            entity_type, entity_id, feature_names
        )

        for name in feature_names:
            if cached and name in cached:
                results[name] = cached[name]
                continue

            # 計算特徵
            if name in self.feature_defs:
                definition = self.feature_defs[name]

                # 計算依賴
                deps = {}
                for dep in definition.dependencies:
                    if dep in results:
                        deps[dep] = results[dep]
                    elif cached and dep in cached:
                        deps[dep] = cached[dep]

                # 執行計算
                value = definition.compute_fn(
                    entity_id=entity_id,
                    context=context,
                    dependencies=deps
                )
                results[name] = value

        # 更新快取
        if results:
            self.feature_store.set_features(
                entity_type, entity_id, results
            )

        return results

# 使用範例
engine = RealtimeFeatureEngine(feature_store)

# 註冊特徵
engine.register_feature(FeatureDefinition(
    name="session_duration",
    compute_fn=lambda **kwargs: kwargs["context"].get("current_time", 0) - kwargs["context"].get("session_start", 0),
    dependencies=[],
    cache_ttl=60
))

engine.register_feature(FeatureDefinition(
    name="engagement_score",
    compute_fn=lambda **kwargs: min(kwargs["dependencies"].get("session_duration", 0) / 600, 1.0),
    dependencies=["session_duration"],
    cache_ttl=60
))

# 計算特徵
features = engine.compute_features(
    entity_type="user",
    entity_id="user_123",
    feature_names=["session_duration", "engagement_score"],
    context={"current_time": time.time(), "session_start": time.time() - 300}
)
```

## 3. 即時向量搜尋

### 高效能向量檢索

```python
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import asyncio

@dataclass
class SearchResult:
    """搜尋結果"""
    id: str
    score: float
    metadata: Dict[str, Any]

class RealtimeVectorSearch:
    """即時向量搜尋"""

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "hnsw"
    ):
        self.dimension = dimension
        self.index_type = index_type

        # 使用 Qdrant
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(":memory:")  # 記憶體模式，生產環境使用持久化

        # 建立集合
        self.client.create_collection(
            collection_name="vectors",
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )

    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]] = None
    ):
        """新增向量"""
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload={"doc_id": doc_id, **(metadata[i] if metadata else {})}
            )
            for i, (doc_id, vec) in enumerate(zip(ids, vectors))
        ]

        self.client.upsert(
            collection_name="vectors",
            points=points
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜尋"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # 建構過濾器
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name="vectors",
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter
        )

        return [
            SearchResult(
                id=hit.payload.get("doc_id", str(hit.id)),
                score=hit.score,
                metadata=hit.payload
            )
            for hit in results
        ]

    async def search_async(
        self,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[SearchResult]:
        """異步搜尋"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query_vector, top_k)
        )

# 使用範例
search = RealtimeVectorSearch(dimension=1536)

# 新增向量
search.add_vectors(
    ids=["doc1", "doc2", "doc3"],
    vectors=[
        [0.1] * 1536,
        [0.2] * 1536,
        [0.3] * 1536
    ],
    metadata=[
        {"category": "tech"},
        {"category": "science"},
        {"category": "tech"}
    ]
)

# 搜尋
results = search.search(
    query_vector=[0.15] * 1536,
    top_k=2,
    filters={"category": "tech"}
)
```

## 4. WebSocket 即時互動

### WebSocket 聊天服務

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio
from openai import AsyncOpenAI

app = FastAPI()
client = AsyncOpenAI()

class ConnectionManager:
    """連接管理器"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "chat":
                # 串流回應
                await stream_chat_response(client_id, data.get("message", ""))

            elif data.get("type") == "ping":
                await manager.send_message(client_id, {"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def stream_chat_response(client_id: str, message: str):
    """串流聊天回應"""
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        stream=True
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            await manager.send_message(client_id, {
                "type": "chat_chunk",
                "content": chunk.choices[0].delta.content
            })

    await manager.send_message(client_id, {
        "type": "chat_complete"
    })
```

## 5. 訊息佇列整合

### Kafka 即時處理

```python
from confluent_kafka import Producer, Consumer, KafkaError
import json
from typing import Callable, Dict, Any
import threading

class KafkaMLPipeline:
    """Kafka ML 管線"""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "ml-pipeline"
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id

        # Producer
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "ml-producer"
        })

        # Consumer
        self.consumer = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest"
        })

        self.handlers: Dict[str, Callable] = {}
        self.running = False

    def register_handler(self, topic: str, handler: Callable):
        """註冊處理器"""
        self.handlers[topic] = handler

    def produce(self, topic: str, message: Dict[str, Any]):
        """發送訊息"""
        self.producer.produce(
            topic,
            value=json.dumps(message).encode("utf-8")
        )
        self.producer.flush()

    def start_consuming(self, topics: list[str]):
        """開始消費"""
        self.consumer.subscribe(topics)
        self.running = True

        while self.running:
            msg = self.consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Error: {msg.error()}")
                    break

            # 處理訊息
            topic = msg.topic()
            value = json.loads(msg.value().decode("utf-8"))

            if topic in self.handlers:
                try:
                    result = self.handlers[topic](value)

                    # 發送結果
                    if result:
                        self.produce(f"{topic}_results", result)
                except Exception as e:
                    print(f"Handler error: {e}")

    def stop(self):
        """停止消費"""
        self.running = False
        self.consumer.close()

# ML 處理器範例
def ml_inference_handler(message: Dict[str, Any]) -> Dict[str, Any]:
    """ML 推論處理器"""
    request_id = message.get("request_id")
    inputs = message.get("inputs", [])

    # 執行推論
    predictions = [x * 2 for x in inputs]  # 模擬

    return {
        "request_id": request_id,
        "predictions": predictions,
        "status": "completed"
    }

# 使用範例
pipeline = KafkaMLPipeline()
pipeline.register_handler("ml_requests", ml_inference_handler)

# 在背景執行緒中消費
# thread = threading.Thread(target=pipeline.start_consuming, args=(["ml_requests"],))
# thread.start()
```

## 6. 效能優化技巧

### 批次處理優化

```python
import asyncio
from typing import List, Any, Callable
from dataclasses import dataclass
import time

@dataclass
class BatchConfig:
    """批次配置"""
    max_batch_size: int = 32
    max_wait_time: float = 0.05  # 50ms

class DynamicBatcher:
    """動態批次處理器"""

    def __init__(
        self,
        process_fn: Callable[[List[Any]], List[Any]],
        config: BatchConfig = None
    ):
        self.process_fn = process_fn
        self.config = config or BatchConfig()

        self.pending_items: List[Any] = []
        self.pending_futures: List[asyncio.Future] = []
        self.lock = asyncio.Lock()
        self.batch_task = None

    async def add_item(self, item: Any) -> Any:
        """新增項目並等待結果"""
        future = asyncio.get_event_loop().create_future()

        async with self.lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)

            # 如果達到批次大小，立即處理
            if len(self.pending_items) >= self.config.max_batch_size:
                await self._process_batch()
            elif self.batch_task is None:
                # 啟動定時器
                self.batch_task = asyncio.create_task(
                    self._wait_and_process()
                )

        return await future

    async def _wait_and_process(self):
        """等待並處理"""
        await asyncio.sleep(self.config.max_wait_time)

        async with self.lock:
            if self.pending_items:
                await self._process_batch()
            self.batch_task = None

    async def _process_batch(self):
        """處理批次"""
        items = self.pending_items
        futures = self.pending_futures

        self.pending_items = []
        self.pending_futures = []

        try:
            # 執行批次處理
            results = await asyncio.to_thread(
                self.process_fn, items
            )

            # 分發結果
            for future, result in zip(futures, results):
                future.set_result(result)

        except Exception as e:
            # 分發錯誤
            for future in futures:
                future.set_exception(e)

# 使用範例
def batch_inference(items: List[dict]) -> List[dict]:
    """批次推論"""
    # 模擬批次處理
    return [{"prediction": item["value"] * 2} for item in items]

batcher = DynamicBatcher(batch_inference)

async def handle_request(value: float):
    result = await batcher.add_item({"value": value})
    return result
```

### 連接池管理

```python
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

class ConnectionPool:
    """連接池"""

    def __init__(
        self,
        create_connection: callable,
        max_size: int = 10,
        min_size: int = 2
    ):
        self.create_connection = create_connection
        self.max_size = max_size
        self.min_size = min_size

        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.size = 0
        self.lock = asyncio.Lock()

    async def initialize(self):
        """初始化最小連接數"""
        for _ in range(self.min_size):
            conn = await self.create_connection()
            await self.pool.put(conn)
            self.size += 1

    @asynccontextmanager
    async def acquire(self):
        """取得連接"""
        conn = None

        try:
            # 嘗試從池中取得
            try:
                conn = self.pool.get_nowait()
            except asyncio.QueueEmpty:
                # 如果池為空且未達上限，建立新連接
                async with self.lock:
                    if self.size < self.max_size:
                        conn = await self.create_connection()
                        self.size += 1

                # 如果達到上限，等待
                if conn is None:
                    conn = await self.pool.get()

            yield conn

        finally:
            # 歸還連接
            if conn is not None:
                try:
                    self.pool.put_nowait(conn)
                except asyncio.QueueFull:
                    # 池已滿，關閉連接
                    await conn.close()
                    async with self.lock:
                        self.size -= 1

# 使用範例
async def create_db_connection():
    """建立資料庫連接"""
    # 模擬連接建立
    await asyncio.sleep(0.1)
    return {"connected": True}

pool = ConnectionPool(create_db_connection, max_size=20)

async def query_database():
    async with pool.acquire() as conn:
        # 使用連接
        result = {"data": "example"}
        return result
```

## 延遲優化指標

```
┌─────────────────────────────────────────────────────────────┐
│                    延遲優化目標                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  層級              目標延遲           優化策略              │
│  ─────────────────────────────────────────────────────────  │
│  網路層            < 5ms             CDN, 區域部署           │
│  API Gateway       < 10ms            快取, 連接池           │
│  特徵擷取          < 20ms            Redis, 預計算          │
│  模型推論          < 50ms            GPU, 量化, 批次        │
│  回應處理          < 5ms             串流, 壓縮             │
│  ─────────────────────────────────────────────────────────  │
│  總延遲            < 100ms           端對端優化             │
│                                                             │
│  P99 延遲          < 200ms           異常處理, 超時控制     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 延伸閱讀

- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [Feature Store Best Practices](https://www.featurestore.org/)
