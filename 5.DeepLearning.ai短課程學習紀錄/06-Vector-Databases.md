# Vector Databases: from Embeddings to Applications

## 📋 課程概述

深入了解向量資料庫的原理、應用和實作，學習如何建立高效的語意搜尋系統。

### 課程目標
- 理解向量嵌入和語意搜尋原理
- 掌握主流向量資料庫的使用
- 學習近似最近鄰搜尋（ANN）演算法
- 實作推薦系統和搜尋引擎

### 課程時長
約 1 小時

## 🎯 向量資料庫核心概念

### 什麼是向量嵌入（Embeddings）？

向量嵌入是將文本、圖像等資料轉換為數值向量的過程，使電腦能夠理解語意相似性。

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    """獲取文本的向量嵌入"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 範例
texts = [
    "台灣的夜市很有名",
    "台北101是著名地標",
    "機器學習是AI的一個分支",
    "深度學習使用神經網路"
]

# 計算嵌入
embeddings = [get_embedding(text) for text in texts]

print(f"嵌入維度：{len(embeddings[0])}")

# 計算相似度
def cosine_similarity(v1, v2):
    """計算餘弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\n相似度矩陣：")
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:  # 只計算上三角
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"'{text1}' <-> '{text2}': {sim:.4f}")
```

## 📊 主流向量資料庫比較

### 1. Pinecone（雲端服務）

```python
from pinecone import Pinecone, ServerlessSpec

# 初始化
pc = Pinecone(api_key="your-api-key")

# 建立索引
index_name = "taiwan-docs"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI ada-002 的維度
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 連接索引
index = pc.Index(index_name)

# 插入向量
vectors = [
    {
        "id": "doc1",
        "values": get_embedding("台灣美食"),
        "metadata": {"category": "food", "text": "台灣美食"}
    },
    {
        "id": "doc2",
        "values": get_embedding("台北景點"),
        "metadata": {"category": "tourism", "text": "台北景點"}
    }
]

index.upsert(vectors=vectors)

# 搜尋
query_embedding = get_embedding("美食推薦")
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for match in results.matches:
    print(f"分數：{match.score:.4f}")
    print(f"文本：{match.metadata['text']}\n")
```

### 2. Chroma（本地資料庫）

```python
import chromadb
from chromadb.config import Settings

# 初始化客戶端
client = chromadb.PersistentClient(path="./chroma_db")

# 建立集合
collection = client.get_or_create_collection(
    name="taiwan_knowledge",
    metadata={"description": "台灣相關知識庫"}
)

# 新增文檔
documents = [
    "台灣位於東亞，是一個美麗的島嶼",
    "台北是台灣的首都，有很多景點",
    "台灣的夜市文化非常豐富",
    "珍珠奶茶是台灣的代表性飲品"
]

collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))],
    metadatas=[{"source": f"doc{i}"} for i in range(len(documents))]
)

# 查詢
results = collection.query(
    query_texts=["台灣有什麼特色？"],
    n_results=3
)

print("查詢結果：")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    print(f"{i+1}. {doc} (距離: {distance:.4f})")
```

### 3. Weaviate（混合搜尋）

```python
import weaviate

# 連接 Weaviate
client = weaviate.Client("http://localhost:8080")

# 定義 schema
schema = {
    "classes": [{
        "class": "Article",
        "description": "台灣相關文章",
        "vectorizer": "text2vec-openai",
        "properties": [
            {
                "name": "title",
                "dataType": ["string"],
                "description": "文章標題"
            },
            {
                "name": "content",
                "dataType": ["text"],
                "description": "文章內容"
            },
            {
                "name": "category",
                "dataType": ["string"],
                "description": "分類"
            }
        ]
    }]
}

# 建立 schema（如果不存在）
# client.schema.create(schema)

# 新增資料
article = {
    "title": "台灣夜市指南",
    "content": "台灣有許多著名夜市，如士林夜市、逢甲夜市等...",
    "category": "tourism"
}

# client.data_object.create(article, "Article")

# 向量搜尋
result = client.query.get(
    "Article", ["title", "content", "category"]
).with_near_text({
    "concepts": ["台灣美食"]
}).with_limit(5).do()

print(result)
```

### 4. FAISS（高效能本地搜尋）

```python
import faiss
import numpy as np

# 建立索引
dimension = 1536  # 向量維度
index = faiss.IndexFlatL2(dimension)  # L2 距離

# 也可以使用更高效的索引
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# 準備資料
embeddings_array = np.array(embeddings).astype('float32')

# 新增向量
index.add(embeddings_array)

print(f"索引中的向量數量：{index.ntotal}")

# 搜尋
query_vector = np.array([get_embedding("台灣文化")]).astype('float32')
k = 3  # 返回最相似的 3 個

distances, indices = index.search(query_vector, k)

print("\n搜尋結果：")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. 文本：{texts[idx]}")
    print(f"   距離：{dist:.4f}\n")

# 儲存索引
faiss.write_index(index, "vector.index")

# 載入索引
index = faiss.read_index("vector.index")
```

## 🔍 近似最近鄰搜尋（ANN）

### HNSW (Hierarchical Navigable Small World)

```python
import hnswlib
import numpy as np

# 初始化 HNSW 索引
dimension = 1536
num_elements = 10000

# 建立索引
index = hnswlib.Index(space='cosine', dim=dimension)

# 初始化索引
index.init_index(
    max_elements=num_elements,
    ef_construction=200,  # 建構時的搜尋深度
    M=16  # 每個節點的連接數
)

# 準備資料（模擬）
data = np.random.random((1000, dimension)).astype('float32')
labels = np.arange(1000)

# 新增資料
index.add_items(data, labels)

# 設定查詢參數
index.set_ef(50)  # 查詢時的搜尋深度

# 查詢
query = np.random.random((1, dimension)).astype('float32')
labels, distances = index.knn_query(query, k=5)

print(f"最近的 5 個鄰居：{labels[0]}")
print(f"距離：{distances[0]}")

# 儲存索引
index.save_index("hnsw.bin")

# 載入索引
index = hnswlib.Index(space='cosine', dim=dimension)
index.load_index("hnsw.bin")
```

## 💡 實戰應用

### 1. 語意搜尋引擎

```python
from typing import List, Dict
import chromadb
from openai import OpenAI

class SemanticSearchEngine:
    def __init__(self, collection_name="semantic_search"):
        self.client = chromadb.PersistentClient(path="./search_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
        self.openai_client = OpenAI()

    def add_documents(self, documents: List[Dict[str, str]]):
        """新增文檔到搜尋引擎"""
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        print(f"✅ 已新增 {len(documents)} 個文檔")

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜尋相關文檔"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # 格式化結果
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "score": 1 - results['distances'][0][i],  # 轉換為相似度分數
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            })

        return formatted_results

    def hybrid_search(self, query: str, filters: Dict = None, n_results: int = 5):
        """混合搜尋（語意 + 過濾）"""
        where_clause = filters if filters else None

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause
        )

        return self._format_results(results)

    def _format_results(self, results):
        """格式化結果"""
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "score": 1 - results['distances'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
            })
        return formatted

# 使用範例
search_engine = SemanticSearchEngine()

# 新增文檔
documents = [
    {
        "id": "doc1",
        "text": "台灣的夜市文化非常豐富，每個城市都有特色夜市",
        "metadata": {"category": "culture", "region": "taiwan"}
    },
    {
        "id": "doc2",
        "text": "台北101曾經是世界最高的建築物",
        "metadata": {"category": "landmark", "region": "taipei"}
    },
    {
        "id": "doc3",
        "text": "珍珠奶茶是台灣最著名的飲品之一",
        "metadata": {"category": "food", "region": "taiwan"}
    },
    {
        "id": "doc4",
        "text": "阿里山日出是台灣必看的美景",
        "metadata": {"category": "nature", "region": "chiayi"}
    }
]

search_engine.add_documents(documents)

# 搜尋
print("\n基本搜尋：")
results = search_engine.search("台灣美食", n_results=3)
for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']} (分數: {result['score']:.4f})")

# 混合搜尋（帶過濾）
print("\n混合搜尋（只搜尋美食類別）：")
filtered_results = search_engine.hybrid_search(
    "推薦",
    filters={"category": "food"},
    n_results=2
)
for result in filtered_results:
    print(f"- {result['text']}")
```

### 2. 推薦系統

```python
import numpy as np
from typing import List, Tuple

class RecommendationSystem:
    def __init__(self):
        self.items = {}  # 物品資訊
        self.embeddings = {}  # 物品嵌入
        self.openai_client = OpenAI()

    def add_items(self, items: List[Dict]):
        """新增物品"""
        for item in items:
            item_id = item['id']
            self.items[item_id] = item

            # 生成嵌入
            description = item.get('description', item.get('name', ''))
            embedding = self._get_embedding(description)
            self.embeddings[item_id] = embedding

        print(f"✅ 已新增 {len(items)} 個物品")

    def _get_embedding(self, text: str):
        """獲取文本嵌入"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def recommend(self, item_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """基於物品推薦相似物品"""
        if item_id not in self.embeddings:
            return []

        query_embedding = self.embeddings[item_id]
        similarities = []

        # 計算與所有其他物品的相似度
        for other_id, other_embedding in self.embeddings.items():
            if other_id != item_id:
                similarity = self._cosine_similarity(query_embedding, other_embedding)
                similarities.append((other_id, similarity))

        # 排序並返回 top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def recommend_by_description(self, description: str, n: int = 5) -> List[Tuple[str, float]]:
        """基於描述推薦物品"""
        query_embedding = self._get_embedding(description)
        similarities = []

        for item_id, item_embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, item_embedding)
            similarities.append((item_id, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def _cosine_similarity(self, v1, v2):
        """計算餘弦相似度"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get_item_info(self, item_id: str) -> Dict:
        """獲取物品資訊"""
        return self.items.get(item_id, {})

# 使用範例
recommender = RecommendationSystem()

# 新增物品（餐廳）
restaurants = [
    {
        "id": "r1",
        "name": "鼎泰豐",
        "description": "著名的小籠包餐廳，提供精緻台式點心"
    },
    {
        "id": "r2",
        "name": "阜杭豆漿",
        "description": "傳統台式早餐店，豆漿和燒餅油條很有名"
    },
    {
        "id": "r3",
        "name": "欣葉台菜",
        "description": "經典台灣菜餐廳，提供道地台灣料理"
    },
    {
        "id": "r4",
        "name": "添好運",
        "description": "港式點心餐廳，米其林一星"
    }
]

recommender.add_items(restaurants)

# 推薦相似餐廳
print("\n與鼎泰豐相似的餐廳：")
recommendations = recommender.recommend("r1", n=3)
for item_id, score in recommendations:
    info = recommender.get_item_info(item_id)
    print(f"- {info['name']} (相似度: {score:.4f})")
    print(f"  {info['description']}\n")

# 基於描述推薦
print("想吃傳統台灣早餐：")
recommendations = recommender.recommend_by_description("傳統台灣早餐", n=2)
for item_id, score in recommendations:
    info = recommender.get_item_info(item_id)
    print(f"- {info['name']} (相似度: {score:.4f})")
```

### 3. 異常檢測

```python
import numpy as np
from sklearn.preprocessing import normalize

class AnomalyDetector:
    def __init__(self, threshold: float = 0.7):
        """
        Args:
            threshold: 相似度閾值，低於此值視為異常
        """
        self.threshold = threshold
        self.normal_embeddings = []

    def fit(self, normal_texts: List[str]):
        """使用正常樣本訓練"""
        self.normal_embeddings = [
            self._get_embedding(text) for text in normal_texts
        ]
        print(f"✅ 已訓練 {len(normal_texts)} 個正常樣本")

    def _get_embedding(self, text: str):
        """獲取嵌入"""
        client = OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)

    def predict(self, text: str) -> Dict:
        """檢測是否為異常"""
        query_embedding = self._get_embedding(text)

        # 計算與所有正常樣本的最大相似度
        max_similarity = 0
        for normal_embedding in self.normal_embeddings:
            similarity = self._cosine_similarity(query_embedding, normal_embedding)
            max_similarity = max(max_similarity, similarity)

        is_anomaly = max_similarity < self.threshold

        return {
            "is_anomaly": is_anomaly,
            "max_similarity": max_similarity,
            "threshold": self.threshold
        }

    def _cosine_similarity(self, v1, v2):
        """計算餘弦相似度"""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 使用範例
detector = AnomalyDetector(threshold=0.75)

# 正常的客服訊息
normal_messages = [
    "我想查詢訂單狀態",
    "如何退換貨？",
    "產品價格是多少？",
    "配送時間要多久？",
    "可以修改訂單嗎？"
]

detector.fit(normal_messages)

# 測試訊息
test_messages = [
    "我要查詢我的訂單",  # 正常
    "這個產品多少錢？",  # 正常
    "快速賺錢的方法",    # 異常
    "免費贈送iPhone"     # 異常
]

print("\n異常檢測結果：")
for msg in test_messages:
    result = detector.predict(msg)
    status = "🚨 異常" if result['is_anomaly'] else "✅ 正常"
    print(f"{status} - {msg}")
    print(f"   相似度：{result['max_similarity']:.4f}\n")
```

## ✅ 最佳實踐

### 1. 選擇合適的向量資料庫

| 資料庫 | 適用場景 | 優點 | 缺點 |
|--------|---------|------|------|
| Pinecone | 生產環境、大規模 | 全託管、高效能 | 需付費 |
| Chroma | 開發測試、中小規模 | 簡單易用、免費 | 功能較少 |
| Weaviate | 企業應用 | 功能豐富、支援混合搜尋 | 複雜度高 |
| FAISS | 高效能需求 | 速度快、免費 | 需自行管理 |

### 2. 索引優化
- 選擇適當的索引類型（Flat, IVF, HNSW）
- 調整參數平衡速度和準確度
- 定期重建索引以優化效能

### 3. 嵌入模型選擇
- **OpenAI ada-002/003**: 品質好，但需付費
- **多語言模型**: 處理中文選擇支援的模型
- **開源模型**: Sentence Transformers（免費）

## 📚 延伸學習

- **進階 ANN 演算法**: LSH, Product Quantization
- **混合搜尋**: 結合關鍵字和向量搜尋
- **多模態搜尋**: 文字、圖像、音訊的聯合搜尋

---

**課程連結**：[DeepLearning.ai - Vector Databases](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)

**完成日期**：2025-01-17
