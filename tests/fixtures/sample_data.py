"""
Sample Test Data

提供測試使用的範例數據集合。
"""

# 範例文本數據
SAMPLE_TEXTS = {
    "zh_ml_intro": """
    機器學習是人工智能的一個分支，它使計算機能夠從數據中學習並做出決策或預測，
    而無需明確編程。深度學習是機器學習的一個子領域，使用多層神經網絡來學習
    數據的層次表示。
    """,
    "en_ml_intro": """
    Machine learning is a branch of artificial intelligence that enables computers
    to learn from data and make decisions or predictions without being explicitly
    programmed. Deep learning is a subset of machine learning that uses multi-layer
    neural networks to learn hierarchical representations of data.
    """,
    "zh_dl_intro": """
    深度學習是一種機器學習方法，使用深層神經網絡來處理複雜的模式識別任務。
    常見的架構包括卷積神經網絡（CNN）、循環神經網絡（RNN）和 Transformer。
    """,
}

# 範例文檔集合
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "機器學習基礎",
        "content": "監督學習、非監督學習和強化學習是機器學習的三大類型。",
        "metadata": {
            "category": "machine_learning",
            "language": "zh",
            "difficulty": "beginner",
        },
    },
    {
        "id": "doc_002",
        "title": "深度學習與神經網絡",
        "content": "神經網絡由輸入層、隱藏層和輸出層組成，通過反向傳播進行訓練。",
        "metadata": {
            "category": "deep_learning",
            "language": "zh",
            "difficulty": "intermediate",
        },
    },
    {
        "id": "doc_003",
        "title": "Introduction to LLM",
        "content": "Large Language Models are trained on massive text corpora using self-supervised learning.",
        "metadata": {
            "category": "llm",
            "language": "en",
            "difficulty": "advanced",
        },
    },
    {
        "id": "doc_004",
        "title": "RAG 系統設計",
        "content": "檢索增強生成（RAG）結合了檢索系統和生成模型，提高回答準確性。",
        "metadata": {
            "category": "rag",
            "language": "zh",
            "difficulty": "advanced",
        },
    },
]

# LLM API 模擬回應
MOCK_LLM_RESPONSES = {
    "openai": {
        "content": "這是一個模擬的 OpenAI GPT 回應。",
        "model": "gpt-4",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        "finish_reason": "stop",
    },
    "anthropic": {
        "content": "這是一個模擬的 Anthropic Claude 回應。",
        "model": "claude-3-opus",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
        },
        "stop_reason": "end_turn",
    },
}

# 範例向量數據
SAMPLE_EMBEDDINGS = {
    "dimension": 384,
    "vectors": [
        {"id": "vec_001", "values": [0.1] * 384, "metadata": {"text": "機器學習"}},
        {"id": "vec_002", "values": [0.2] * 384, "metadata": {"text": "深度學習"}},
        {"id": "vec_003", "values": [0.3] * 384, "metadata": {"text": "自然語言處理"}},
    ],
}

# 範例配置
SAMPLE_CONFIGS = {
    "model_config": {
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1.0,
    },
    "rag_config": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "retrieval_k": 5,
        "similarity_threshold": 0.7,
    },
    "training_config": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "early_stopping_patience": 3,
    },
}
