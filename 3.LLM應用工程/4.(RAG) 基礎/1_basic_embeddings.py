"""
基礎嵌入向量範例
展示如何使用 Sentence Transformers 生成文本嵌入向量並計算相似度
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def demo_basic_embeddings():
    """基礎嵌入向量演示"""
    print("=" * 60)
    print("基礎嵌入向量演示")
    print("=" * 60)

    # 載入預訓練模型
    print("\n1. 載入 Sentence Transformer 模型...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   模型載入完成！")

    # 準備文檔
    documents = [
        "機器學習是人工智慧的一個分支，通過數據學習模式",
        "深度學習使用多層神經網絡來處理複雜數據",
        "自然語言處理幫助計算機理解和生成人類語言",
        "計算機視覺專注於讓計算機理解圖像和視頻",
        "強化學習通過獎勵機制訓練智能體"
    ]

    print(f"\n2. 準備了 {len(documents)} 個文檔")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc}")

    # 生成嵌入向量
    print("\n3. 生成嵌入向量...")
    embeddings = model.encode(documents)
    print(f"   嵌入向量形狀: {embeddings.shape}")
    print(f"   每個文檔被轉換為 {embeddings.shape[1]} 維向量")
    print(f"   第一個文檔的向量 (前10維): {embeddings[0][:10]}")

    # 查詢
    query = "什麼是神經網絡？"
    print(f"\n4. 用戶查詢: '{query}'")

    # 生成查詢向量
    query_embedding = model.encode([query])
    print(f"   查詢向量形狀: {query_embedding.shape}")

    # 計算相似度
    print("\n5. 計算餘弦相似度...")
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # 排序並獲取最相似的文檔
    most_similar_indices = np.argsort(similarities)[::-1]

    print("\n6. 檢索結果 (按相似度排序):")
    print("-" * 60)
    for rank, idx in enumerate(most_similar_indices, 1):
        print(f"\n   排名 {rank}: 相似度 {similarities[idx]:.4f}")
        print(f"   內容: {documents[idx]}")

    return model, documents, embeddings


def demo_multilingual_embeddings():
    """多語言嵌入向量演示"""
    print("\n\n" + "=" * 60)
    print("多語言嵌入向量演示")
    print("=" * 60)

    # 載入多語言模型
    print("\n1. 載入多語言模型...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("   模型載入完成！")

    # 多語言文檔
    documents = {
        "中文": "機器學習是人工智慧的重要分支",
        "英文": "Machine learning is an important branch of artificial intelligence",
        "日文": "機械学習は人工知能の重要な分野です",
        "韓文": "기계 학습은 인공 지능의 중요한 분야입니다"
    }

    print("\n2. 多語言文檔:")
    for lang, doc in documents.items():
        print(f"   {lang}: {doc}")

    # 生成嵌入向量
    doc_list = list(documents.values())
    embeddings = model.encode(doc_list)

    # 計算相似度矩陣
    similarity_matrix = cosine_similarity(embeddings)

    print("\n3. 跨語言相似度矩陣:")
    print("-" * 60)

    langs = list(documents.keys())
    # 打印表頭
    print(f"{'':>8}", end="")
    for lang in langs:
        print(f"{lang:>10}", end="")
    print()

    # 打印矩陣
    for i, lang1 in enumerate(langs):
        print(f"{lang1:>8}", end="")
        for j, lang2 in enumerate(langs):
            print(f"{similarity_matrix[i][j]:>10.4f}", end="")
        print()

    print("\n   分析: 所有語言版本的相似度都很高，")
    print("   說明多語言模型能夠理解不同語言的語義相似性！")


def demo_semantic_search():
    """語義搜索演示"""
    print("\n\n" + "=" * 60)
    print("語義搜索演示")
    print("=" * 60)

    # 載入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 知識庫文檔
    knowledge_base = [
        "Python 是一種高級程式語言，廣泛應用於數據科學和機器學習",
        "Java 是一種物件導向的程式語言，常用於企業級應用開發",
        "JavaScript 主要用於網頁前端開發，也可用於後端開發",
        "深度學習框架如 TensorFlow 和 PyTorch 簡化了神經網絡的開發",
        "Git 是一個版本控制系統，幫助開發者管理代碼變更",
        "Docker 容器技術使應用部署更加便捷和一致",
        "Kubernetes 用於自動化容器應用的部署、擴展和管理",
        "SQL 是用於管理和查詢關係數據庫的標準語言",
        "NumPy 是 Python 的數值計算庫，提供高效的數組操作",
        "Pandas 是數據分析工具，提供了強大的數據結構和分析功能"
    ]

    print(f"\n知識庫包含 {len(knowledge_base)} 個文檔")

    # 生成知識庫嵌入
    print("\n正在為知識庫生成嵌入向量...")
    kb_embeddings = model.encode(knowledge_base)
    print("完成！")

    # 測試查詢
    queries = [
        "最適合機器學習的程式語言是什麼？",
        "如何管理代碼版本？",
        "有哪些容器相關的技術？",
        "Python 中哪個庫適合數據分析？"
    ]

    print("\n" + "=" * 60)
    for query in queries:
        print(f"\n查詢: {query}")
        print("-" * 60)

        # 生成查詢嵌入
        query_embedding = model.encode([query])

        # 計算相似度
        similarities = cosine_similarity(query_embedding, kb_embeddings)[0]

        # 獲取 Top-3 結果
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]

        print(f"Top {top_k} 相關文檔:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"\n  {rank}. 相似度: {similarities[idx]:.4f}")
            print(f"     內容: {knowledge_base[idx]}")


def main():
    """主函數"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "基礎嵌入向量範例" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    # 執行各個演示
    demo_basic_embeddings()
    demo_multilingual_embeddings()
    demo_semantic_search()

    print("\n\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n重點回顧:")
    print("1. 嵌入向量將文本轉換為數值向量，捕捉語義信息")
    print("2. 餘弦相似度用於衡量文本之間的語義相似性")
    print("3. 多語言模型可以理解不同語言的語義關聯")
    print("4. 語義搜索比關鍵字搜索更智能，能理解查詢意圖")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
