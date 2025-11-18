"""
測試 Graph RAG 系統

運行方式：
    python test_graph_rag.py
    或
    pytest test_graph_rag.py -v
"""

import os
import sys
import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from graph_rag import (
    KnowledgeGraphBuilder,
    GraphRAGSystem,
    Entity,
    Relation
)

load_dotenv()

# 測試文檔
TEST_DOCUMENTS = [
    "Alice 在 Google 工作。Google 總部位於加州。",
    "Bob 認識 Alice。Bob 使用 Python 編程。",
    "Python 是一種編程語言。Python 由 Guido van Rossum 創建。"
]


class TestKnowledgeGraphBuilder:
    """測試知識圖譜構建器"""

    @pytest.fixture
    def kg_builder(self):
        """創建知識圖譜構建器"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY 環境變數")
        return KnowledgeGraphBuilder()

    def test_extract_entities_and_relations(self, kg_builder):
        """測試實體和關係抽取"""
        text = "Alice 在 Google 工作。Google 位於舊金山。"
        entities, relations = kg_builder.extract_entities_and_relations(text)

        assert isinstance(entities, list)
        assert isinstance(relations, list)

        print(f"\n抽取到 {len(entities)} 個實體")
        for entity in entities:
            print(f"  - {entity.name} ({entity.entity_type})")

        print(f"\n抽取到 {len(relations)} 個關係")
        for relation in relations:
            print(f"  - {relation.source} --[{relation.relation_type}]--> {relation.target}")

    def test_add_entity(self, kg_builder):
        """測試添加實體"""
        entity = Entity(
            name="Python",
            entity_type="Programming Language",
            description="A high-level programming language"
        )

        kg_builder.add_entity(entity)

        assert "Python" in kg_builder.graph.nodes()
        assert kg_builder.graph.nodes["Python"]["entity_type"] == "Programming Language"

        print(f"\n✓ 成功添加實體: {entity.name}")

    def test_add_relation(self, kg_builder):
        """測試添加關係"""
        # 添加兩個實體
        kg_builder.add_entity(Entity("Alice", "Person"))
        kg_builder.add_entity(Entity("Google", "Organization"))

        # 添加關係
        relation = Relation(
            source="Alice",
            target="Google",
            relation_type="works_at"
        )
        kg_builder.add_relation(relation)

        assert kg_builder.graph.has_edge("Alice", "Google")
        edge_data = kg_builder.graph.get_edge_data("Alice", "Google")
        assert edge_data["relation_type"] == "works_at"

        print(f"\n✓ 成功添加關係: Alice --[works_at]--> Google")

    def test_build_from_documents(self, kg_builder):
        """測試從文檔構建圖譜"""
        kg_builder.build_from_documents(TEST_DOCUMENTS, verbose=False)

        assert kg_builder.graph.number_of_nodes() > 0
        assert kg_builder.graph.number_of_edges() >= 0

        print(f"\n✓ 成功構建知識圖譜")
        print(f"  節點數: {kg_builder.graph.number_of_nodes()}")
        print(f"  邊數: {kg_builder.graph.number_of_edges()}")

    def test_get_neighbors(self, kg_builder):
        """測試獲取鄰居節點"""
        # 構建簡單圖
        kg_builder.add_entity(Entity("A", "Type1"))
        kg_builder.add_entity(Entity("B", "Type2"))
        kg_builder.add_entity(Entity("C", "Type3"))
        kg_builder.add_relation(Relation("A", "B", "relates_to"))
        kg_builder.add_relation(Relation("B", "C", "connects_to"))

        neighbors = kg_builder.get_neighbors("B", direction="both")

        assert "A" in neighbors or "C" in neighbors
        print(f"\n✓ B 的鄰居: {neighbors}")

    def test_find_path(self, kg_builder):
        """測試路徑查找"""
        # 構建路徑 A -> B -> C
        kg_builder.add_entity(Entity("A", "Type1"))
        kg_builder.add_entity(Entity("B", "Type2"))
        kg_builder.add_entity(Entity("C", "Type3"))
        kg_builder.add_relation(Relation("A", "B", "step1"))
        kg_builder.add_relation(Relation("B", "C", "step2"))

        paths = kg_builder.find_path("A", "C", max_length=3)

        assert len(paths) > 0
        assert ["A", "B", "C"] in paths

        print(f"\n✓ 找到 {len(paths)} 條路徑")
        for path in paths:
            print(f"  路徑: {' -> '.join(path)}")

    def test_get_subgraph(self, kg_builder):
        """測試獲取子圖"""
        # 構建圖
        kg_builder.add_entity(Entity("Center", "Type"))
        kg_builder.add_entity(Entity("Node1", "Type"))
        kg_builder.add_entity(Entity("Node2", "Type"))
        kg_builder.add_relation(Relation("Center", "Node1", "connects"))
        kg_builder.add_relation(Relation("Center", "Node2", "connects"))

        subgraph = kg_builder.get_subgraph(["Center"], k_hop=1)

        assert subgraph.number_of_nodes() >= 2
        print(f"\n✓ 子圖: {subgraph.number_of_nodes()} 個節點")

    def test_visualize(self, kg_builder):
        """測試可視化"""
        # 構建小圖
        kg_builder.build_from_documents(TEST_DOCUMENTS[:2], verbose=False)

        output_path = "test_graph.png"
        kg_builder.visualize(output_path=output_path, figsize=(10, 8))

        assert os.path.exists(output_path)
        print(f"\n✓ 圖譜已保存到: {output_path}")

        # 清理
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_save_and_load(self, kg_builder):
        """測試保存和載入"""
        # 構建圖
        kg_builder.build_from_documents(TEST_DOCUMENTS[:2], verbose=False)
        original_nodes = kg_builder.graph.number_of_nodes()

        # 保存
        save_path = "test_graph.pkl"
        kg_builder.save_graph(save_path)
        assert os.path.exists(save_path)

        # 載入
        new_builder = KnowledgeGraphBuilder()
        new_builder.load_graph(save_path)

        assert new_builder.graph.number_of_nodes() == original_nodes
        print(f"\n✓ 成功保存和載入圖譜")

        # 清理
        if os.path.exists(save_path):
            os.remove(save_path)


class TestGraphRAGSystem:
    """測試 Graph RAG 系統"""

    @pytest.fixture
    def graph_rag(self):
        """創建 Graph RAG 系統"""
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("需要 OPENAI_API_KEY 環境變數")

        system = GraphRAGSystem()
        system.build_knowledge_graph(TEST_DOCUMENTS)
        return system

    def test_build_knowledge_graph(self, graph_rag):
        """測試構建知識圖譜"""
        assert graph_rag.kg_builder.graph.number_of_nodes() > 0
        print(f"\n✓ 知識圖譜構建成功")
        print(f"  節點數: {graph_rag.kg_builder.graph.number_of_nodes()}")

    def test_identify_entities_in_query(self, graph_rag):
        """測試識別查詢中的實體"""
        query = "Alice 和 Google 有什麼關係？"
        entities = graph_rag.identify_entities_in_query(query)

        assert isinstance(entities, list)
        print(f"\n查詢: {query}")
        print(f"識別到的實體: {entities}")

    def test_multi_hop_query(self, graph_rag):
        """測試多跳查詢"""
        query = "Alice 和 Python 有什麼關係？"
        result = graph_rag.multi_hop_query(query, max_hops=3, verbose=False)

        assert result is not None
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

        print(f"\n查詢: {query}")
        print(f"答案: {result.answer[:200]}...")

    def test_explain_relationship(self, graph_rag):
        """測試關係解釋"""
        # 先確保圖中有這些實體
        if "Alice" in graph_rag.kg_builder.graph.nodes():
            entities = list(graph_rag.kg_builder.graph.nodes())
            if len(entities) >= 2:
                entity1, entity2 = entities[0], entities[1]
                explanation = graph_rag.explain_relationship(entity1, entity2)

                assert isinstance(explanation, str)
                print(f"\n關係解釋 ({entity1} - {entity2}):")
                print(explanation[:200])


def run_manual_tests():
    """手動運行測試"""
    print("=" * 80)
    print("Graph RAG 系統測試")
    print("=" * 80)
    print()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 錯誤: 需要設置 OPENAI_API_KEY 環境變數")
        return

    try:
        # 測試 1: 知識圖譜構建
        print("1. 測試知識圖譜構建")
        print("-" * 80)

        kg_builder = KnowledgeGraphBuilder()
        print("✓ 初始化完成")

        kg_builder.build_from_documents(TEST_DOCUMENTS, verbose=True)
        print()

        # 測試 2: 圖操作
        print("2. 測試圖操作")
        print("-" * 80)

        # 測試路徑查找
        nodes = list(kg_builder.graph.nodes())
        if len(nodes) >= 2:
            source, target = nodes[0], nodes[-1]
            paths = kg_builder.find_path(source, target)
            print(f"從 {source} 到 {target} 的路徑: {len(paths)} 條")
            if paths:
                print(f"第一條路徑: {' -> '.join(paths[0])}")
        print()

        # 測試 3: Graph RAG 系統
        print("3. 測試 Graph RAG 系統")
        print("-" * 80)

        graph_rag = GraphRAGSystem()
        graph_rag.build_knowledge_graph(TEST_DOCUMENTS)
        print("✓ Graph RAG 初始化完成\n")

        # 測試查詢
        test_queries = [
            "Alice 在哪裡工作？",
            "Python 是誰創建的？"
        ]

        for query in test_queries:
            print(f"查詢: {query}")
            result = graph_rag.multi_hop_query(query, verbose=False)
            print(f"答案: {result.answer}\n")

        # 測試 4: 可視化
        print("4. 測試可視化")
        print("-" * 80)

        output_path = "test_graph_manual.png"
        kg_builder.visualize(output_path=output_path)

        if os.path.exists(output_path):
            print(f"✓ 圖譜已生成: {output_path}")
            # 可選：清理
            # os.remove(output_path)
        print()

        print("=" * 80)
        print("✅ 所有測試完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_manual_tests()
