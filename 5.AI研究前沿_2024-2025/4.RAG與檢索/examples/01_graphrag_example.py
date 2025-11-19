"""
GraphRAG 知識圖譜增強檢索
結合知識圖譜與RAG技術，實現複雜推理和多跳問答
"""

from typing import List, Dict, Tuple
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Entity:
    """實體"""
    id: str
    name: str
    type: str
    description: str
    embedding: np.ndarray = None


@dataclass
class Relation:
    """關係"""
    source: str
    target: str
    relation_type: str
    description: str


class KnowledgeGraph:
    """知識圖譜"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

    def add_entity(self, entity: Entity):
        """添加實體"""
        self.entities[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description
        )

    def add_relation(self, relation: Relation):
        """添加關係"""
        self.relations.append(relation)
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            description=relation.description
        )

    def get_neighbors(self, entity_id: str, relation_type: str = None) -> List[str]:
        """獲取鄰居實體"""
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            if relation_type is None or edge_data.get('relation_type') == relation_type:
                neighbors.append(neighbor)
        return neighbors

    def get_paths(
        self,
        source: str,
        target: str,
        max_length: int = 3
    ) -> List[List[str]]:
        """獲取兩個實體間的路徑"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, entity_ids: List[str], depth: int = 1) -> 'KnowledgeGraph':
        """獲取子圖"""
        nodes = set(entity_ids)

        # 擴展到指定深度的鄰居
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)

        # 創建子圖
        subgraph = KnowledgeGraph()
        for node_id in nodes:
            if node_id in self.entities:
                subgraph.add_entity(self.entities[node_id])

        for relation in self.relations:
            if relation.source in nodes and relation.target in nodes:
                subgraph.add_relation(relation)

        return subgraph


class GraphRAG:
    """GraphRAG 系統"""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        初始化 GraphRAG

        Args:
            embedding_model: 嵌入模型
        """
        self.kg = KnowledgeGraph()
        self.encoder = SentenceTransformer(embedding_model)
        print(f"已載入嵌入模型: {embedding_model}")

    def build_knowledge_graph(
        self,
        documents: List[str],
        extract_entities_fn=None,
        extract_relations_fn=None
    ):
        """
        從文檔構建知識圖譜

        Args:
            documents: 文檔列表
            extract_entities_fn: 實體抽取函數
            extract_relations_fn: 關係抽取函數
        """
        print(f"正在從 {len(documents)} 個文檔構建知識圖譜...")

        # 簡化版：使用規則抽取（實際應用中應使用 NER 和關係抽取模型）
        if extract_entities_fn is None:
            extract_entities_fn = self._simple_entity_extraction

        if extract_relations_fn is None:
            extract_relations_fn = self._simple_relation_extraction

        # 抽取實體和關係
        all_entities = []
        all_relations = []

        for doc in documents:
            entities = extract_entities_fn(doc)
            relations = extract_relations_fn(doc, entities)

            all_entities.extend(entities)
            all_relations.extend(relations)

        # 添加到知識圖譜
        for entity in all_entities:
            # 計算嵌入
            entity.embedding = self.encoder.encode(
                f"{entity.name}: {entity.description}"
            )
            self.kg.add_entity(entity)

        for relation in all_relations:
            self.kg.add_relation(relation)

        print(f"知識圖譜構建完成:")
        print(f"  - 實體數: {len(self.kg.entities)}")
        print(f"  - 關係數: {len(self.kg.relations)}")

    def _simple_entity_extraction(self, text: str) -> List[Entity]:
        """簡單的實體抽取（示例用）"""
        # 實際應用應使用 spaCy、BERT-NER 等
        entities = []
        # 這裡僅作示例，實際需要更複雜的NER
        return entities

    def _simple_relation_extraction(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """簡單的關係抽取（示例用）"""
        # 實際應用應使用關係抽取模型
        relations = []
        return relations

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_graph_expansion: bool = True,
        expansion_depth: int = 1
    ) -> Dict:
        """
        檢索相關信息

        Args:
            query: 查詢
            top_k: 返回Top-K實體
            use_graph_expansion: 是否使用圖擴展
            expansion_depth: 擴展深度

        Returns:
            檢索結果
        """
        print(f"\n查詢: {query}")

        # 1. 向量檢索相關實體
        query_embedding = self.encoder.encode(query)

        similarities = []
        for entity_id, entity in self.kg.entities.items():
            if entity.embedding is not None:
                sim = np.dot(query_embedding, entity.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entity.embedding)
                )
                similarities.append((entity_id, sim))

        # 排序並獲取Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_entities = [entity_id for entity_id, _ in similarities[:top_k]]

        print(f"找到 {len(top_entities)} 個相關實體")

        # 2. 圖擴展（可選）
        if use_graph_expansion:
            subgraph = self.kg.get_subgraph(top_entities, depth=expansion_depth)
            print(f"圖擴展後實體數: {len(subgraph.entities)}")
        else:
            subgraph = self.kg

        # 3. 構建上下文
        context = self._build_context(subgraph, top_entities)

        return {
            "entities": top_entities,
            "subgraph": subgraph,
            "context": context
        }

    def _build_context(
        self,
        subgraph: KnowledgeGraph,
        focus_entities: List[str]
    ) -> str:
        """構建上下文文本"""
        context_parts = []

        # 添加實體信息
        context_parts.append("=== 相關實體 ===")
        for entity_id in focus_entities:
            if entity_id in subgraph.entities:
                entity = subgraph.entities[entity_id]
                context_parts.append(
                    f"- {entity.name} ({entity.type}): {entity.description}"
                )

        # 添加關係信息
        context_parts.append("\n=== 相關關係 ===")
        for relation in subgraph.relations:
            if relation.source in focus_entities or relation.target in focus_entities:
                source = subgraph.entities[relation.source].name
                target = subgraph.entities[relation.target].name
                context_parts.append(
                    f"- {source} --[{relation.relation_type}]--> {target}: {relation.description}"
                )

        return "\n".join(context_parts)

    def answer_question(
        self,
        question: str,
        llm_generate_fn=None
    ) -> str:
        """
        回答問題

        Args:
            question: 問題
            llm_generate_fn: LLM 生成函數

        Returns:
            答案
        """
        # 檢索相關信息
        result = self.retrieve(question)

        # 構建提示
        prompt = f"""基於以下知識圖譜信息回答問題。

{result['context']}

問題: {question}

答案:"""

        # 使用 LLM 生成答案
        if llm_generate_fn is not None:
            answer = llm_generate_fn(prompt)
        else:
            # 如果沒有 LLM，返回檢索到的上下文
            answer = "（請集成 LLM 以生成答案）\n\n" + result['context']

        return answer


def example_build_knowledge_graph():
    """示例 1: 構建知識圖譜"""
    print("=== 示例 1: 構建知識圖譜 ===")

    # 創建 GraphRAG 實例
    graph_rag = GraphRAG()

    # 手動構建示例知識圖譜（實際應從文檔抽取）
    # 添加實體
    entities = [
        Entity("e1", "GPT-4", "Model", "OpenAI 的大型語言模型"),
        Entity("e2", "OpenAI", "Organization", "人工智能研究公司"),
        Entity("e3", "Transformer", "Architecture", "注意力機制為基礎的模型架構"),
        Entity("e4", "ChatGPT", "Product", "基於 GPT 的對話式 AI"),
        Entity("e5", "RLHF", "Technique", "人類反饋強化學習"),
    ]

    for entity in entities:
        entity.embedding = graph_rag.encoder.encode(
            f"{entity.name}: {entity.description}"
        )
        graph_rag.kg.add_entity(entity)

    # 添加關係
    relations = [
        Relation("e1", "e2", "developed_by", "GPT-4 由 OpenAI 開發"),
        Relation("e1", "e3", "based_on", "GPT-4 基於 Transformer 架構"),
        Relation("e4", "e1", "uses", "ChatGPT 使用 GPT-4"),
        Relation("e1", "e5", "trained_with", "GPT-4 使用 RLHF 訓練"),
    ]

    for relation in relations:
        graph_rag.kg.add_relation(relation)

    print("知識圖譜構建完成！")

    # 可視化（可選）
    print(f"\n實體: {list(graph_rag.kg.entities.keys())}")
    print(f"關係數: {len(graph_rag.kg.relations)}")

    return graph_rag


def example_retrieval():
    """示例 2: 檢索查詢"""
    print("\n=== 示例 2: GraphRAG 檢索 ===")

    # 構建知識圖譜
    graph_rag = example_build_knowledge_graph()

    # 執行查詢
    queries = [
        "GPT-4 是什麼？",
        "誰開發了 GPT-4？",
        "ChatGPT 使用什麼模型？"
    ]

    for query in queries:
        result = graph_rag.retrieve(query, top_k=3)
        print(f"\n查詢: {query}")
        print(f"相關實體: {result['entities']}")
        print(f"\n上下文:\n{result['context']}")


def example_multi_hop_reasoning():
    """示例 3: 多跳推理"""
    print("\n=== 示例 3: 多跳推理 ===")

    graph_rag = example_build_knowledge_graph()

    # 需要多跳推理的查詢
    query = "ChatGPT 基於什麼架構？"

    # ChatGPT -> GPT-4 -> Transformer (需要兩跳)
    result = graph_rag.retrieve(query, use_graph_expansion=True, expansion_depth=2)

    print(f"\n查詢: {query}")
    print(f"\n上下文:\n{result['context']}")
    print("\n（通過圖擴展，系統能找到 ChatGPT -> GPT-4 -> Transformer 的路徑）")


if __name__ == "__main__":
    print("GraphRAG 知識圖譜增強檢索示例")
    print("=" * 60)

    # 運行示例
    example_build_knowledge_graph()
    example_retrieval()
    example_multi_hop_reasoning()

    print("\n所有示例完成！")
    print("\nGraphRAG 優勢:")
    print("1. 支持多跳推理")
    print("2. 結構化知識表達")
    print("3. 更好的可解釋性")
    print("4. 適合複雜領域知識")

    print("\n實際應用建議:")
    print("1. 使用專業 NER 模型抽取實體（spaCy, BERT-NER）")
    print("2. 使用關係抽取模型（OpenIE, REBEL）")
    print("3. 整合 LLM 進行答案生成")
    print("4. 使用圖數據庫（Neo4j）存儲大規模知識圖譜")
