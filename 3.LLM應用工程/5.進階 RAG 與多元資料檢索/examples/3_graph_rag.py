"""
Graph RAG: 知識圖譜增強檢索

這個模組展示如何使用知識圖譜來增強 RAG 系統：
1. 從文本自動構建知識圖譜
2. 基於圖結構的檢索
3. 多跳推理查詢
4. 實體關係抽取
5. 圖譜可視化

使用 NetworkX 構建本地知識圖譜（可擴展至 Neo4j）

使用場景：
- 複雜關係推理
- 多跳問答
- 知識發現
- 實體鏈接
"""

import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle

# 圖處理
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式後端

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Entity:
    """實體"""
    name: str
    entity_type: str
    description: str = ""
    properties: Dict = field(default_factory=dict)


@dataclass
class Relation:
    """關係"""
    source: str  # 來源實體
    target: str  # 目標實體
    relation_type: str  # 關係類型
    properties: Dict = field(default_factory=dict)


@dataclass
class GraphQueryResult:
    """圖查詢結果"""
    query: str
    entities: List[Entity]
    relations: List[Relation]
    paths: List[List[str]]  # 多跳路徑
    answer: str
    reasoning_chain: List[str]  # 推理鏈


class KnowledgeGraphBuilder:
    """
    知識圖譜構建器

    從文本中抽取實體和關係，構建知識圖譜
    """

    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        """初始化"""
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph = nx.DiGraph()
        self.entity_descriptions = {}  # 實體描述

    def extract_entities_and_relations(self, text: str) -> Tuple[List[Entity], List[Relation]]:
        """
        從文本中抽取實體和關係

        Args:
            text: 輸入文本

        Returns:
            (實體列表, 關係列表)
        """
        extraction_prompt = ChatPromptTemplate.from_template("""
從以下文本中抽取實體和它們之間的關係。

文本：
{text}

請抽取：
1. 實體：人物、組織、地點、概念、技術等
2. 關係：實體之間的關係（如：創建、位於、屬於、使用等）

輸出格式（JSON）：
{{
    "entities": [
        {{"name": "實體名稱", "type": "實體類型", "description": "簡短描述"}}
    ],
    "relations": [
        {{"source": "來源實體", "target": "目標實體", "type": "關係類型"}}
    ]
}}

JSON 輸出：
""")

        messages = extraction_prompt.format_messages(text=text)
        response = self.llm.invoke(messages)

        try:
            # 解析 JSON
            content = response.content.strip()
            # 移除可能的 markdown 標記
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            entities = [
                Entity(
                    name=e["name"],
                    entity_type=e.get("type", "Unknown"),
                    description=e.get("description", "")
                )
                for e in data.get("entities", [])
            ]

            relations = [
                Relation(
                    source=r["source"],
                    target=r["target"],
                    relation_type=r.get("type", "related_to")
                )
                for r in data.get("relations", [])
            ]

            return entities, relations

        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析失敗: {e}")
            print(f"原始內容: {response.content}")
            return [], []

    def add_entity(self, entity: Entity):
        """添加實體到圖譜"""
        self.graph.add_node(
            entity.name,
            entity_type=entity.entity_type,
            description=entity.description,
            **entity.properties
        )
        self.entity_descriptions[entity.name] = entity.description

    def add_relation(self, relation: Relation):
        """添加關係到圖譜"""
        # 確保節點存在
        if relation.source not in self.graph:
            self.graph.add_node(relation.source)
        if relation.target not in self.graph:
            self.graph.add_node(relation.target)

        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            **relation.properties
        )

    def build_from_documents(self, documents: List[str], verbose: bool = True):
        """
        從文檔構建知識圖譜

        Args:
            documents: 文檔列表
            verbose: 是否輸出進度
        """
        for i, doc in enumerate(documents):
            if verbose:
                print(f"處理文檔 {i+1}/{len(documents)}...")

            entities, relations = self.extract_entities_and_relations(doc)

            # 添加到圖譜
            for entity in entities:
                self.add_entity(entity)

            for relation in relations:
                self.add_relation(relation)

        if verbose:
            print(f"\n✓ 知識圖譜構建完成")
            print(f"  節點數: {self.graph.number_of_nodes()}")
            print(f"  邊數: {self.graph.number_of_edges()}")

    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """獲取實體資訊"""
        if entity_name in self.graph:
            return dict(self.graph.nodes[entity_name])
        return None

    def get_neighbors(self, entity_name: str, direction: str = "both") -> List[str]:
        """
        獲取鄰居節點

        Args:
            entity_name: 實體名稱
            direction: 方向（"in", "out", "both"）

        Returns:
            鄰居節點列表
        """
        if entity_name not in self.graph:
            return []

        if direction == "out":
            return list(self.graph.successors(entity_name))
        elif direction == "in":
            return list(self.graph.predecessors(entity_name))
        else:  # both
            return list(set(self.graph.successors(entity_name)) |
                       set(self.graph.predecessors(entity_name)))

    def find_path(self, source: str, target: str, max_length: int = 5) -> List[List[str]]:
        """
        尋找兩個實體之間的路徑

        Args:
            source: 來源實體
            target: 目標實體
            max_length: 最大路徑長度

        Returns:
            路徑列表
        """
        if source not in self.graph or target not in self.graph:
            return []

        try:
            # 找所有簡單路徑
            all_paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            return all_paths
        except nx.NetworkXNoPath:
            return []

    def get_subgraph(self, entities: List[str], k_hop: int = 1) -> nx.DiGraph:
        """
        獲取以指定實體為中心的子圖

        Args:
            entities: 中心實體列表
            k_hop: 跳數

        Returns:
            子圖
        """
        nodes = set(entities)

        # 擴展 k 跳鄰居
        for _ in range(k_hop):
            new_nodes = set()
            for node in nodes:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
            nodes.update(new_nodes)

        return self.graph.subgraph(nodes).copy()

    def visualize(self, output_path: str = "knowledge_graph.png", figsize: Tuple = (15, 10)):
        """
        可視化知識圖譜

        Args:
            output_path: 輸出圖片路徑
            figsize: 圖片大小
        """
        plt.figure(figsize=figsize)

        # 計算布局
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # 繪製節點
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('entity_type', 'Unknown')
            # 根據類型分配顏色
            type_colors = {
                'Person': '#FF6B6B',
                'Organization': '#4ECDC4',
                'Technology': '#45B7D1',
                'Concept': '#FFA07A',
                'Location': '#98D8C8'
            }
            node_colors.append(type_colors.get(node_type, '#CCCCCC'))

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=1000,
            alpha=0.9
        )

        # 繪製邊
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.5,
            width=2
        )

        # 繪製標籤
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=8,
            font_weight='bold'
        )

        # 繪製邊標籤
        edge_labels = {
            (u, v): data.get('relation_type', '')
            for u, v, data in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels,
            font_size=6
        )

        plt.title("Knowledge Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 知識圖譜已保存到: {output_path}")

    def save_graph(self, path: str = "knowledge_graph.pkl"):
        """保存圖譜"""
        with open(path, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'entity_descriptions': self.entity_descriptions
            }, f)
        print(f"✓ 知識圖譜已保存到: {path}")

    def load_graph(self, path: str = "knowledge_graph.pkl"):
        """載入圖譜"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.entity_descriptions = data['entity_descriptions']
        print(f"✓ 知識圖譜已從 {path} 載入")


class GraphRAGSystem:
    """
    Graph RAG 系統

    結合知識圖譜的 RAG 系統，支援：
    - 實體識別和檢索
    - 多跳推理
    - 關係查詢
    - 路徑解釋
    """

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        graph_path: Optional[str] = None
    ):
        """初始化"""
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.kg_builder = KnowledgeGraphBuilder(llm_model=llm_model)

        if graph_path and os.path.exists(graph_path):
            self.kg_builder.load_graph(graph_path)

    def build_knowledge_graph(self, documents: List[str]):
        """構建知識圖譜"""
        self.kg_builder.build_from_documents(documents)

    def identify_entities_in_query(self, query: str) -> List[str]:
        """
        識別查詢中的實體

        Args:
            query: 使用者查詢

        Returns:
            實體名稱列表
        """
        # 獲取圖譜中的所有實體
        all_entities = list(self.kg_builder.graph.nodes())

        if not all_entities:
            return []

        entity_prompt = ChatPromptTemplate.from_template("""
從使用者查詢中識別出存在於知識圖譜中的實體。

知識圖譜中的實體：
{entities}

使用者查詢：{query}

請列出查詢中提到的、存在於知識圖譜中的實體名稱。
只輸出實體名稱，每行一個，不要解釋。

實體：
""")

        messages = entity_prompt.format_messages(
            query=query,
            entities=", ".join(all_entities[:50])  # 限制數量避免超長
        )

        response = self.llm.invoke(messages)

        # 解析實體
        identified_entities = [
            line.strip()
            for line in response.content.strip().split('\n')
            if line.strip() and line.strip() in all_entities
        ]

        return identified_entities

    def multi_hop_query(
        self,
        query: str,
        max_hops: int = 3,
        verbose: bool = True
    ) -> GraphQueryResult:
        """
        多跳查詢

        支援需要多步推理的複雜查詢

        Args:
            query: 使用者查詢
            max_hops: 最大跳數
            verbose: 是否輸出詳細信息

        Returns:
            GraphQueryResult
        """
        result = GraphQueryResult(
            query=query,
            entities=[],
            relations=[],
            paths=[],
            answer="",
            reasoning_chain=[]
        )

        # 1. 識別查詢中的實體
        identified_entities = self.identify_entities_in_query(query)

        if verbose:
            print(f"識別到的實體: {identified_entities}")

        if not identified_entities:
            result.answer = "無法在知識圖譜中找到相關實體。"
            return result

        # 2. 獲取相關子圖
        subgraph = self.kg_builder.get_subgraph(identified_entities, k_hop=max_hops)

        if verbose:
            print(f"相關子圖: {subgraph.number_of_nodes()} 個節點, "
                  f"{subgraph.number_of_edges()} 條邊")

        # 3. 提取子圖信息
        for node in subgraph.nodes():
            entity_info = self.kg_builder.get_entity_info(node)
            if entity_info:
                result.entities.append(Entity(
                    name=node,
                    entity_type=entity_info.get('entity_type', 'Unknown'),
                    description=entity_info.get('description', '')
                ))

        for u, v, data in subgraph.edges(data=True):
            result.relations.append(Relation(
                source=u,
                target=v,
                relation_type=data.get('relation_type', 'related_to')
            ))

        # 4. 找出重要路徑
        if len(identified_entities) >= 2:
            for i in range(len(identified_entities)):
                for j in range(i + 1, len(identified_entities)):
                    paths = self.kg_builder.find_path(
                        identified_entities[i],
                        identified_entities[j],
                        max_length=max_hops
                    )
                    result.paths.extend(paths)

        if verbose and result.paths:
            print(f"找到 {len(result.paths)} 條路徑")
            for path in result.paths[:3]:  # 只顯示前3條
                print(f"  路徑: {' -> '.join(path)}")

        # 5. 基於圖譜信息生成答案
        result.answer = self.generate_answer_from_graph(query, result)

        return result

    def generate_answer_from_graph(
        self,
        query: str,
        graph_result: GraphQueryResult
    ) -> str:
        """
        基於圖譜信息生成答案

        Args:
            query: 使用者查詢
            graph_result: 圖查詢結果

        Returns:
            答案
        """
        # 構建上下文
        context_parts = []

        # 實體信息
        if graph_result.entities:
            context_parts.append("相關實體：")
            for entity in graph_result.entities[:10]:  # 限制數量
                context_parts.append(
                    f"- {entity.name} ({entity.entity_type}): {entity.description}"
                )

        # 關係信息
        if graph_result.relations:
            context_parts.append("\n關係：")
            for relation in graph_result.relations[:15]:  # 限制數量
                context_parts.append(
                    f"- {relation.source} --[{relation.relation_type}]--> {relation.target}"
                )

        # 路徑信息
        if graph_result.paths:
            context_parts.append("\n推理路徑：")
            for path in graph_result.paths[:5]:  # 限制數量
                context_parts.append(f"- {' → '.join(path)}")

        context = "\n".join(context_parts)

        # 生成答案
        answer_prompt = ChatPromptTemplate.from_template("""
基於知識圖譜中的信息回答問題。

知識圖譜信息：
{context}

問題：{query}

請基於圖譜中的實體、關係和路徑，提供準確、詳細的回答。
如果需要推理，請說明推理過程。

回答：
""")

        messages = answer_prompt.format_messages(
            context=context,
            query=query
        )

        response = self.llm.invoke(messages)
        return response.content.strip()

    def explain_relationship(self, entity1: str, entity2: str) -> str:
        """
        解釋兩個實體之間的關係

        Args:
            entity1: 實體1
            entity2: 實體2

        Returns:
            關係解釋
        """
        # 找路徑
        paths = self.kg_builder.find_path(entity1, entity2)

        if not paths:
            return f"{entity1} 和 {entity2} 在知識圖譜中沒有直接連接。"

        # 構建路徑說明
        path_descriptions = []
        for path in paths[:3]:  # 只取前3條路徑
            path_edges = []
            for i in range(len(path) - 1):
                edge_data = self.kg_builder.graph.get_edge_data(path[i], path[i+1])
                relation = edge_data.get('relation_type', 'related_to')
                path_edges.append(f"{path[i]} --[{relation}]--> {path[i+1]}")

            path_descriptions.append(" ; ".join(path_edges))

        # 生成解釋
        explain_prompt = ChatPromptTemplate.from_template("""
解釋兩個實體之間的關係。

實體1：{entity1}
實體2：{entity2}

知識圖譜中的連接路徑：
{paths}

請用自然語言解釋這兩個實體之間的關係。

解釋：
""")

        messages = explain_prompt.format_messages(
            entity1=entity1,
            entity2=entity2,
            paths="\n".join(path_descriptions)
        )

        response = self.llm.invoke(messages)
        return response.content.strip()


def main():
    """示例程式"""
    print("=" * 80)
    print("Graph RAG 系統示範")
    print("=" * 80)
    print()

    # 準備文檔
    documents = [
        """
        OpenAI 是一家人工智能研究公司，總部位於舊金山。OpenAI 開發了 GPT 系列模型，
        包括 GPT-3 和 GPT-4。Sam Altman 是 OpenAI 的 CEO。OpenAI 與微軟建立了戰略合作關係。
        """,
        """
        GPT（Generative Pre-trained Transformer）是一種基於 Transformer 架構的語言模型。
        GPT-3 有 1750 億個參數，展現了強大的少樣本學習能力。GPT-4 是 GPT-3 的後續版本，
        性能更強。GPT 模型使用自回歸方式生成文本。
        """,
        """
        Transformer 架構由 Google 在 2017 年提出，是現代 NLP 的基礎。Transformer 使用
        注意力機制來處理序列數據。BERT 和 GPT 都基於 Transformer 架構。Vaswani 等人
        在論文「Attention is All You Need」中提出了 Transformer。
        """,
        """
        微軟是一家科技公司，總部位於華盛頓州雷德蒙德。微軟投資了 OpenAI，並將 GPT 模型
        整合到其產品中，如 Microsoft 365 Copilot。Satya Nadella 是微軟的 CEO。
        微軟也開發了 Azure OpenAI Service。
        """,
        """
        BERT 是 Google 開發的預訓練語言模型，使用 Transformer 的編碼器部分。BERT 通過
        雙向訓練來學習語言表示。BERT 在多個 NLP 任務上都取得了顯著的性能提升。
        """
    ]

    # 初始化系統
    print("初始化 Graph RAG 系統...")
    graph_rag = GraphRAGSystem()

    # 構建知識圖譜
    print("\n構建知識圖譜...")
    graph_rag.build_knowledge_graph(documents)
    print()

    # 可視化
    print("生成知識圖譜視覺化...")
    graph_rag.kg_builder.visualize(output_path="knowledge_graph_demo.png")
    print()

    # 測試查詢
    test_queries = [
        "OpenAI 和 GPT 有什麼關係？",
        "Transformer 架構與 BERT 和 GPT 的關係是什麼？",
        "微軟和 OpenAI 之間有什麼合作？",
    ]

    for i, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"測試 {i}: {query}")
        print("=" * 80)

        result = graph_rag.multi_hop_query(query, max_hops=2, verbose=True)

        print(f"\n答案:\n{result.answer}\n")

    # 測試關係解釋
    print("=" * 80)
    print("關係解釋測試")
    print("=" * 80)

    explanation = graph_rag.explain_relationship("OpenAI", "Transformer")
    print(f"\nOpenAI 和 Transformer 的關係:\n{explanation}\n")

    # 保存圖譜
    graph_rag.kg_builder.save_graph("demo_knowledge_graph.pkl")

    print("=" * 80)
    print("示範完成！")
    print("=" * 80)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("錯誤: 請設置 OPENAI_API_KEY 環境變數")
    else:
        main()
