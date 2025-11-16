#!/usr/bin/env python3
"""
測試 AI 入門範例代碼
"""

import numpy as np
import sys

print("=" * 60)
print("測試 AI 基礎範例代碼")
print("=" * 60)

# 測試 1: 簡單的專家系統
print("\n測試 1: 專家系統")
print("-" * 60)

class MedicalExpertSystem:
    def __init__(self):
        self.knowledge_base = {
            ('發燒', '咳嗽', '喉嚨痛'): '感冒',
            ('發燒', '頭痛', '肌肉痠痛'): '流感',
            ('腹痛', '腹瀉', '噁心'): '腸胃炎',
            ('頭痛', '噁心', '畏光'): '偏頭痛'
        }

    def diagnose(self, symptoms):
        symptoms_tuple = tuple(sorted(symptoms))
        if symptoms_tuple in self.knowledge_base:
            return self.knowledge_base[symptoms_tuple]

        possible_diseases = []
        for known_symptoms, disease in self.knowledge_base.items():
            match_count = len(set(symptoms) & set(known_symptoms))
            if match_count >= 2:
                possible_diseases.append((disease, match_count))

        if possible_diseases:
            possible_diseases.sort(key=lambda x: x[1], reverse=True)
            return f"可能是 {possible_diseases[0][0]} (匹配度: {possible_diseases[0][1]}/3)"

        return "無法診斷，建議就醫"

expert_system = MedicalExpertSystem()
symptoms = ['發燒', '咳嗽', '喉嚨痛']
diagnosis = expert_system.diagnose(symptoms)
print(f"症狀: {symptoms}")
print(f"診斷結果: {diagnosis}")
assert '感冒' in diagnosis, "專家系統測試失敗"
print("✓ 專家系統測試通過")

# 測試 2: NumPy 基礎操作
print("\n測試 2: NumPy 基礎操作")
print("-" * 60)

arr = np.array([1, 2, 3, 4, 5])
print(f"數組: {arr}")
print(f"形狀: {arr.shape}")
print(f"平均值: {arr.mean()}")
print(f"總和: {arr.sum()}")

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n矩陣:\n{matrix}")
print(f"形狀: {matrix.shape}")
print(f"轉置:\n{matrix.T}")

assert arr.mean() == 3.0, "NumPy 測試失敗"
assert matrix.shape == (2, 3), "矩陣形狀測試失敗"
print("✓ NumPy 基礎測試通過")

# 測試 3: 簡單的線性回歸
print("\n測試 3: 線性回歸")
print("-" * 60)

class SimpleLinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, epochs=100):
        n = len(X)
        self.weight = 0
        self.bias = 0

        for _ in range(epochs):
            y_pred = self.weight * X + self.bias
            dw = -(2/n) * np.sum(X * (y - y_pred))
            db = -(2/n) * np.sum(y - y_pred)
            self.weight -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return self.weight * X + self.bias

# 生成測試數據: y = 2x + 1
np.random.seed(42)
X_train = np.array([1, 2, 3, 4, 5])
y_train = 2 * X_train + 1 + np.random.randn(5) * 0.1

model = SimpleLinearRegression()
model.fit(X_train, y_train, learning_rate=0.01, epochs=1000)

print(f"學習到的權重: {model.weight:.2f}")
print(f"學習到的偏置: {model.bias:.2f}")
print(f"預測 X=6: y={model.predict(6):.2f}")

assert 1.8 < model.weight < 2.2, "線性回歸權重測試失敗"
assert 0.5 < model.bias < 1.5, "線性回歸偏置測試失敗"
print("✓ 線性回歸測試通過")

# 測試 4: Sigmoid 函數
print("\n測試 4: Sigmoid 激活函數")
print("-" * 60)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

test_values = np.array([-2, -1, 0, 1, 2])
results = sigmoid(test_values)
print(f"輸入: {test_values}")
print(f"Sigmoid 輸出: {results}")

assert sigmoid(0) == 0.5, "Sigmoid(0) 應該等於 0.5"
assert sigmoid(100) > 0.99, "Sigmoid(大正數) 應該接近 1"
assert sigmoid(-100) < 0.01, "Sigmoid(大負數) 應該接近 0"
print("✓ Sigmoid 函數測試通過")

# 測試 5: K-Means 聚類
print("\n測試 5: K-Means 聚類")
print("-" * 60)

class SimpleKMeans:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(self, X, max_iters=10):
        # 隨機初始化
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(max_iters):
            # 分配到最近的聚類中心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # 更新聚類中心
            new_centroids = np.array([X[labels == i].mean(axis=0)
                                     for i in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return labels

# 生成兩個聚類的數據
np.random.seed(42)
cluster1 = np.random.randn(20, 2) + np.array([2, 2])
cluster2 = np.random.randn(20, 2) + np.array([-2, -2])
X_cluster = np.vstack([cluster1, cluster2])

kmeans = SimpleKMeans(n_clusters=2)
labels = kmeans.fit(X_cluster)

print(f"聚類中心:\n{kmeans.centroids}")
print(f"前 5 個樣本的標籤: {labels[:5]}")
print(f"後 5 個樣本的標籤: {labels[-5:]}")

# 驗證聚類效果（簡單檢查）
unique_labels = np.unique(labels)
assert len(unique_labels) == 2, "應該有 2 個聚類"
print("✓ K-Means 聚類測試通過")

# 測試 6: 路徑規劃（簡化版）
print("\n測試 6: 簡單路徑規劃")
print("-" * 60)

def simple_path_planning(grid, start, goal):
    """簡單的廣度優先搜索路徑規劃"""
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (row, col), path = queue.popleft()

        if (row, col) == goal:
            return path

        # 探索四個方向
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc

            if (0 <= new_row < rows and
                0 <= new_col < cols and
                (new_row, new_col) not in visited and
                grid[new_row][new_col] == 0):

                visited.add((new_row, new_col))
                queue.append(((new_row, new_col), path + [(new_row, new_col)]))

    return None

# 測試網格
grid = [
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

path = simple_path_planning(grid, (0, 0), (3, 3))
print(f"起點: (0, 0)")
print(f"終點: (3, 3)")
print(f"路徑: {path}")
print(f"路徑長度: {len(path) if path else 0}")

assert path is not None, "應該能找到路徑"
assert path[0] == (0, 0), "路徑應該從起點開始"
assert path[-1] == (3, 3), "路徑應該在終點結束"
print("✓ 路徑規劃測試通過")

# 總結
print("\n" + "=" * 60)
print("所有測試通過! ✓")
print("=" * 60)
print("\n測試涵蓋:")
print("  1. 專家系統 (知識表達與推理)")
print("  2. NumPy 基礎操作")
print("  3. 線性回歸 (機器學習)")
print("  4. Sigmoid 函數 (神經網路)")
print("  5. K-Means 聚類 (無監督學習)")
print("  6. 路徑規劃 (規劃與決策)")
print("\n所有核心概念都已驗證!")
