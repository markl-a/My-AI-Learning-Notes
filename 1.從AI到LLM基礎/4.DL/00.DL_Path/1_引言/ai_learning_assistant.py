#!/usr/bin/env python3
"""
AI 學習助手 - 深度學習引言章節
提供概念解釋、代碼生成和學習路徑規劃功能
"""

import argparse
import json
import sys
from typing import Dict, List, Optional

# 深度學習概念庫
CONCEPTS = {
    "機器學習": {
        "簡短定義": "一種讓計算機從數據中學習而無需明確編程的技術",
        "詳細解釋": """
機器學習是人工智能的一個分支，它使計算機系統能夠通過經驗自動改進。
與傳統編程不同（程序員編寫明確的規則），機器學習系統通過分析大量數據來學習模式。

關鍵特點：
1. 數據驅動：從數據中學習，而非硬編碼規則
2. 自動改進：隨著更多數據，性能會提升
3. 泛化能力：能處理未見過的新數據

生活中的例子：
- 垃圾郵件過濾器學習識別垃圾郵件
- 推薦系統學習你的偏好
- 語音助手學習理解你的聲音
        """,
        "代碼示例": """
# 簡單的線性回歸示例
import numpy as np
from sklearn.linear_model import LinearRegression

# 訓練數據：房屋面積 vs 價格
X = np.array([[50], [60], [70], [80], [90]])  # 面積（平方米）
y = np.array([150, 180, 210, 240, 270])        # 價格（萬元）

# 創建並訓練模型
model = LinearRegression()
model.fit(X, y)

# 預測新房屋的價格
new_house = np.array([[75]])
predicted_price = model.predict(new_house)
print(f"預測 75 平方米房屋價格: {predicted_price[0]:.2f} 萬元")
        """,
        "相關概念": ["深度學習", "監督學習", "無監督學習", "模型", "訓練"]
    },

    "深度學習": {
        "簡短定義": "使用多層神經網絡進行學習的機器學習方法",
        "詳細解釋": """
深度學習是機器學習的一個子集，使用具有多個處理層的神經網絡來學習數據的表示。
'深度' 指的是網絡的層數。

為什麼需要深度學習：
1. 自動特徵提取：無需手動設計特徵
2. 處理複雜數據：圖像、語音、文本
3. 端到端學習：從原始數據到最終結果

成功應用：
- 圖像識別（如人臉識別）
- 語音識別（Siri、Alexa）
- 自然語言處理（ChatGPT）
- 自動駕駛
        """,
        "代碼示例": """
# 使用 PyTorch 構建簡單的深度神經網絡
import torch
import torch.nn as nn

class SimpleDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定義多層網絡
        self.layer1 = nn.Linear(10, 64)   # 輸入層到隱藏層1
        self.layer2 = nn.Linear(64, 32)   # 隱藏層1到隱藏層2
        self.layer3 = nn.Linear(32, 1)    # 隱藏層2到輸出層
        self.relu = nn.ReLU()             # 激活函數

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 創建模型
model = SimpleDeepNet()
print(model)
        """,
        "相關概念": ["神經網絡", "反向傳播", "梯度下降", "激活函數"]
    },

    "模型": {
        "簡短定義": "一個參數化的函數，將輸入映射到輸出",
        "詳細解釋": """
在機器學習中，模型是一個數學函數，它接收輸入並產生輸出。
模型的行為由其參數決定。

模型的組成：
1. 輸入：模型接收的數據（如圖像像素）
2. 參數：可調整的變量（如權重和偏置）
3. 輸出：模型的預測結果

模型的類型：
- 線性模型：y = wx + b
- 神經網絡：多層非線性變換
- 決策樹：基於規則的分類
        """,
        "代碼示例": """
# 簡單的線性模型示例
import torch
import torch.nn as nn

# 定義一個線性模型 y = wx + b
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 1個輸入，1個輸出

    def forward(self, x):
        return self.linear(x)

# 創建模型
model = LinearModel()

# 查看模型參數
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# 使用模型進行預測
x = torch.tensor([[2.0]])
y = model(x)
print(f"輸入 {x.item()}, 輸出 {y.item():.4f}")
        """,
        "相關概念": ["參數", "訓練", "推理", "損失函數"]
    },

    "參數": {
        "簡短定義": "模型中可以通過訓練調整的變量",
        "詳細解釋": """
參數是模型的可學習組件，通過訓練過程自動調整。

主要類型：
1. 權重（Weights）：連接的強度
2. 偏置（Bias）：輸出的偏移量

參數 vs 超參數：
- 參數：通過訓練自動學習（如神經網絡的權重）
- 超參數：需要手動設置（如學習率、層數）

訓練過程：
1. 初始化參數（通常是隨機值）
2. 使用數據計算損失
3. 調整參數以減少損失
4. 重複直到收斂
        """,
        "代碼示例": """
# 查看和更新模型參數
import torch
import torch.nn as nn

# 創建一個簡單的線性層
layer = nn.Linear(3, 2)  # 3個輸入，2個輸出

print("初始參數：")
print(f"權重形狀: {layer.weight.shape}")
print(f"權重值:\\n{layer.weight.data}")
print(f"偏置值: {layer.bias.data}")

# 手動更新參數（實際訓練中由優化器自動完成）
with torch.no_grad():
    layer.weight.fill_(0.5)
    layer.bias.fill_(0.1)

print("\\n更新後參數：")
print(f"權重值:\\n{layer.weight.data}")
print(f"偏置值: {layer.bias.data}")
        """,
        "相關概念": ["訓練", "梯度", "優化器", "學習率"]
    },

    "訓練": {
        "簡短定義": "通過數據調整模型參數以優化性能的過程",
        "詳細解釋": """
訓練是機器學習的核心過程，目標是找到最佳的模型參數。

訓練步驟：
1. 前向傳播：輸入數據通過模型得到預測
2. 計算損失：比較預測和真實標籤的差異
3. 反向傳播：計算損失相對於參數的梯度
4. 更新參數：使用優化算法調整參數

訓練循環：
- Epoch：遍歷整個訓練集一次
- Batch：一次處理的樣本數量
- Iteration：更新參數一次

監控訓練：
- 訓練損失：模型在訓練數據上的表現
- 驗證損失：模型在驗證數據上的表現
- 避免過擬合：模型在訓練集表現好但驗證集差
        """,
        "代碼示例": """
# 完整的訓練循環示例
import torch
import torch.nn as nn
import torch.optim as optim

# 準備數據
X_train = torch.randn(100, 10)  # 100個樣本，10個特徵
y_train = torch.randn(100, 1)   # 100個標籤

# 定義模型
model = nn.Linear(10, 1)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練循環
num_epochs = 50
for epoch in range(num_epochs):
    # 前向傳播
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # 反向傳播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 計算梯度
    optimizer.step()       # 更新參數

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        """,
        "相關概念": ["損失函數", "優化器", "梯度下降", "反向傳播"]
    },

    "監督學習": {
        "簡短定義": "使用帶標籤的數據進行訓練的學習方法",
        "詳細解釋": """
監督學習是最常見的機器學習範式，使用輸入-輸出對進行訓練。

特點：
1. 需要標註數據：每個樣本都有對應的標籤
2. 學習映射：從輸入到輸出的函數
3. 可評估：有明確的正確答案

主要任務類型：
1. 分類：預測離散類別（如垃圾郵件檢測）
2. 回歸：預測連續值（如房價預測）

常見算法：
- 線性回歸、邏輯回歸
- 支持向量機（SVM）
- 決策樹、隨機森林
- 神經網絡
        """,
        "代碼示例": """
# 監督學習：分類任務示例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 加載鳶尾花數據集（經典分類數據集）
iris = load_iris()
X, y = iris.data, iris.target

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 訓練分類器
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 預測和評估
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"準確率: {accuracy:.2%}")

# 預測新樣本
new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = clf.predict(new_sample)
print(f"預測類別: {iris.target_names[prediction[0]]}")
        """,
        "相關概念": ["無監督學習", "分類", "回歸", "標籤"]
    },

    "無監督學習": {
        "簡短定義": "從無標籤數據中發現模式的學習方法",
        "詳細解釋": """
無監督學習處理沒有標籤的數據，目標是發現數據的內在結構。

主要任務：
1. 聚類：將相似數據分組（如客戶細分）
2. 降維：減少特徵數量（如PCA）
3. 異常檢測：發現異常數據點
4. 關聯規則：發現項目之間的關係

應用場景：
- 市場細分：根據行為分組客戶
- 推薦系統：發現用戶和商品的潛在關係
- 數據可視化：降維到2D/3D
- 異常檢測：信用卡欺詐檢測
        """,
        "代碼示例": """
# 無監督學習：聚類示例
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成模擬數據
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# 應用 K-Means 聚類
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# 可視化結果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200,
            edgecolors='black', label='聚類中心')
plt.title('K-Means 聚類結果')
plt.legend()
plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
print("聚類完成！結果已保存。")
        """,
        "相關概念": ["聚類", "降維", "K-Means", "PCA"]
    }
}

# 學習路徑規劃
LEARNING_PATHS = {
    "beginner": {
        "名稱": "初學者路徑",
        "時長": "4-6 小時",
        "步驟": [
            {
                "步驟": 1,
                "標題": "理解基本概念",
                "時長": "1 小時",
                "任務": [
                    "閱讀 index.ipynb 中的機器學習定義",
                    "理解機器學習與傳統編程的區別",
                    "學習關鍵術語：模型、參數、訓練"
                ],
                "檢查點": "能用自己的話解釋什麼是機器學習"
            },
            {
                "步驟": 2,
                "標題": "認識應用場景",
                "時長": "0.5 小時",
                "任務": [
                    "列舉日常生活中的機器學習應用",
                    "理解不同應用背後的機器學習類型",
                    "思考身邊還有哪些潛在的 ML 應用"
                ],
                "檢查點": "能識別並分類至少5個機器學習應用"
            },
            {
                "步驟": 3,
                "標題": "動手實踐基礎",
                "時長": "2 小時",
                "任務": [
                    "運行 01_ml_concepts_demo.ipynb",
                    "觀察參數變化對模型的影響",
                    "嘗試修改代碼並觀察結果"
                ],
                "檢查點": "成功運行並理解所有示例代碼"
            },
            {
                "步驟": 4,
                "標題": "學習機器學習類型",
                "時長": "1 小時",
                "任務": [
                    "理解監督學習和無監督學習的區別",
                    "學習分類和回歸任務",
                    "了解強化學習的基本概念"
                ],
                "檢查點": "能區分不同類型的機器學習問題"
            },
            {
                "步驟": 5,
                "標題": "鞏固與測試",
                "時長": "0.5-1 小時",
                "任務": [
                    "使用 quiz_generator.py 進行自測",
                    "完成練習題",
                    "總結學習筆記"
                ],
                "檢查點": "測驗正確率達到 80% 以上"
            }
        ],
        "下一步": "進入第 2 章學習必要的數學基礎"
    },

    "intermediate": {
        "名稱": "進階路徑",
        "時長": "2-3 小時",
        "說明": "適合有基本編程經驗的學習者",
        "步驟": [
            {
                "步驟": 1,
                "標題": "快速概覽",
                "時長": "0.5 小時",
                "任務": [
                    "瀏覽 index.ipynb，重點關注不熟悉的概念",
                    "建立機器學習的整體框架"
                ]
            },
            {
                "步驟": 2,
                "標題": "深入實踐",
                "時長": "1.5 小時",
                "任務": [
                    "完成所有實踐 notebook",
                    "嘗試修改參數和模型結構",
                    "實現自己的簡單模型"
                ]
            },
            {
                "步驟": 3,
                "標題": "擴展學習",
                "時長": "1 小時",
                "任務": [
                    "閱讀推薦的額外資源",
                    "探索 TensorFlow Playground",
                    "思考實際應用場景"
                ]
            }
        ],
        "下一步": "可選擇性複習數學基礎，或直接進入第 3 章實作"
    },

    "advanced": {
        "名稱": "快速通道",
        "時長": "1 小時",
        "說明": "適合有機器學習基礎的學習者",
        "步驟": [
            {
                "步驟": 1,
                "標題": "查缺補漏",
                "時長": "0.5 小時",
                "任務": [
                    "快速掃描章節內容",
                    "重點關注不熟悉的部分",
                    "複習關鍵術語"
                ]
            },
            {
                "步驟": 2,
                "標題": "實作驗證",
                "時長": "0.5 小時",
                "任務": [
                    "快速運行代碼示例",
                    "完成測驗檢驗理解",
                    "準備進入下一章節"
                ]
            }
        ],
        "下一步": "直接進入感興趣的章節（建議從第 3 章開始）"
    }
}

# 代碼模板
CODE_TEMPLATES = {
    "線性回歸": """
# 使用 PyTorch 實現簡單線性回歸
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成模擬數據
torch.manual_seed(42)
X = torch.randn(100, 1) * 10
y = 3 * X + 7 + torch.randn(100, 1) * 2  # y = 3x + 7 + noise

# 定義模型
model = nn.Linear(1, 1)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練
losses = []
for epoch in range(100):
    # 前向傳播
    predictions = model(X)
    loss = criterion(predictions, y)

    # 反向傳播和優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# 可視化結果
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='真實數據')
plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', label='擬合線')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('線性回歸結果')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('訓練損失')

plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150, bbox_inches='tight')
print(f"\\n學習到的參數: w={model.weight.item():.2f}, b={model.bias.item():.2f}")
print("(真實參數: w=3.00, b=7.00)")
    """,

    "分類": """
# 使用 PyTorch 實現簡單二元分類
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 生成數據
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 轉換為 PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

# 定義模型
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = Classifier()

# 定義損失函數和優化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 訓練
for epoch in range(200):
    # 前向傳播
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        # 計算準確率
        with torch.no_grad():
            test_pred = model(X_test)
            test_pred_class = (test_pred > 0.5).float()
            accuracy = (test_pred_class == y_test).float().mean()
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# 可視化決策邊界
plt.figure(figsize=(10, 5))

# 原始數據
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('原始數據')

# 決策邊界
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.detach().numpy().reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
plt.title('分類結果（決策邊界）')

plt.tight_layout()
plt.savefig('classification_result.png', dpi=150, bbox_inches='tight')
print("\\n分類器訓練完成！")
    """
}


def explain_concept(concept_name: str, detail_level: str = "medium") -> str:
    """
    解釋概念

    Args:
        concept_name: 概念名稱
        detail_level: 詳細程度 (brief/medium/detailed)
    """
    if concept_name not in CONCEPTS:
        # 提供相似概念建議
        available = ", ".join(CONCEPTS.keys())
        return f"❌ 概念 '{concept_name}' 未找到。\n\n可用概念：{available}"

    concept = CONCEPTS[concept_name]

    output = f"\n{'='*60}\n"
    output += f"📚 概念：{concept_name}\n"
    output += f"{'='*60}\n\n"

    if detail_level == "brief":
        output += f"💡 {concept['簡短定義']}\n"
    elif detail_level == "detailed":
        output += f"💡 簡短定義：\n{concept['簡短定義']}\n\n"
        output += f"📖 詳細解釋：{concept['詳細解釋']}\n"
        output += f"\n💻 代碼示例：\n{concept['代碼示例']}\n"
        output += f"\n🔗 相關概念：{', '.join(concept['相關概念'])}\n"
    else:  # medium
        output += f"💡 簡短定義：\n{concept['簡短定義']}\n\n"
        output += f"📖 詳細解釋：{concept['詳細解釋']}\n"
        output += f"\n🔗 相關概念：{', '.join(concept['相關概念'])}\n"

    return output


def generate_code(task: str) -> str:
    """生成代碼示例"""
    if task not in CODE_TEMPLATES:
        available = ", ".join(CODE_TEMPLATES.keys())
        return f"❌ 任務 '{task}' 的代碼模板未找到。\n\n可用任務：{available}"

    output = f"\n{'='*60}\n"
    output += f"💻 任務：{task}\n"
    output += f"{'='*60}\n\n"
    output += CODE_TEMPLATES[task]
    output += f"\n\n💡 提示：將此代碼保存為 .py 文件並運行，或在 Jupyter Notebook 中執行。\n"

    return output


def show_roadmap(level: str) -> str:
    """顯示學習路徑"""
    if level not in LEARNING_PATHS:
        available = ", ".join(LEARNING_PATHS.keys())
        return f"❌ 等級 '{level}' 未找到。\n\n可用等級：{available}"

    path = LEARNING_PATHS[level]

    output = f"\n{'='*60}\n"
    output += f"🗺️  學習路徑：{path['名稱']}\n"
    output += f"{'='*60}\n\n"
    output += f"⏱️  預估時長：{path['時長']}\n"

    if "說明" in path:
        output += f"📝 說明：{path['說明']}\n"

    output += f"\n📋 學習步驟：\n\n"

    for step in path['步驟']:
        output += f"  {step['步驟']}. {step['標題']} ({step['時長']})\n"
        output += f"  {'─' * 55}\n"
        if '任務' in step:
            for task in step['任務']:
                output += f"     • {task}\n"
        if '檢查點' in step:
            output += f"     ✓ 檢查點：{step['檢查點']}\n"
        output += f"\n"

    output += f"➡️  下一步：{path['下一步']}\n"

    return output


def list_all_concepts() -> str:
    """列出所有可用概念"""
    output = f"\n{'='*60}\n"
    output += f"📚 可用概念列表\n"
    output += f"{'='*60}\n\n"

    for i, (name, concept) in enumerate(CONCEPTS.items(), 1):
        output += f"{i}. {name}\n"
        output += f"   💡 {concept['簡短定義']}\n\n"

    output += f"\n使用方法：\n"
    output += f"  python ai_learning_assistant.py --mode explain --concept \"概念名稱\"\n"

    return output


def interactive_mode():
    """交互式模式"""
    print("\n" + "="*60)
    print("🤖 AI 學習助手 - 交互式模式")
    print("="*60)
    print("\n可用命令：")
    print("  explain <概念名稱>  - 解釋概念")
    print("  code <任務名稱>     - 生成代碼")
    print("  roadmap <等級>      - 查看學習路徑")
    print("  list                - 列出所有概念")
    print("  quit                - 退出")
    print()

    while True:
        try:
            cmd = input(">>> ").strip()

            if not cmd:
                continue

            if cmd.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再見！祝學習順利！")
                break

            parts = cmd.split(maxsplit=1)
            command = parts[0].lower()

            if command == "list":
                print(list_all_concepts())
            elif command == "explain" and len(parts) == 2:
                print(explain_concept(parts[1], "detailed"))
            elif command == "code" and len(parts) == 2:
                print(generate_code(parts[1]))
            elif command == "roadmap" and len(parts) == 2:
                print(show_roadmap(parts[1]))
            else:
                print("❌ 無效命令。輸入 'list' 查看幫助。")

        except KeyboardInterrupt:
            print("\n\n👋 再見！祝學習順利！")
            break
        except Exception as e:
            print(f"❌ 錯誤：{e}")


def main():
    parser = argparse.ArgumentParser(
        description="AI 學習助手 - 深度學習引言章節",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 解釋概念
  python ai_learning_assistant.py --mode explain --concept 機器學習

  # 生成代碼
  python ai_learning_assistant.py --mode code --task 線性回歸

  # 查看學習路徑
  python ai_learning_assistant.py --mode roadmap --level beginner

  # 列出所有概念
  python ai_learning_assistant.py --mode list

  # 交互式模式
  python ai_learning_assistant.py --interactive
        """
    )

    parser.add_argument('--mode', choices=['explain', 'code', 'roadmap', 'list'],
                        help='操作模式')
    parser.add_argument('--concept', help='要解釋的概念名稱')
    parser.add_argument('--detail', choices=['brief', 'medium', 'detailed'],
                        default='medium', help='詳細程度')
    parser.add_argument('--task', help='要生成代碼的任務')
    parser.add_argument('--level', choices=['beginner', 'intermediate', 'advanced'],
                        help='學習等級')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='啟動交互式模式')

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    if not args.mode:
        parser.print_help()
        return

    if args.mode == 'explain':
        if not args.concept:
            print("❌ 請指定概念名稱（--concept）")
            return
        print(explain_concept(args.concept, args.detail))

    elif args.mode == 'code':
        if not args.task:
            print("❌ 請指定任務名稱（--task）")
            return
        print(generate_code(args.task))

    elif args.mode == 'roadmap':
        if not args.level:
            print("❌ 請指定學習等級（--level）")
            return
        print(show_roadmap(args.level))

    elif args.mode == 'list':
        print(list_all_concepts())


if __name__ == "__main__":
    main()
