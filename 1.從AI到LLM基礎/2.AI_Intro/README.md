# AI 領域概述與實作介紹

AI（人工智慧）是一個相當廣泛的領域，它並不僅限於機器學習（Machine Learning）與深度學習（Deep Learning）。這兩者雖然是近年來相當炙手可熱的應用領域，但AI的概念本身涵蓋更多元的研究方向與技術範疇。以下列出一些其他常見的AI相關領域與子領域：

## 目錄
1. [知識表達與推理](#知識表達與推理)
2. [自然語言處理](#自然語言處理)
3. [電腦視覺](#電腦視覺)
4. [規劃與決策](#規劃與決策)
5. [強化學習](#強化學習)
6. [進化計算與啟發式優化](#進化計算與啟發式優化)
7. [機器人學](#機器人學)
8. [多智能體系統](#多智能體系統)
9. [符號AI與混合式AI](#符號ai與混合式ai)

---

## 知識表達與推理（Knowledge Representation and Reasoning）

### 概念介紹
該領域致力於如何以形式化的方式將知識以結構化方式儲存，並利用邏輯推理、規則系統、語義網路等工具，使機器能理解、操作與推論人類已知的知識。此領域包含專家系統（Expert Systems）、描述邏輯、貝氏網路等方法。

### 實際應用案例
- **醫療診斷系統**：基於症狀和醫學知識庫進行疾病診斷
- **法律諮詢系統**：根據法律條文和案例進行法律推理
- **智能客服**：基於知識圖譜回答用戶問題

### Python 實作範例：簡單的專家系統
```python
# 簡單的醫療診斷專家系統範例
class MedicalExpertSystem:
    def __init__(self):
        # 知識庫：症狀 -> 可能疾病
        self.knowledge_base = {
            ('發燒', '咳嗽', '喉嚨痛'): '感冒',
            ('發燒', '頭痛', '肌肉痠痛'): '流感',
            ('腹痛', '腹瀉', '噁心'): '腸胃炎',
            ('頭痛', '噁心', '畏光'): '偏頭痛'
        }

    def diagnose(self, symptoms):
        """根據症狀進行診斷推理"""
        symptoms_tuple = tuple(sorted(symptoms))

        # 精確匹配
        if symptoms_tuple in self.knowledge_base:
            return self.knowledge_base[symptoms_tuple]

        # 模糊匹配（至少匹配兩個症狀）
        possible_diseases = []
        for known_symptoms, disease in self.knowledge_base.items():
            match_count = len(set(symptoms) & set(known_symptoms))
            if match_count >= 2:
                possible_diseases.append((disease, match_count))

        if possible_diseases:
            # 返回匹配度最高的疾病
            possible_diseases.sort(key=lambda x: x[1], reverse=True)
            return f"可能是 {possible_diseases[0][0]} (匹配度: {possible_diseases[0][1]}/3)"

        return "無法診斷，建議就醫"

# 使用範例
expert_system = MedicalExpertSystem()
symptoms = ['發燒', '咳嗽', '喉嚨痛']
diagnosis = expert_system.diagnose(symptoms)
print(f"症狀: {symptoms}")
print(f"診斷結果: {diagnosis}")
```

---

## 自然語言處理（Natural Language Processing, NLP）

### 概念介紹
透過演算法與模型，讓電腦能理解、生成及分析人類的自然語言，包括語音辨識、語言理解、語言生成、機器翻譯、情感分析、對話系統等。

### 實際應用案例
- **聊天機器人**：ChatGPT、Claude、Gemini 等對話系統
- **機器翻譯**：Google 翻譯、DeepL
- **情感分析**：社交媒體輿情分析、產品評論分析
- **文本摘要**：新聞摘要、文檔總結

### Python 實作範例：簡單的情感分析
```python
# 簡單的中文情感分析範例
class SimpleSentimentAnalyzer:
    def __init__(self):
        # 情感詞典
        self.positive_words = ['好', '棒', '優秀', '喜歡', '開心', '滿意', '推薦', '完美', '出色', '讚']
        self.negative_words = ['壞', '差', '糟糕', '討厭', '失望', '不滿', '爛', '難過', '遺憾', '後悔']

    def analyze(self, text):
        """分析文本情感"""
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)

        score = positive_count - negative_count

        if score > 0:
            return '正面', score
        elif score < 0:
            return '負面', abs(score)
        else:
            return '中性', 0

# 使用範例
analyzer = SimpleSentimentAnalyzer()

texts = [
    "這個產品真的很棒，我非常滿意！",
    "品質太差了，非常失望",
    "還可以，沒什麼特別的"
]

for text in texts:
    sentiment, score = analyzer.analyze(text)
    print(f"文本: {text}")
    print(f"情感: {sentiment} (強度: {score})\n")
```

### 進階：使用 transformers 函式庫
```python
# 需要先安裝: pip install transformers torch

from transformers import pipeline

# 使用預訓練模型進行情感分析
sentiment_pipeline = pipeline("sentiment-analysis",
                              model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "I love this product! It's amazing!",
    "This is terrible, I'm very disappointed.",
    "It's okay, nothing special."
]

results = sentiment_pipeline(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}\n")
```

---

## 電腦視覺（Computer Vision）

### 概念介紹
讓機器可透過影像與視覺資訊理解週遭環境，包括影像辨識、物件偵測、影像分割、人臉識別、動作偵測等，從而能辨別、分析和理解影像或視訊數據。

### 實際應用案例
- **人臉識別**：手機解鎖、門禁系統
- **自動駕駛**：道路標誌識別、行人偵測
- **醫療影像分析**：X光、CT掃描分析
- **工業檢測**：產品瑕疵檢測

### Python 實作範例：使用 OpenCV 進行圖像處理
```python
# 需要先安裝: pip install opencv-python numpy

import cv2
import numpy as np

class SimpleImageProcessor:
    @staticmethod
    def edge_detection(image_path):
        """邊緣檢測"""
        # 讀取圖像
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 使用 Canny 邊緣檢測
        edges = cv2.Canny(img, 100, 200)

        return edges

    @staticmethod
    def face_detection(image_path):
        """人臉檢測"""
        # 讀取圖像
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 載入人臉檢測分類器
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 檢測人臉
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # 在圖像上繪製矩形框
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img, len(faces)

    @staticmethod
    def object_detection_color(image_path, lower_color, upper_color):
        """基於顏色的物體檢測"""
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 創建顏色遮罩
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # 找到輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 在圖像上繪製輪廓
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        return img, len(contours)

# 使用範例
processor = SimpleImageProcessor()

# 邊緣檢測
# edges = processor.edge_detection('sample.jpg')
# cv2.imwrite('edges.jpg', edges)

# 人臉檢測
# result, face_count = processor.face_detection('people.jpg')
# print(f"檢測到 {face_count} 張人臉")
# cv2.imwrite('faces_detected.jpg', result)

# 顏色物體檢測（檢測紅色物體）
# lower_red = np.array([0, 120, 70])
# upper_red = np.array([10, 255, 255])
# result, obj_count = processor.object_detection_color('objects.jpg', lower_red, upper_red)
# print(f"檢測到 {obj_count} 個紅色物體")
```

---

## 規劃與決策（Planning and Decision Making）

### 概念介紹
該領域研究如何讓AI系統在特定目標和限制條件下，自動制定出可行且最優或近似最優的行動策略。例如在機器人或自動化系統中，如何由初始狀態規劃出完成任務的行動序列。

### 實際應用案例
- **路徑規劃**：導航系統、物流配送路線優化
- **任務調度**：生產排程、人員排班
- **遊戲AI**：策略遊戲的決策樹

### Python 實作範例：A* 路徑規劃算法
```python
import heapq
from typing import List, Tuple, Set

class AStarPathPlanner:
    def __init__(self, grid: List[List[int]]):
        """
        初始化路徑規劃器
        grid: 二維網格，0表示可通行，1表示障礙物
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """曼哈頓距離啟發式函數"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """獲取相鄰可通行的位置"""
        row, col = pos
        neighbors = []

        # 上下左右四個方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 檢查是否在邊界內且可通行
            if (0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row][new_col] == 0):
                neighbors.append((new_row, new_col))

        return neighbors

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """使用 A* 算法尋找最短路徑"""
        # 優先隊列：(f_score, 位置)
        open_set = [(0, start)]

        # 記錄每個位置的來源
        came_from = {}

        # g_score: 從起點到當前位置的實際成本
        g_score = {start: 0}

        # f_score: g_score + 啟發式估計
        f_score = {start: self.heuristic(start, goal)}

        # 已訪問的位置
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            # 取出 f_score 最小的位置
            current_f, current = heapq.heappop(open_set)

            # 如果已訪問過，跳過
            if current in visited:
                continue

            visited.add(current)

            # 到達目標
            if current == goal:
                return self.reconstruct_path(came_from, current)

            # 探索相鄰位置
            for neighbor in self.get_neighbors(current):
                if neighbor in visited:
                    continue

                # 計算新的 g_score
                tentative_g = g_score[current] + 1

                # 如果找到更好的路徑
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # 沒有找到路徑

    def reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路徑"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def visualize_path(self, path: List[Tuple[int, int]]):
        """可視化路徑"""
        # 創建網格副本
        visual_grid = [row[:] for row in self.grid]

        # 標記路徑
        for row, col in path:
            if visual_grid[row][col] == 0:
                visual_grid[row][col] = 2  # 2 表示路徑

        # 打印網格
        symbols = {0: '·', 1: '█', 2: '*'}
        for row in visual_grid:
            print(' '.join(symbols[cell] for cell in row))

# 使用範例
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

planner = AStarPathPlanner(grid)
start = (0, 0)
goal = (4, 4)

path = planner.find_path(start, goal)

if path:
    print(f"找到路徑，長度: {len(path)}")
    print(f"路徑: {path}")
    print("\n路徑可視化:")
    planner.visualize_path(path)
else:
    print("無法找到路徑")
```

---

## 強化學習（Reinforcement Learning）

### 概念介紹
雖屬於機器學習分支，但思想和傳統監督/非監督學習略有不同。強化學習中，智能體透過在環境中不斷試誤，以回饋（獎勵或懲罰）指引行為策略的改進。此領域常用於遊戲AI、機器人控制、自動駕駛、資源分配等。

### 實際應用案例
- **遊戲AI**：AlphaGo、OpenAI Five（Dota 2）
- **機器人控制**：機械臂操作、無人機飛行
- **推薦系統**：個性化內容推薦
- **金融交易**：量化交易策略

### Python 實作範例：Q-Learning 玩簡單遊戲
```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Q-Learning 智能體
        n_states: 狀態數量
        n_actions: 動作數量
        learning_rate: 學習率
        discount_factor: 折扣因子
        epsilon: 探索率
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        # Q表：狀態-動作價值函數
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """選擇動作（ε-greedy策略）"""
        if random.random() < self.epsilon:
            # 探索：隨機選擇
            return random.randint(0, self.n_actions - 1)
        else:
            # 利用：選擇最優動作
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        """更新Q表"""
        # Q-Learning更新公式
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

class SimpleGridWorld:
    """簡單的網格世界環境"""
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.goal = (size - 1, size - 1)
        self.reset()

    def reset(self):
        """重置環境"""
        self.position = (0, 0)
        return self.position_to_state(self.position)

    def position_to_state(self, position):
        """將位置轉換為狀態編號"""
        return position[0] * self.size + position[1]

    def state_to_position(self, state):
        """將狀態編號轉換為位置"""
        return (state // self.size, state % self.size)

    def step(self, action):
        """執行動作"""
        row, col = self.position

        # 動作：0=上, 1=下, 2=左, 3=右
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            row = min(self.size - 1, row + 1)
        elif action == 2:
            col = max(0, col - 1)
        elif action == 3:
            col = min(self.size - 1, col + 1)

        self.position = (row, col)
        state = self.position_to_state(self.position)

        # 獎勵
        if self.position == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1  # 每步小懲罰，鼓勵快速到達目標
            done = False

        return state, reward, done

# 訓練範例
env = SimpleGridWorld(size=5)
agent = QLearningAgent(env.n_states, env.n_actions)

# 訓練
n_episodes = 1000
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):  # 最多100步
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}")

# 測試訓練好的智能體
print("\n測試訓練好的智能體:")
agent.epsilon = 0  # 關閉探索
state = env.reset()
path = [env.state_to_position(state)]

for step in range(20):
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    path.append(env.state_to_position(next_state))
    state = next_state

    if done:
        break

print(f"路徑: {path}")
print(f"步數: {len(path) - 1}")
```

---

## 進化計算與啟發式優化（Evolutionary Computation & Heuristic Methods）

### 概念介紹
利用生物進化、遺傳選擇、突變與交配的概念來搜尋複雜問題的近似解答。典型方法包括遺傳演算法、粒子群優化、螞蟻演算法、模擬退火等。

### 實際應用案例
- **工程優化**：結構設計、電路設計
- **排程問題**：生產排程、課表安排
- **參數調優**：機器學習超參數優化
- **組合優化**：旅行商問題（TSP）

### Python 實作範例：遺傳演算法求解背包問題
```python
import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def create_individual(self, n_items):
        """創建個體（隨機二進制串）"""
        return [random.randint(0, 1) for _ in range(n_items)]

    def create_population(self, n_items):
        """創建初始種群"""
        return [self.create_individual(n_items) for _ in range(self.population_size)]

    def fitness(self, individual, weights, values, max_weight):
        """計算適應度（背包問題）"""
        total_weight = sum(w * g for w, g in zip(weights, individual))
        total_value = sum(v * g for v, g in zip(values, individual))

        # 如果超重，適應度為0
        if total_weight > max_weight:
            return 0

        return total_value

    def selection(self, population, fitnesses):
        """輪盤賭選擇"""
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0

        for individual, fitness in zip(population, fitnesses):
            current += fitness
            if current > pick:
                return individual

        return population[-1]

    def crossover(self, parent1, parent2):
        """單點交叉"""
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutate(self, individual):
        """變異"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]  # 翻轉位元
        return individual

    def evolve(self, weights, values, max_weight, n_generations=100):
        """進化過程"""
        n_items = len(weights)
        population = self.create_population(n_items)

        best_individual = None
        best_fitness = 0

        for generation in range(n_generations):
            # 計算適應度
            fitnesses = [self.fitness(ind, weights, values, max_weight)
                        for ind in population]

            # 記錄最佳個體
            max_fitness = max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = population[fitnesses.index(max_fitness)][:]

            # 創建新種群
            new_population = []

            while len(new_population) < self.population_size:
                # 選擇
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)

                # 交叉
                child1, child2 = self.crossover(parent1, parent2)

                # 變異
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            if (generation + 1) % 20 == 0:
                print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        return best_individual, best_fitness

# 使用範例：背包問題
weights = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
values = [20, 30, 66, 40, 60, 10, 30, 10, 20, 15]
max_weight = 200

print("背包問題:")
print(f"物品重量: {weights}")
print(f"物品價值: {values}")
print(f"背包容量: {max_weight}\n")

ga = GeneticAlgorithm(population_size=50, mutation_rate=0.05, crossover_rate=0.8)
best_solution, best_value = ga.evolve(weights, values, max_weight, n_generations=100)

selected_items = [i for i, gene in enumerate(best_solution) if gene == 1]
total_weight = sum(weights[i] for i in selected_items)
total_value = sum(values[i] for i in selected_items)

print(f"\n最佳解決方案:")
print(f"選擇的物品索引: {selected_items}")
print(f"總重量: {total_weight}")
print(f"總價值: {total_value}")
```

---

## 機器人學（Robotics）

### 概念介紹
結合感測技術、控制理論、計算機視覺與AI規劃，使機器人能夠感知環境、自主決策與執行任務。自主導航、工業自動化、服務型機器人均屬其範疇。

### 實際應用案例
- **工業機器人**：汽車製造、電子組裝
- **服務機器人**：掃地機器人、送餐機器人
- **醫療機器人**：手術輔助、復健治療
- **探索機器人**：火星探測車、深海探測

### Python 實作範例：機器人運動學模擬
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleRobot:
    """簡單的二維移動機器人"""
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x  # x 位置
        self.y = y  # y 位置
        self.theta = theta  # 方向角（弧度）
        self.path = [(x, y)]

    def move(self, velocity, angular_velocity, dt=0.1):
        """運動學模型：差動驅動"""
        # 更新位置和方向
        self.x += velocity * np.cos(self.theta) * dt
        self.y += velocity * np.sin(self.theta) * dt
        self.theta += angular_velocity * dt

        # 記錄路徑
        self.path.append((self.x, self.y))

    def move_to_goal(self, goal_x, goal_y, max_steps=100):
        """移動到目標位置"""
        for _ in range(max_steps):
            # 計算到目標的距離和角度
            dx = goal_x - self.x
            dy = goal_y - self.y
            distance = np.sqrt(dx**2 + dy**2)

            # 如果已到達目標
            if distance < 0.1:
                print(f"到達目標! 位置: ({self.x:.2f}, {self.y:.2f})")
                return True

            # 計算目標角度
            goal_theta = np.arctan2(dy, dx)

            # 計算角度差
            angle_diff = goal_theta - self.theta
            # 正規化到 [-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # 控制策略
            velocity = min(distance, 1.0)  # 速度與距離成正比，但有上限
            angular_velocity = 2.0 * angle_diff  # 角速度與角度差成正比

            # 執行移動
            self.move(velocity, angular_velocity)

        print(f"未能在{max_steps}步內到達目標")
        return False

    def plot_path(self):
        """繪製路徑"""
        path_array = np.array(self.path)
        plt.figure(figsize=(10, 10))
        plt.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Robot Path')
        plt.plot(path_array[0, 0], path_array[0, 1], 'go', markersize=10, label='Start')
        plt.plot(path_array[-1, 0], path_array[-1, 1], 'ro', markersize=10, label='End')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Navigation Path')
        plt.legend()
        plt.show()

# 使用範例
robot = SimpleRobot(x=0.0, y=0.0, theta=0.0)

# 移動到多個目標點
goals = [(5, 5), (10, 3), (8, 8), (2, 10)]

for goal in goals:
    print(f"\n移動到目標: {goal}")
    robot.move_to_goal(goal[0], goal[1])

# 繪製路徑
robot.plot_path()
```

---

## 多智能體系統（Multi-agent Systems）

### 概念介紹
研究多個智能體（agents）在共享環境中彼此交互、合作、競爭與溝通的行為。此領域的研究常涵蓋協同決策、資源分配、談判與博弈理論等。

### 實際應用案例
- **分散式系統**：區塊鏈、P2P網路
- **智能交通**：多車協同、交通流優化
- **協同機器人**：多機器人倉儲系統
- **市場模擬**：經濟系統模擬

### Python 實作範例：多智能體協同搜索
```python
import random
import matplotlib.pyplot as plt
import numpy as np

class SearchAgent:
    """搜索智能體"""
    def __init__(self, agent_id, x, y, communication_range=5.0):
        self.id = agent_id
        self.x = x
        self.y = y
        self.communication_range = communication_range
        self.path = [(x, y)]
        self.found_target = False

    def distance_to(self, x, y):
        """計算到某點的距離"""
        return np.sqrt((self.x - x)**2 + (self.y - y)**2)

    def move_random(self, step_size=1.0, bounds=(0, 20)):
        """隨機移動"""
        angle = random.uniform(0, 2 * np.pi)
        new_x = self.x + step_size * np.cos(angle)
        new_y = self.y + step_size * np.sin(angle)

        # 確保在邊界內
        self.x = np.clip(new_x, bounds[0], bounds[1])
        self.y = np.clip(new_y, bounds[0], bounds[1])
        self.path.append((self.x, self.y))

    def move_towards(self, target_x, target_y, step_size=1.0):
        """朝目標移動"""
        angle = np.arctan2(target_y - self.y, target_x - self.x)
        self.x += step_size * np.cos(angle)
        self.y += step_size * np.sin(angle)
        self.path.append((self.x, self.y))

class MultiAgentSystem:
    """多智能體系統"""
    def __init__(self, n_agents, target_x, target_y, area_size=20):
        self.area_size = area_size
        self.target = (target_x, target_y)

        # 創建智能體
        self.agents = []
        for i in range(n_agents):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            self.agents.append(SearchAgent(i, x, y))

    def communicate(self):
        """智能體之間通信"""
        # 檢查是否有智能體發現目標
        for agent in self.agents:
            if agent.distance_to(*self.target) < 1.0 and not agent.found_target:
                agent.found_target = True
                print(f"智能體 {agent.id} 發現目標!")

                # 通知通信範圍內的其他智能體
                for other in self.agents:
                    if agent.id != other.id:
                        dist = agent.distance_to(other.x, other.y)
                        if dist < agent.communication_range:
                            other.found_target = True

    def step(self):
        """執行一步模擬"""
        for agent in self.agents:
            if agent.found_target:
                # 如果已知目標位置，朝目標移動
                agent.move_towards(*self.target)
            else:
                # 否則隨機搜索
                agent.move_random()

        # 智能體之間通信
        self.communicate()

    def simulate(self, max_steps=100):
        """運行模擬"""
        for step in range(max_steps):
            self.step()

            # 檢查是否所有智能體都到達目標
            all_arrived = all(
                agent.distance_to(*self.target) < 1.0
                for agent in self.agents
            )

            if all_arrived:
                print(f"所有智能體在第 {step + 1} 步到達目標!")
                return step + 1

        print(f"模擬結束，部分智能體未到達目標")
        return max_steps

    def visualize(self):
        """可視化智能體路徑"""
        plt.figure(figsize=(12, 12))

        # 繪製目標
        plt.plot(self.target[0], self.target[1], 'r*', markersize=20, label='Target')

        # 繪製每個智能體的路徑
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.agents)))
        for agent, color in zip(self.agents, colors):
            path = np.array(agent.path)
            plt.plot(path[:, 0], path[:, 1], '-', color=color, alpha=0.6, linewidth=1.5)
            plt.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=8, label=f'Agent {agent.id} Start')
            plt.plot(path[-1, 0], path[-1, 1], 's', color=color, markersize=8)

        plt.xlim(0, self.area_size)
        plt.ylim(0, self.area_size)
        plt.grid(True, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Multi-Agent Collaborative Search')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# 使用範例
system = MultiAgentSystem(n_agents=5, target_x=15, target_y=15)
steps = system.simulate(max_steps=150)
system.visualize()
```

---

## 符號AI與混合式AI（Symbolic AI & Neuro-Symbolic AI）

### 概念介紹
符號AI重視以邏輯、規則、符號的方式來詮釋與處理問題。近來也有將符號AI與深度學習結合的混合式方法，試圖兼顧可解釋性與強大學習能力。

### 實際應用案例
- **知識推理**：醫療診斷、法律推理
- **自然語言理解**：語義解析、問答系統
- **可解釋AI**：需要解釋決策過程的場景
- **混合系統**：結合規則和學習的系統

### Python 實作範例：簡單的符號推理系統
```python
class SymbolicReasoner:
    """符號推理系統"""
    def __init__(self):
        self.facts = set()  # 事實庫
        self.rules = []     # 規則庫

    def add_fact(self, fact):
        """添加事實"""
        self.facts.add(fact)
        print(f"添加事實: {fact}")

    def add_rule(self, conditions, conclusion):
        """
        添加推理規則
        conditions: 前提條件列表
        conclusion: 結論
        """
        self.rules.append((conditions, conclusion))
        print(f"添加規則: IF {' AND '.join(conditions)} THEN {conclusion}")

    def forward_chaining(self, max_iterations=10):
        """前向鏈推理"""
        print("\n開始前向鏈推理...")
        new_facts_found = True
        iteration = 0

        while new_facts_found and iteration < max_iterations:
            new_facts_found = False
            iteration += 1

            for conditions, conclusion in self.rules:
                # 檢查所有條件是否都滿足
                if all(cond in self.facts for cond in conditions):
                    # 如果結論還不在事實庫中
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        print(f"推導出新事實: {conclusion}")
                        new_facts_found = True

        print(f"推理完成，共進行 {iteration} 輪")

    def query(self, fact):
        """查詢事實是否成立"""
        return fact in self.facts

    def explain(self, fact):
        """解釋某個事實是如何推導出來的"""
        if fact not in self.facts:
            return f"{fact} 不在事實庫中"

        # 查找推導路徑
        for conditions, conclusion in self.rules:
            if conclusion == fact:
                return f"{fact} 由以下條件推導: {', '.join(conditions)}"

        return f"{fact} 是初始事實"

# 使用範例：動物分類推理系統
reasoner = SymbolicReasoner()

# 添加初始事實
reasoner.add_fact("tweety 有羽毛")
reasoner.add_fact("tweety 會飛")
reasoner.add_fact("tweety 會下蛋")

# 添加推理規則
reasoner.add_rule(["X 有羽毛", "X 會飛", "X 會下蛋"], "X 是鳥類")
reasoner.add_rule(["X 是鳥類", "X 會游泳"], "X 是水鳥")
reasoner.add_rule(["X 是鳥類", "X 不會飛"], "X 是企鵝或鴕鳥")

# 執行前向鏈推理
reasoner.forward_chaining()

# 查詢
print(f"\ntweety 是鳥類嗎? {reasoner.query('tweety 是鳥類')}")
print(f"tweety 是水鳥嗎? {reasoner.query('tweety 是水鳥')}")

# 解釋推理過程
print(f"\n{reasoner.explain('tweety 是鳥類')}")
```

### 混合式 AI 範例：結合規則和神經網路
```python
# 概念範例：神經符號混合系統

class NeuroSymbolicSystem:
    """神經符號混合系統示例"""
    def __init__(self):
        self.symbolic_rules = {}
        self.neural_predictions = {}

    def add_symbolic_rule(self, name, rule_func):
        """添加符號規則"""
        self.symbolic_rules[name] = rule_func

    def add_neural_model(self, name, model_func):
        """添加神經網路模型（簡化表示）"""
        self.neural_predictions[name] = model_func

    def hybrid_inference(self, input_data):
        """混合推理"""
        results = {}

        # 1. 使用神經網路進行初步預測
        print("步驟 1: 神經網路預測")
        for name, model in self.neural_predictions.items():
            prediction = model(input_data)
            results[f"neural_{name}"] = prediction
            print(f"  {name}: {prediction}")

        # 2. 使用符號規則進行驗證和修正
        print("\n步驟 2: 符號規則驗證")
        for name, rule in self.symbolic_rules.items():
            verified = rule(results)
            results[f"verified_{name}"] = verified
            print(f"  {name}: {verified}")

        return results

# 使用範例
system = NeuroSymbolicSystem()

# 添加神經網路模型（簡化表示）
system.add_neural_model(
    "age_prediction",
    lambda data: 25  # 簡化：實際應該是神經網路預測
)

system.add_neural_model(
    "income_prediction",
    lambda data: 50000
)

# 添加符號規則
def validate_credit_rule(results):
    """信用評估規則"""
    age = results.get("neural_age_prediction", 0)
    income = results.get("neural_income_prediction", 0)

    if age >= 18 and income >= 30000:
        return "合格"
    else:
        return "不合格"

system.add_symbolic_rule("credit_check", validate_credit_rule)

# 執行混合推理
input_data = {"features": [1, 2, 3]}  # 簡化的輸入
results = system.hybrid_inference(input_data)

print(f"\n最終結果: {results}")
```

---

## 總結

AI領域包含了大量多元的研究與應用方向，而機器學習與深度學習只是其中受到極大矚目的分支。藉由不同領域間的整合，AI的發展不斷推陳出新，也更加貼近人類智能的各種面向。

### 學習建議

1. **理論與實踐結合**：不要只學習理論，要動手實作
2. **從簡單開始**：先掌握基礎算法，再深入複雜模型
3. **關注應用場景**：理解不同技術適用的場景
4. **持續學習**：AI領域發展快速，保持學習熱情

### 進階學習資源

- **課程**：Stanford CS229、MIT 6.S191、fast.ai
- **書籍**：《Deep Learning》、《Reinforcement Learning: An Introduction》
- **實踐平台**：Kaggle、LeetCode、GitHub
- **論文**：arXiv.org、Papers with Code

### 下一步

完成本章學習後，建議：
1. 選擇一個感興趣的領域深入學習
2. 完成相關的實作項目
3. 閱讀該領域的經典論文
4. 參與開源項目或競賽