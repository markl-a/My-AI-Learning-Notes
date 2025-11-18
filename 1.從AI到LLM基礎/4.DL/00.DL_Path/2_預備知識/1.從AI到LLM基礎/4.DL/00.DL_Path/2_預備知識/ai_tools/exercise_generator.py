"""
AI 輔助練習生成器
================

根據學習主題和難度自動生成個性化練習題。
支持的主題：張量操作、線性代數、微積分、概率統計等。

使用方法：
    python exercise_generator.py --topic linear_algebra --difficulty medium --count 5

作者：AI Learning Community
版本：v1.0
"""

import argparse
import random
import json
from typing import List, Dict, Any
from datetime import datetime


class ExerciseGenerator:
    """練習題生成器"""

    def __init__(self, topic: str, difficulty: str):
        self.topic = topic
        self.difficulty = difficulty
        self.difficulty_levels = {
            'easy': 1,
            'medium': 2,
            'hard': 3,
            'expert': 4
        }

    def generate(self, count: int = 5) -> List[Dict[str, Any]]:
        """生成指定數量的練習題"""
        exercises = []

        generators = {
            'ndarray': self._generate_ndarray_exercises,
            'linear_algebra': self._generate_linear_algebra_exercises,
            'calculus': self._generate_calculus_exercises,
            'autograd': self._generate_autograd_exercises,
            'probability': self._generate_probability_exercises,
        }

        if self.topic in generators:
            exercises = generators[self.topic](count)
        else:
            raise ValueError(f"不支持的主題: {self.topic}")

        return exercises

    def _generate_ndarray_exercises(self, count: int) -> List[Dict[str, Any]]:
        """生成張量操作練習題"""
        exercises = []
        templates = {
            'easy': [
                {
                    'question': '創建一個形狀為 ({shape}) 的零張量，並將其轉換為形狀 ({reshape})。',
                    'solution': 'import torch\nx = torch.zeros({shape})\ny = x.reshape({reshape})',
                    'hints': ['使用 torch.zeros() 創建零張量', '使用 .reshape() 改變形狀'],
                    'concepts': ['張量創建', '形狀變換']
                },
                {
                    'question': '創建一個從 0 到 {n} 的整數張量，並計算其總和。',
                    'solution': 'import torch\nx = torch.arange({n})\ntotal = x.sum()',
                    'hints': ['使用 torch.arange()', '使用 .sum() 方法'],
                    'concepts': ['張量創建', '聚合操作']
                }
            ],
            'medium': [
                {
                    'question': '創建兩個形狀為 ({m}, {n}) 的隨機張量 A 和 B，執行按元素乘法和矩陣乘法，並比較結果的形狀。',
                    'solution': 'import torch\nA = torch.randn({m}, {n})\nB = torch.randn({m}, {n})\nelement_wise = A * B  # 形狀: ({m}, {n})\nmatrix_mult = torch.mm(A, B.T)  # 形狀: ({m}, {m})',
                    'hints': ['按元素乘法使用 *', '矩陣乘法使用 torch.mm()', '注意矩陣乘法的維度要求'],
                    'concepts': ['張量運算', '矩陣乘法', '廣播機制']
                },
                {
                    'question': '使用廣播機制，將形狀為 ({m}, 1) 的張量與形狀為 (1, {n}) 的張量相加，並解釋結果的形狀。',
                    'solution': 'import torch\nA = torch.randn({m}, 1)\nB = torch.randn(1, {n})\nC = A + B  # 形狀: ({m}, {n})',
                    'hints': ['理解廣播規則', '觀察輸出形狀的變化'],
                    'concepts': ['廣播機制', '張量形狀']
                }
            ],
            'hard': [
                {
                    'question': '實現一個函數，使用 PyTorch 張量操作計算批量數據的標準化（z-score normalization）。輸入形狀為 (batch_size, features)。',
                    'solution': 'import torch\n\ndef normalize(x):\n    mean = x.mean(dim=0, keepdim=True)\n    std = x.std(dim=0, keepdim=True)\n    return (x - mean) / (std + 1e-8)\n\n# 測試\ndata = torch.randn(100, 10)\nnormalized = normalize(data)',
                    'hints': ['使用 .mean() 和 .std()', '注意 keepdim 參數', '避免除以零'],
                    'concepts': ['數據標準化', '統計運算', '數值穩定性']
                }
            ]
        }

        level_templates = templates.get(self.difficulty, templates['easy'])

        for i in range(count):
            template = random.choice(level_templates)
            exercise = template.copy()

            # 填充隨機參數
            params = {
                'shape': f"({random.randint(2, 5)}, {random.randint(2, 5)})",
                'reshape': f"({random.randint(2, 10)}, -1)",
                'n': random.randint(10, 100),
                'm': random.randint(3, 6),
                'n': random.randint(3, 6)
            }

            exercise['question'] = exercise['question'].format(**params)
            exercise['solution'] = exercise['solution'].format(**params)
            exercise['id'] = f"{self.topic}_{self.difficulty}_{i+1}"
            exercise['difficulty'] = self.difficulty

            exercises.append(exercise)

        return exercises

    def _generate_linear_algebra_exercises(self, count: int) -> List[Dict[str, Any]]:
        """生成線性代數練習題"""
        exercises = []
        templates = {
            'easy': [
                {
                    'question': '計算向量 v = [{v}] 的 L2 範數（歐幾里得範數）。',
                    'solution': 'import torch\nv = torch.tensor([{v}], dtype=torch.float32)\nnorm = torch.norm(v)',
                    'hints': ['使用 torch.norm()', 'L2 範數是元素平方和的平方根'],
                    'concepts': ['向量範數', '向量運算']
                }
            ],
            'medium': [
                {
                    'question': '給定矩陣 A ({m}×{n})，計算其轉置並驗證 (A^T)^T = A。',
                    'solution': 'import torch\nA = torch.randn({m}, {n})\nA_T = A.T\nA_T_T = A_T.T\nprint(torch.equal(A, A_T_T))  # True',
                    'hints': ['使用 .T 屬性', '使用 torch.equal() 比較'],
                    'concepts': ['矩陣轉置', '矩陣性質']
                },
                {
                    'question': '計算兩個向量 u = [{u}] 和 v = [{v}] 的點積，並驗證結果。',
                    'solution': 'import torch\nu = torch.tensor([{u}], dtype=torch.float32)\nv = torch.tensor([{v}], dtype=torch.float32)\ndot_product = torch.dot(u, v)',
                    'hints': ['使用 torch.dot()', '點積等於按元素乘積的和'],
                    'concepts': ['向量點積', '內積運算']
                }
            ],
            'hard': [
                {
                    'question': '實現 Hadamard 積（按元素乘法）和矩陣乘法，並分析它們的計算複雜度差異。',
                    'solution': '''import torch
import time

A = torch.randn(1000, 1000)
B = torch.randn(1000, 1000)

# Hadamard 積
start = time.time()
hadamard = A * B
print(f"Hadamard 時間: {time.time() - start:.4f}s")

# 矩陣乘法
start = time.time()
matmul = torch.mm(A, B)
print(f"矩陣乘法時間: {time.time() - start:.4f}s")''',
                    'hints': ['Hadamard 積: O(n²)', '矩陣乘法: O(n³)', '測量實際執行時間'],
                    'concepts': ['計算複雜度', '矩陣運算', '性能分析']
                }
            ]
        }

        level_templates = templates.get(self.difficulty, templates['easy'])

        for i in range(count):
            template = random.choice(level_templates)
            exercise = template.copy()

            # 生成隨機參數
            params = {
                'v': ', '.join(str(random.randint(1, 10)) for _ in range(3)),
                'u': ', '.join(str(random.randint(1, 10)) for _ in range(4)),
                'm': random.randint(3, 5),
                'n': random.randint(3, 5)
            }

            exercise['question'] = exercise['question'].format(**params)
            exercise['solution'] = exercise['solution'].format(**params)
            exercise['id'] = f"{self.topic}_{self.difficulty}_{i+1}"
            exercise['difficulty'] = self.difficulty

            exercises.append(exercise)

        return exercises

    def _generate_calculus_exercises(self, count: int) -> List[Dict[str, Any]]:
        """生成微積分練習題"""
        exercises = []
        templates = {
            'easy': [
                {
                    'question': '計算函數 f(x) = x² 在 x = {x} 處的導數（使用數值方法）。',
                    'solution': '''import torch

def f(x):
    return x ** 2

x = torch.tensor([{x}], requires_grad=True)
y = f(x)
y.backward()
derivative = x.grad  # 應該接近 2*{x}''',
                    'hints': ['使用 requires_grad=True', '調用 .backward()', '從 .grad 獲取梯度'],
                    'concepts': ['導數', '自動微分']
                }
            ],
            'medium': [
                {
                    'question': '計算函數 f(x, y) = x² + y² 在點 ({x}, {y}) 處的梯度。',
                    'solution': '''import torch

x = torch.tensor([{x}], requires_grad=True)
y = torch.tensor([{y}], requires_grad=True)
f = x**2 + y**2
f.backward()
grad_x = x.grad  # 2*{x}
grad_y = y.grad  # 2*{y}''',
                    'hints': ['多變量函數的梯度', '偏導數的計算'],
                    'concepts': ['梯度', '偏導數', '多元微積分']
                }
            ]
        }

        level_templates = templates.get(self.difficulty, templates['easy'])

        for i in range(count):
            template = random.choice(level_templates)
            exercise = template.copy()

            params = {
                'x': random.uniform(1.0, 5.0),
                'y': random.uniform(1.0, 5.0)
            }

            exercise['question'] = exercise['question'].format(**params)
            exercise['solution'] = exercise['solution'].format(**params)
            exercise['id'] = f"{self.topic}_{self.difficulty}_{i+1}"
            exercise['difficulty'] = self.difficulty

            exercises.append(exercise)

        return exercises

    def _generate_autograd_exercises(self, count: int) -> List[Dict[str, Any]]:
        """生成自動微分練習題"""
        exercises = []
        # 實現類似的模板
        return self._generate_calculus_exercises(count)

    def _generate_probability_exercises(self, count: int) -> List[Dict[str, Any]]:
        """生成概率統計練習題"""
        exercises = []
        templates = {
            'easy': [
                {
                    'question': '從標準正態分佈中生成 {n} 個樣本，並計算其均值和標準差。',
                    'solution': '''import torch

samples = torch.randn({n})
mean = samples.mean()
std = samples.std()
print(f"均值: {{mean:.4f}}, 標準差: {{std:.4f}}")''',
                    'hints': ['使用 torch.randn()', '理論均值=0，標準差=1'],
                    'concepts': ['正態分佈', '統計量計算']
                }
            ]
        }

        level_templates = templates.get(self.difficulty, templates['easy'])

        for i in range(count):
            template = random.choice(level_templates)
            exercise = template.copy()

            params = {'n': random.randint(100, 1000)}

            exercise['question'] = exercise['question'].format(**params)
            exercise['solution'] = exercise['solution'].format(**params)
            exercise['id'] = f"{self.topic}_{self.difficulty}_{i+1}"
            exercise['difficulty'] = self.difficulty

            exercises.append(exercise)

        return exercises


def save_exercises(exercises: List[Dict[str, Any]], output_file: str):
    """保存練習題到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_count': len(exercises),
            'exercises': exercises
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 {len(exercises)} 道練習題到 {output_file}")


def print_exercises(exercises: List[Dict[str, Any]]):
    """打印練習題"""
    print(f"\n{'='*80}")
    print(f"生成了 {len(exercises)} 道練習題")
    print(f"{'='*80}\n")

    for i, ex in enumerate(exercises, 1):
        print(f"📝 練習 {i}: {ex['id']}")
        print(f"難度: {ex['difficulty']}")
        print(f"\n問題：\n{ex['question']}")
        print(f"\n概念: {', '.join(ex['concepts'])}")
        print(f"\n提示：")
        for hint in ex['hints']:
            print(f"  💡 {hint}")
        print(f"\n參考解答：\n```python\n{ex['solution']}\n```")
        print(f"\n{'-'*80}\n")


def main():
    parser = argparse.ArgumentParser(description='AI 輔助練習生成器')
    parser.add_argument('--topic', type=str, required=True,
                       choices=['ndarray', 'linear_algebra', 'calculus', 'autograd', 'probability'],
                       help='練習主題')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard', 'expert'],
                       help='難度等級')
    parser.add_argument('--count', type=int, default=5,
                       help='生成練習題數量')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出文件路徑（可選）')

    args = parser.parse_args()

    print(f"🤖 正在生成 {args.topic} 主題的 {args.difficulty} 難度練習題...")

    generator = ExerciseGenerator(args.topic, args.difficulty)
    exercises = generator.generate(args.count)

    print_exercises(exercises)

    if args.output:
        save_exercises(exercises, args.output)


if __name__ == '__main__':
    main()
