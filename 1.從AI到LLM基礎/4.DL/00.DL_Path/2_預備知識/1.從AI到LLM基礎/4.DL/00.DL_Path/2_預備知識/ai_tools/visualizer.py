"""
數學概念可視化工具
================

交互式可視化深度學習中的數學概念。
支持：梯度下降、線性變換、激活函數、概率分佈等。

使用方法：
    python visualizer.py --concept gradient_descent
    python visualizer.py --concept activation_functions --interactive

作者：AI Learning Community
版本：v1.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Tuple
import torch


class MathVisualizer:
    """數學概念可視化器"""

    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def visualize_gradient_descent(self, interactive=False):
        """可視化梯度下降過程"""
        print("📊 可視化梯度下降...")

        # 定義目標函數 f(x) = x^2
        def f(x):
            return x ** 2

        def df(x):
            return 2 * x

        # 梯度下降
        x_init = 5.0
        learning_rate = 0.1
        iterations = 20

        x_history = [x_init]
        x = x_init

        for _ in range(iterations):
            grad = df(x)
            x = x - learning_rate * grad
            x_history.append(x)

        # 繪圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # 左圖：函數曲線和梯度下降路徑
        x_range = np.linspace(-6, 6, 200)
        y_range = f(x_range)

        ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²')
        ax1.plot(x_history, [f(x) for x in x_history], 'ro-',
                markersize=8, linewidth=1.5, label='梯度下降路徑')
        ax1.scatter([x_history[0]], [f(x_history[0])], color='green',
                   s=200, marker='*', label='起點', zorder=5)
        ax1.scatter([x_history[-1]], [f(x_history[-1])], color='red',
                   s=200, marker='*', label='終點', zorder=5)
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('f(x)', fontsize=12)
        ax1.set_title('梯度下降優化過程', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右圖：收斂曲線
        ax2.plot(range(len(x_history)), [f(x) for x in x_history],
                'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次數', fontsize=12)
        ax2.set_ylabel('函數值 f(x)', fontsize=12)
        ax2.set_title('收斂曲線', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./ai_tools/gradient_descent.png', dpi=150, bbox_inches='tight')
        print("✅ 圖片已保存至 gradient_descent.png")
        plt.show()

    def visualize_linear_transformation(self):
        """可視化線性變換"""
        print("📊 可視化線性變換...")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始向量
        original_vectors = np.array([[1, 0], [0, 1], [1, 1], [2, 1]])

        transformations = [
            ("恆等變換", np.array([[1, 0], [0, 1]])),
            ("縮放變換", np.array([[2, 0], [0, 2]])),
            ("旋轉變換 (45°)", np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                       [np.sin(np.pi/4), np.cos(np.pi/4)]])),
            ("剪切變換", np.array([[1, 0.5], [0, 1]])),
            ("反射變換", np.array([[1, 0], [0, -1]])),
            ("投影變換", np.array([[1, 0], [0, 0]])),
        ]

        for idx, (title, matrix) in enumerate(transformations):
            ax = axes[idx // 3, idx % 3]

            # 繪製原始向量
            for vec in original_vectors:
                ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1,
                        fc='blue', ec='blue', alpha=0.3, linewidth=1.5, label='原始' if vec[0] == 1 and vec[1] == 0 else '')

            # 繪製變換後的向量
            transformed_vectors = (matrix @ original_vectors.T).T
            for vec in transformed_vectors:
                ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1,
                        fc='red', ec='red', linewidth=2, label='變換後' if vec[0] == matrix[0,0] and vec[1] == matrix[1,0] else '')

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            if idx == 0:
                ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('./ai_tools/linear_transformations.png', dpi=150, bbox_inches='tight')
        print("✅ 圖片已保存至 linear_transformations.png")
        plt.show()

    def visualize_activation_functions(self):
        """可視化激活函數"""
        print("📊 可視化激活函數...")

        x = torch.linspace(-5, 5, 200)

        activations = {
            'Sigmoid': torch.sigmoid(x),
            'Tanh': torch.tanh(x),
            'ReLU': torch.relu(x),
            'Leaky ReLU': torch.nn.functional.leaky_relu(x, 0.1),
            'ELU': torch.nn.functional.elu(x),
            'GELU': torch.nn.functional.gelu(x)
        }

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()

        for idx, (name, y) in enumerate(activations.items()):
            ax = axes[idx]
            ax.plot(x.numpy(), y.numpy(), linewidth=2.5, color='royalblue')
            ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linewidth=0.5, linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_title(name, fontsize=13, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')

        plt.tight_layout()
        plt.savefig('./ai_tools/activation_functions.png', dpi=150, bbox_inches='tight')
        print("✅ 圖片已保存至 activation_functions.png")
        plt.show()

    def visualize_probability_distributions(self):
        """可視化概率分佈"""
        print("📊 可視化概率分佈...")

        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        axes = axes.flatten()

        # 1. 正態分佈
        x = np.linspace(-5, 5, 200)
        for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            axes[0].plot(x, y, linewidth=2, label=f'μ={mu}, σ={sigma}')
        axes[0].set_title('正態分佈', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 均勻分佈
        samples = torch.rand(10000)
        axes[1].hist(samples.numpy(), bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].set_title('均勻分佈', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 3. 伯努利分佈
        p_values = [0.3, 0.5, 0.7]
        x_bernoulli = [0, 1]
        for p in p_values:
            y_bernoulli = [1-p, p]
            axes[2].plot(x_bernoulli, y_bernoulli, 'o-', linewidth=2, markersize=8, label=f'p={p}')
        axes[2].set_title('伯努利分佈', fontweight='bold')
        axes[2].set_xticks([0, 1])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. 指數分佈
        x_exp = np.linspace(0, 5, 200)
        for lambda_param in [0.5, 1, 2]:
            y_exp = lambda_param * np.exp(-lambda_param * x_exp)
            axes[3].plot(x_exp, y_exp, linewidth=2, label=f'λ={lambda_param}')
        axes[3].set_title('指數分佈', fontweight='bold')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # 5. 二項分佈
        n, p = 10, 0.5
        from scipy.stats import binom
        x_binom = np.arange(0, n+1)
        y_binom = binom.pmf(x_binom, n, p)
        axes[4].bar(x_binom, y_binom, alpha=0.7, color='coral', edgecolor='black')
        axes[4].set_title(f'二項分佈 (n={n}, p={p})', fontweight='bold')
        axes[4].grid(True, alpha=0.3)

        # 6. 泊松分佈
        from scipy.stats import poisson
        x_poisson = np.arange(0, 15)
        for lambda_p in [1, 4, 7]:
            y_poisson = poisson.pmf(x_poisson, lambda_p)
            axes[5].plot(x_poisson, y_poisson, 'o-', linewidth=2, markersize=6, label=f'λ={lambda_p}')
        axes[5].set_title('泊松分佈', fontweight='bold')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./ai_tools/probability_distributions.png', dpi=150, bbox_inches='tight')
        print("✅ 圖片已保存至 probability_distributions.png")
        plt.show()

    def visualize_matrix_multiplication(self):
        """可視化矩陣乘法"""
        print("📊 可視化矩陣乘法...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 創建示例矩陣
        A = np.array([[1, 2, 3], [4, 5, 6]])
        B = np.array([[7, 8], [9, 10], [11, 12]])
        C = A @ B

        matrices = [
            (A, 'Matrix A (2×3)'),
            (B, 'Matrix B (3×2)'),
            (C, 'Result C = A @ B (2×2)')
        ]

        for ax, (matrix, title) in zip(axes, matrices):
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=13, fontweight='bold')

            # 添加數值
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    text = ax.text(j, i, f'{matrix[i, j]:.0f}',
                                 ha="center", va="center", color="white", fontsize=14, fontweight='bold')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig('./ai_tools/matrix_multiplication.png', dpi=150, bbox_inches='tight')
        print("✅ 圖片已保存至 matrix_multiplication.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='數學概念可視化工具')
    parser.add_argument('--concept', type=str, required=True,
                       choices=['gradient_descent', 'linear_transformation', 'activation_functions',
                               'probability_distributions', 'matrix_multiplication', 'all'],
                       help='要可視化的概念')
    parser.add_argument('--interactive', action='store_true',
                       help='啟用交互模式')

    args = parser.parse_args()

    visualizer = MathVisualizer()

    concepts = {
        'gradient_descent': visualizer.visualize_gradient_descent,
        'linear_transformation': visualizer.visualize_linear_transformation,
        'activation_functions': visualizer.visualize_activation_functions,
        'probability_distributions': visualizer.visualize_probability_distributions,
        'matrix_multiplication': visualizer.visualize_matrix_multiplication,
    }

    if args.concept == 'all':
        for name, func in concepts.items():
            print(f"\n{'='*60}")
            print(f"可視化: {name}")
            print(f"{'='*60}")
            func() if name != 'gradient_descent' else func(args.interactive)
    else:
        concepts[args.concept](args.interactive) if args.concept == 'gradient_descent' else concepts[args.concept]()


if __name__ == '__main__':
    main()
