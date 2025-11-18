"""
CNN 可視化工具箱
================

本模組提供了豐富的 CNN 可視化工具，幫助你深入理解模型的工作原理：
1. 特徵圖可視化 - 查看每層的激活
2. 卷積核可視化 - 理解學到的濾波器
3. 類激活映射 (CAM) - 了解模型關注的區域
4. Grad-CAM - 梯度加權的類激活映射
5. 激活值統計分析
6. 濾波器響應可視化

作者：AI Learning Notes
日期：2024-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import seaborn as sns


class CNNVisualizer:
    """
    CNN 可視化器

    提供多種可視化方法來理解 CNN 的工作原理
    """

    def __init__(self, model, device='cpu'):
        """
        初始化可視化器

        Args:
            model: 要可視化的 CNN 模型
            device: 計算設備
        """
        self.model = model
        self.device = device
        self.model.eval()

        # 用於存儲中間層輸出
        self.activations = {}
        self.gradients = {}

    def register_hooks(self, target_layers=None):
        """
        註冊鉤子函數以捕獲中間層輸出

        Args:
            target_layers: 目標層的名稱列表，None表示所有卷積層
        """

        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        # 註冊前向和後向鉤子
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if target_layers is None or name in target_layers:
                    module.register_forward_hook(forward_hook(name))
                    module.register_full_backward_hook(backward_hook(name))

    def visualize_feature_maps(self, input_tensor, layer_name, num_features=16, save_path=None):
        """
        可視化特徵圖

        Args:
            input_tensor: 輸入張量 (1, C, H, W)
            layer_name: 要可視化的層名稱
            num_features: 要顯示的特徵圖數量
            save_path: 保存路徑
        """
        # 清空之前的激活
        self.activations.clear()

        # 前向傳播
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        # 獲取特徵圖
        if layer_name not in self.activations:
            print(f"警告: 層 '{layer_name}' 未找到")
            print(f"可用的層: {list(self.activations.keys())}")
            return

        feature_maps = self.activations[layer_name].cpu().numpy()[0]
        num_features = min(num_features, feature_maps.shape[0])

        # 計算網格大小
        n_cols = 4
        n_rows = (num_features + n_cols - 1) // n_cols

        # 創建圖形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            if idx < num_features:
                # 顯示特徵圖
                feature_map = feature_maps[idx]
                im = ax.imshow(feature_map, cmap='viridis')
                ax.set_title(f'Feature {idx}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')

        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=14, y=1.002)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"特徵圖已保存至: {save_path}")

        plt.show()

    def visualize_filters(self, layer_name, num_filters=16, save_path=None):
        """
        可視化卷積核(濾波器)

        Args:
            layer_name: 層名稱
            num_filters: 要顯示的濾波器數量
            save_path: 保存路徑
        """
        # 獲取指定層
        layer = None
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                layer = module
                break

        if layer is None:
            print(f"錯誤: 未找到卷積層 '{layer_name}'")
            return

        # 獲取權重
        weights = layer.weight.data.cpu().numpy()
        num_filters = min(num_filters, weights.shape[0])

        # 計算網格大小
        n_cols = 4
        n_rows = (num_filters + n_cols - 1) // n_cols

        # 創建圖形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx in range(n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            if idx < num_filters:
                # 顯示濾波器
                filter_weight = weights[idx]

                # 如果是多通道，取平均或顯示第一個通道
                if filter_weight.shape[0] > 1:
                    filter_weight = filter_weight.mean(axis=0)
                else:
                    filter_weight = filter_weight[0]

                # 歸一化到 [0, 1]
                vmin, vmax = filter_weight.min(), filter_weight.max()
                if vmax - vmin > 0:
                    filter_weight = (filter_weight - vmin) / (vmax - vmin)

                im = ax.imshow(filter_weight, cmap='gray')
                ax.set_title(f'Filter {idx}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.axis('off')

        plt.suptitle(f'Convolutional Filters from {layer_name}', fontsize=14, y=1.002)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"濾波器已保存至: {save_path}")

        plt.show()

    def visualize_activation_statistics(self, input_tensor, save_path=None):
        """
        可視化激活值統計

        Args:
            input_tensor: 輸入張量
            save_path: 保存路徑
        """
        # 清空之前的激活
        self.activations.clear()

        # 前向傳播
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        # 計算統計信息
        stats = {}
        for name, activation in self.activations.items():
            act = activation.cpu().numpy()
            stats[name] = {
                'mean': act.mean(),
                'std': act.std(),
                'min': act.min(),
                'max': act.max(),
                'sparsity': (act == 0).sum() / act.size * 100
            }

        # 繪製統計圖
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        layer_names = list(stats.keys())
        metrics = ['mean', 'std', 'min', 'max', 'sparsity']
        titles = ['Mean Activation', 'Std Deviation', 'Min Value', 'Max Value', 'Sparsity (%)']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            values = [stats[name][metric] for name in layer_names]
            axes[idx].bar(range(len(layer_names)), values)
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel('Value')
            axes[idx].set_xticks(range(len(layer_names)))
            axes[idx].set_xticklabels([f'L{i}' for i in range(len(layer_names))], rotation=45)
            axes[idx].grid(True, alpha=0.3)

        # 隱藏最後一個子圖
        axes[-1].axis('off')

        plt.suptitle('Activation Statistics Across Layers', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"統計圖已保存至: {save_path}")

        plt.show()

        # 打印詳細統計
        print("\n詳細統計信息:")
        print("="*80)
        print(f"{'Layer':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Sparsity':>10}")
        print("="*80)
        for name, stat in stats.items():
            print(f"{name:<20} {stat['mean']:>10.4f} {stat['std']:>10.4f} "
                  f"{stat['min']:>10.4f} {stat['max']:>10.4f} {stat['sparsity']:>9.2f}%")
        print("="*80)


class GradCAM:
    """
    Grad-CAM 實現

    梯度加權類激活映射，用於可視化模型關注的區域
    """

    def __init__(self, model, target_layer):
        """
        初始化 Grad-CAM

        Args:
            model: CNN 模型
            target_layer: 目標層（通常是最後一個卷積層）
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 註冊鉤子
        self._register_hooks()

    def _register_hooks(self):
        """註冊前向和後向鉤子"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 找到目標層並註冊鉤子
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break

    def generate_cam(self, input_tensor, target_class=None):
        """
        生成類激活映射

        Args:
            input_tensor: 輸入張量 (1, C, H, W)
            target_class: 目標類別，None表示預測類別

        Returns:
            cam: 類激活映射
            prediction: 預測類別
        """
        # 前向傳播
        self.model.eval()
        output = self.model(input_tensor)

        # 獲取預測類別
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向傳播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 計算權重
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        # 全局平均池化得到權重
        weights = gradients.mean(axis=(1, 2))  # (C,)

        # 加權求和
        cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)

        # ReLU
        cam = np.maximum(cam, 0)

        # 歸一化到 [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class

    def visualize(self, input_tensor, original_image, target_class=None, save_path=None):
        """
        可視化 Grad-CAM

        Args:
            input_tensor: 輸入張量
            original_image: 原始圖像 (PIL Image 或 numpy array)
            target_class: 目標類別
            save_path: 保存路徑
        """
        # 生成 CAM
        cam, predicted_class = self.generate_cam(input_tensor, target_class)

        # 將 CAM 調整到輸入圖像大小
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        h, w = original_image.shape[:2]
        cam_resized = np.array(Image.fromarray(cam).resize((w, h), Image.BILINEAR))

        # 創建熱力圖
        heatmap = plt.cm.jet(cam_resized)[:, :, :3]

        # 疊加在原始圖像上
        alpha = 0.5
        superimposed = heatmap * alpha + original_image.astype(np.float32) / 255 * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 1)

        # 顯示結果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始圖像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 熱力圖
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title(f'Grad-CAM Heatmap\n(Class: {predicted_class})')
        axes[1].axis('off')

        # 疊加圖像
        axes[2].imshow(superimposed)
        axes[2].set_title('Superimposed')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM 可視化已保存至: {save_path}")

        plt.show()


class FilterResponseVisualizer:
    """
    濾波器響應可視化器

    顯示不同濾波器對輸入圖像的響應
    """

    def __init__(self, model, layer_name):
        """
        初始化可視化器

        Args:
            model: CNN 模型
            layer_name: 要可視化的層名稱
        """
        self.model = model
        self.layer_name = layer_name
        self.activations = None

        # 註冊鉤子
        self._register_hook()

    def _register_hook(self):
        """註冊前向鉤子"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # 找到目標層並註冊鉤子
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(forward_hook)
                break

    def visualize(self, input_tensor, original_image, num_filters=9, save_path=None):
        """
        可視化濾波器響應

        Args:
            input_tensor: 輸入張量
            original_image: 原始圖像
            num_filters: 要顯示的濾波器數量
            save_path: 保存路徑
        """
        # 前向傳播
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.activations is None:
            print(f"錯誤: 未能捕獲層 '{self.layer_name}' 的激活")
            return

        # 獲取激活
        activations = self.activations.cpu().numpy()[0]
        num_filters = min(num_filters, activations.shape[0])

        # 計算網格大小
        n_cols = 3
        n_rows = (num_filters + n_cols) // n_cols

        # 創建圖形
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 顯示原始圖像
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 顯示濾波器響應
        filter_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if row == 0 and col == 0:
                    continue

                ax = axes[row, col]
                if filter_idx < num_filters:
                    response = activations[filter_idx]
                    im = ax.imshow(response, cmap='viridis')
                    ax.set_title(f'Filter {filter_idx}')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    filter_idx += 1
                ax.axis('off')

        plt.suptitle(f'Filter Responses from {self.layer_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"濾波器響應已保存至: {save_path}")

        plt.show()


# ==================== 使用示例 ====================

def demo_visualization():
    """演示可視化功能"""
    print("CNN 可視化工具演示")
    print("="*60)

    # 創建一個簡單的 CNN 模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.Linear(64 * 4 * 4, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # 創建模型和隨機輸入
    model = SimpleCNN()
    input_tensor = torch.randn(1, 3, 32, 32)

    # 1. 特徵圖可視化
    print("\n1. 特徵圖可視化...")
    visualizer = CNNVisualizer(model)
    visualizer.register_hooks()
    visualizer.visualize_feature_maps(input_tensor, 'conv1', num_features=16,
                                     save_path='feature_maps.png')

    # 2. 濾波器可視化
    print("\n2. 濾波器可視化...")
    visualizer.visualize_filters('conv1', num_filters=16,
                                 save_path='filters.png')

    # 3. 激活值統計
    print("\n3. 激活值統計...")
    visualizer.visualize_activation_statistics(input_tensor,
                                               save_path='activation_stats.png')

    # 4. Grad-CAM（需要真實圖像）
    print("\n4. Grad-CAM 示例...")
    print("提示: 使用真實圖像和訓練好的模型可獲得更好的效果")

    # 創建隨機圖像作為示例
    random_image = (torch.randn(3, 32, 32).permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
    random_image = np.clip(random_image, 0, 255).astype(np.uint8)

    grad_cam = GradCAM(model, 'conv3')
    grad_cam.visualize(input_tensor, random_image, save_path='gradcam.png')

    # 5. 濾波器響應
    print("\n5. 濾波器響應可視化...")
    filter_viz = FilterResponseVisualizer(model, 'conv2')
    filter_viz.visualize(input_tensor, random_image, num_filters=9,
                        save_path='filter_responses.png')

    print("\n" + "="*60)
    print("演示完成！所有可視化結果已保存。")


if __name__ == '__main__':
    demo_visualization()
