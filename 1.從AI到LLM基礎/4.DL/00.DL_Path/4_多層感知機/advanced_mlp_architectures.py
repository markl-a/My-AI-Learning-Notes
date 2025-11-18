"""
進階 MLP 架構示例
包含現代深度學習技術的 MLP 實現

作者：Claude AI
日期：2025-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import List, Optional


# ==================== 1. 基礎 MLP（作為對比） ====================
class BasicMLP(nn.Module):
    """基礎的多層感知機"""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.net(x)


# ==================== 2. 帶 Dropout 的 MLP ====================
class MLPWithDropout(nn.Module):
    """帶 Dropout 正則化的 MLP"""
    def __init__(self, input_dim=784, hidden_dims=[256, 128],
                 output_dim=10, dropout_rate=0.5):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # 添加 Dropout
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ==================== 3. 帶批次正規化的 MLP ====================
class MLPWithBatchNorm(nn.Module):
    """帶批次正規化的 MLP（加速訓練、提高穩定性）"""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 批次正規化
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ==================== 4. 帶 Layer Normalization 的 MLP ====================
class MLPWithLayerNorm(nn.Module):
    """帶 Layer Normalization 的 MLP（適用於小批次或序列數據）"""
    def __init__(self, input_dim=784, hidden_dims=[256, 128], output_dim=10):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Layer Normalization
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# ==================== 5. 殘差 MLP（Residual MLP） ====================
class ResidualBlock(nn.Module):
    """殘差塊：允許梯度更好地流動"""
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # 殘差連接：y = x + f(x)
        return self.norm(x + self.net(x))


class ResidualMLP(nn.Module):
    """使用殘差連接的 MLP（可以訓練更深的網絡）"""
    def __init__(self, input_dim=784, hidden_dim=256, num_blocks=3, output_dim=10):
        super().__init__()

        # 輸入投影層
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 殘差塊
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # 輸出層
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)
        x = F.relu(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)


# ==================== 6. 現代化 MLP（組合多種技術） ====================
class ModernMLP(nn.Module):
    """
    現代化的 MLP 架構，組合多種技術：
    - 批次正規化
    - Dropout
    - 殘差連接
    - 更好的激活函數（GELU）
    """
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128],
                 output_dim=10, dropout_rate=0.3, use_residual=True):
        super().__init__()

        self.use_residual = use_residual

        # 輸入層
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),  # GELU 激活函數（比 ReLU 更平滑）
            nn.Dropout(dropout_rate)
        )

        # 隱藏層
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.hidden_layers.append(layer)

            # 如果使用殘差連接，需要投影層來匹配維度
            if use_residual and hidden_dims[i] != hidden_dims[i+1]:
                proj = nn.Linear(hidden_dims[i], hidden_dims[i+1])
                self.hidden_layers.append(proj)

        # 輸出層
        self.output = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)

        prev_x = x
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Sequential):
                x = layer(x)
                # 殘差連接
                if self.use_residual and x.shape == prev_x.shape:
                    x = x + prev_x
                prev_x = x
            else:  # 投影層
                prev_x = layer(prev_x)

        return self.output(x)


# ==================== 7. 帶注意力機制的 MLP（MLP-Mixer 風格） ====================
class MLPMixerLayer(nn.Module):
    """MLP-Mixer 風格的層（用於特徵混合）"""
    def __init__(self, dim, expansion_factor=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Token mixing MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim)
        )

        # Channel mixing MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim)
        )

    def forward(self, x):
        x = x + self.mlp1(self.norm1(x))
        x = x + self.mlp2(self.norm2(x))
        return x


class MLPMixer(nn.Module):
    """MLP-Mixer 架構（僅使用 MLP 實現類似注意力的效果）"""
    def __init__(self, input_dim=784, hidden_dim=256, num_layers=4, output_dim=10):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            MLPMixerLayer(hidden_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output(x)


# ==================== 8. 自適應 MLP（動態調整深度） ====================
class AdaptiveDepthMLP(nn.Module):
    """
    自適應深度 MLP（可以根據輸入動態調整網絡深度）
    使用早退出機制（Early Exit）
    """
    def __init__(self, input_dim=784, hidden_dim=256, max_depth=5, output_dim=10):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 多個處理塊，每個都可以輸出結果
        self.blocks = nn.ModuleList()
        self.exits = nn.ModuleList()

        for _ in range(max_depth):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            exit_layer = nn.Linear(hidden_dim, output_dim)

            self.blocks.append(block)
            self.exits.append(exit_layer)

        self.confidence_threshold = 0.9

    def forward(self, x, return_all=False):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_proj(x))

        outputs = []

        for block, exit_layer in zip(self.blocks, self.exits):
            x = block(x)
            output = exit_layer(x)
            outputs.append(output)

            # 訓練時返回所有輸出，推理時可以提前退出
            if not self.training and not return_all:
                confidence = F.softmax(output, dim=-1).max(dim=-1)[0].mean()
                if confidence > self.confidence_threshold:
                    return output

        return outputs if return_all else outputs[-1]


# ==================== 訓練輔助函數 ====================
def get_data_loaders(batch_size=256):
    """獲取 Fashion-MNIST 數據加載器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """評估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def compare_models(models_dict, num_epochs=10, batch_size=256):
    """
    比較不同 MLP 架構的性能

    Args:
        models_dict: 字典，包含模型名稱和模型實例
        num_epochs: 訓練輪數
        batch_size: 批次大小
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(batch_size)

    results = {}

    for model_name, model in models_dict.items():
        print(f"\n訓練 {model_name}...")
        print("=" * 50)

        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses, train_accs = [], []
        test_losses, test_accs = [], []

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(f'Epoch {epoch+1}/{num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_losses': test_losses,
            'test_accs': test_accs,
            'final_test_acc': test_accs[-1]
        }

    return results


def plot_comparison(results):
    """繪製模型比較圖"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 繪製損失
    for model_name, metrics in results.items():
        axes[0].plot(metrics['train_losses'], label=f'{model_name} (train)', linestyle='--')
        axes[0].plot(metrics['test_losses'], label=f'{model_name} (test)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 繪製準確率
    for model_name, metrics in results.items():
        axes[1].plot(metrics['test_accs'], label=model_name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


# ==================== 主程序示例 ====================
if __name__ == "__main__":
    # 創建不同的模型進行比較
    models = {
        'Basic MLP': BasicMLP(),
        'MLP + Dropout': MLPWithDropout(),
        'MLP + BatchNorm': MLPWithBatchNorm(),
        'Residual MLP': ResidualMLP(),
        'Modern MLP': ModernMLP(),
    }

    # 比較模型性能
    print("開始比較不同 MLP 架構...")
    results = compare_models(models, num_epochs=10)

    # 繪製比較圖
    plot_comparison(results)

    # 打印最終結果
    print("\n" + "=" * 50)
    print("最終測試準確率:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:20s}: {metrics['final_test_acc']:.2f}%")
