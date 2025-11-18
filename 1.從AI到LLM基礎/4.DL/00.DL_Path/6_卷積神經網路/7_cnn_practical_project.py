"""
CNN 完整實戰項目 - CIFAR-10 圖像分類
=====================================

本教程將帶你完成一個完整的 CNN 項目，包括：
1. 數據加載和預處理
2. 數據增強技術
3. 模型設計和實現
4. 訓練過程管理
5. 性能評估和優化
6. 模型保存和加載

作者：AI Learning Notes
日期：2024-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os


# ==================== 第一部分：數據處理 ====================

class CIFAR10DataLoader:
    """
    CIFAR-10 數據加載器

    CIFAR-10 包含 10 個類別的 60,000 張 32x32 彩色圖像：
    - airplane, automobile, bird, cat, deer
    - dog, frog, horse, ship, truck
    """

    def __init__(self, data_dir='./data', batch_size=128, num_workers=2):
        """
        初始化數據加載器

        Args:
            data_dir: 數據保存目錄
            batch_size: 批量大小
            num_workers: 數據加載的工作進程數
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # CIFAR-10 的統計數據（用於歸一化）
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

        # 類別名稱
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    def get_transforms(self, augment=True):
        """
        獲取數據轉換管道

        Args:
            augment: 是否使用數據增強

        Returns:
            transform: 數據轉換管道
        """
        if augment:
            # 訓練集：使用數據增強
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            # 測試集：只進行基本轉換
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

        return transform

    def load_data(self, val_split=0.1):
        """
        加載並劃分數據集

        Args:
            val_split: 驗證集比例（從訓練集中劃分）

        Returns:
            train_loader: 訓練集數據加載器
            val_loader: 驗證集數據加載器
            test_loader: 測試集數據加載器
        """
        # 加載訓練集
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.get_transforms(augment=True)
        )

        # 劃分訓練集和驗證集
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 加載測試集
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.get_transforms(augment=False)
        )

        # 創建數據加載器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        print(f"數據集統計:")
        print(f"訓練集: {len(train_dataset)} 樣本")
        print(f"驗證集: {len(val_dataset)} 樣本")
        print(f"測試集: {len(test_dataset)} 樣本")

        return train_loader, val_loader, test_loader

    def show_samples(self, dataloader, num_samples=8):
        """
        顯示數據樣本

        Args:
            dataloader: 數據加載器
            num_samples: 顯示的樣本數量
        """
        # 獲取一個批次的數據
        images, labels = next(iter(dataloader))

        # 反歸一化以便顯示
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        images = images * std + mean

        # 創建圖形
        fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 5))
        axes = axes.flatten()

        for idx in range(num_samples):
            img = images[idx].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)

            axes[idx].imshow(img)
            axes[idx].set_title(self.classes[labels[idx]])
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("樣本圖像已保存為 cifar10_samples.png")


# ==================== 第二部分：模型設計 ====================

class ImprovedCNN(nn.Module):
    """
    改進的 CNN 模型

    架構特點：
    - 使用現代的卷積塊設計
    - 批量歸一化加速訓練
    - Dropout 防止過擬合
    - 殘差連接提升性能
    """

    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()

        # 第一個卷積塊
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )

        # 第二個卷積塊
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # 第三個卷積塊
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )

        # 全連接層
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """前向傳播"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

    def count_parameters(self):
        """計算模型參數量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== 第三部分：訓練管理器 ====================

class Trainer:
    """
    訓練管理器

    負責管理整個訓練過程，包括：
    - 訓練和驗證
    - 學習率調度
    - 早停機制
    - 模型保存
    - 訓練可視化
    """

    def __init__(self, model, train_loader, val_loader, test_loader, device):
        """
        初始化訓練器

        Args:
            model: 神經網絡模型
            train_loader: 訓練集數據加載器
            val_loader: 驗證集數據加載器
            test_loader: 測試集數據加載器
            device: 計算設備
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # 損失函數和優化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # 訓練歷史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # 最佳模型
        self.best_val_acc = 0.0
        self.best_model_path = 'best_model.pth'

    def train_epoch(self):
        """訓練一個 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            self.optimizer.step()

            # 統計
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

            # 同步設備
            if self.device.type == 'mps':
                torch.mps.synchronize()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """驗證模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if self.device.type == 'mps':
                    torch.mps.synchronize()

        val_loss = running_loss / total
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train(self, num_epochs=50, early_stopping_patience=10):
        """
        完整訓練流程

        Args:
            num_epochs: 訓練輪數
            early_stopping_patience: 早停耐心值
        """
        print(f"\n開始訓練 (設備: {self.device})")
        print(f"模型參數量: {self.model.count_parameters():,}")
        print("="*60)

        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)

            # 訓練
            train_loss, train_acc = self.train_epoch()

            # 驗證
            val_loss, val_acc = self.validate()

            # 學習率調整
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # 打印結果
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.best_model_path)
                print(f"✓ 保存最佳模型 (Acc: {val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"\n早停！驗證準確率 {early_stopping_patience} 個 epoch 未提升")
                break

        print("\n" + "="*60)
        print(f"訓練完成！最佳驗證準確率: {self.best_val_acc:.2f}%")

        # 繪製訓練曲線
        self.plot_history()

        # 在測試集上評估
        self.evaluate_test_set()

    def plot_history(self):
        """繪製訓練歷史曲線"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        epochs = range(1, len(self.history['train_loss']) + 1)

        # 損失曲線
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_title('Loss History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 準確率曲線
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_title('Accuracy History')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 學習率曲線
        axes[2].plot(epochs, self.history['learning_rate'], 'g-')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n訓練歷史已保存為 training_history.png")

    def evaluate_test_set(self):
        """在測試集上評估模型"""
        # 加載最佳模型
        checkpoint = torch.load(self.best_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        correct = 0
        total = 0

        # 每個類別的統計
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 統計每個類別
                c = predicted.eq(labels)
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                if self.device.type == 'mps':
                    torch.mps.synchronize()

        # 總體準確率
        test_acc = 100. * correct / total
        print(f"\n測試集準確率: {test_acc:.2f}%")

        # 每個類別的準確率
        print("\n各類別準確率:")
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        for i in range(10):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                print(f"{classes[i]:>12}: {acc:.2f}%")


# ==================== 第四部分：主程序 ====================

def main():
    """主程序"""
    print("="*60)
    print("CNN 完整實戰項目 - CIFAR-10 圖像分類")
    print("="*60)

    # 設置隨機種子
    torch.manual_seed(42)
    np.random.seed(42)

    # 檢測設備
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n使用設備: {device}")

    # 步驟1：加載數據
    print("\n步驟1：加載數據...")
    data_loader = CIFAR10DataLoader(batch_size=128)
    train_loader, val_loader, test_loader = data_loader.load_data(val_split=0.1)

    # 顯示樣本
    data_loader.show_samples(train_loader)

    # 步驟2：創建模型
    print("\n步驟2：創建模型...")
    model = ImprovedCNN(num_classes=10)
    print(f"模型參數量: {model.count_parameters():,}")

    # 步驟3：訓練模型
    print("\n步驟3：訓練模型...")
    trainer = Trainer(model, train_loader, val_loader, test_loader, device)
    trainer.train(num_epochs=50, early_stopping_patience=10)

    print("\n" + "="*60)
    print("項目完成！")
    print("="*60)


if __name__ == '__main__':
    main()
