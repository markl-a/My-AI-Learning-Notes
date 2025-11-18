"""
遷移學習實戰教程
================

本教程將教你如何使用預訓練模型進行遷移學習：
1. 使用預訓練的 ResNet、VGG 等模型
2. 特徵提取 vs 微調
3. 適配自定義數據集
4. 超參數調優策略
5. 實戰案例：貓狗分類、花卉識別

作者：AI Learning Notes
日期：2024-11
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time


# ==================== 第一部分：遷移學習基礎 ====================

class TransferLearningModel:
    """
    遷移學習模型包裝器

    支持不同的預訓練模型和遷移學習策略
    """

    def __init__(self, model_name='resnet18', num_classes=2, pretrained=True):
        """
        初始化遷移學習模型

        Args:
            model_name: 預訓練模型名稱 ('resnet18', 'resnet50', 'vgg16', 'mobilenet_v2')
            num_classes: 目標類別數
            pretrained: 是否使用預訓練權重
        """
        self.model_name = model_name
        self.num_classes = num_classes

        # 加載預訓練模型
        self.model = self._load_model(pretrained)

        print(f"已加載 {model_name} (pretrained={pretrained})")
        print(f"目標類別數: {num_classes}")

    def _load_model(self, pretrained):
        """加載預訓練模型並修改最後一層"""
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, self.num_classes)

        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        return model

    def freeze_backbone(self, freeze=True):
        """
        凍結/解凍骨幹網絡

        Args:
            freeze: True 表示凍結，False 表示解凍
        """
        # 確定要凍結的參數
        if 'resnet' in self.model_name:
            params_to_freeze = list(self.model.parameters())[:-2]  # 除了最後的 fc 層
        elif 'vgg' in self.model_name:
            params_to_freeze = self.model.features.parameters()
        elif 'mobilenet' in self.model_name:
            params_to_freeze = self.model.features.parameters()
        else:
            params_to_freeze = []

        # 凍結/解凍參數
        for param in params_to_freeze:
            param.requires_grad = not freeze

        status = "已凍結" if freeze else "已解凍"
        print(f"{status}骨幹網絡參數")

    def get_trainable_params(self):
        """獲取可訓練參數"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only=False):
        """計算參數量"""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())


# ==================== 第二部分：數據準備 ====================

class CustomDataset(Dataset):
    """
    自定義數據集

    支持從文件夾加載圖像數據
    """

    def __init__(self, root_dir, transform=None):
        """
        初始化數據集

        Args:
            root_dir: 根目錄，包含各類別子文件夾
            transform: 數據轉換
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = []

        # 掃描目錄
        self._scan_directory()

    def _scan_directory(self):
        """掃描目錄並構建數據集"""
        if not os.path.exists(self.root_dir):
            print(f"警告: 目錄 {self.root_dir} 不存在")
            return

        # 獲取類別名稱
        self.class_names = sorted([d for d in os.listdir(self.root_dir)
                                  if os.path.isdir(os.path.join(self.root_dir, d))])

        # 掃描每個類別
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.root_dir, class_name)

            # 獲取所有圖像文件
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

        print(f"找到 {len(self.images)} 張圖像，{len(self.class_names)} 個類別")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加載圖像
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # 應用轉換
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def get_data_transforms(input_size=224, augment=True):
    """
    獲取數據轉換

    Args:
        input_size: 輸入圖像大小
        augment: 是否使用數據增強

    Returns:
        train_transform, val_transform: 訓練和驗證的轉換
    """
    # ImageNet 的均值和標準差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        # 訓練集：使用數據增強
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # 不使用數據增強
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # 驗證集：不使用數據增強
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transform, val_transform


# ==================== 第三部分：訓練策略 ====================

class TransferLearningTrainer:
    """
    遷移學習訓練器

    支持兩種策略：
    1. 特徵提取：凍結骨幹網絡，只訓練新的分類層
    2. 微調：先訓練分類層，再解凍並微調整個網絡
    """

    def __init__(self, model, train_loader, val_loader, device='cpu'):
        """
        初始化訓練器

        Args:
            model: 模型（TransferLearningModel 的實例）
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            device: 計算設備
        """
        self.model_wrapper = model
        self.model = model.model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 損失函數
        self.criterion = nn.CrossEntropyLoss()

        # 訓練歷史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, optimizer):
        """訓練一個 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向傳播
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            optimizer.step()

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

        val_loss = running_loss / total
        val_acc = 100. * correct / total

        return val_loss, val_acc

    def train_feature_extraction(self, num_epochs=10, lr=0.001):
        """
        特徵提取策略

        凍結骨幹網絡，只訓練新的分類層

        Args:
            num_epochs: 訓練輪數
            lr: 學習率
        """
        print("\n" + "="*60)
        print("階段 1: 特徵提取（凍結骨幹網絡）")
        print("="*60)

        # 凍結骨幹網絡
        self.model_wrapper.freeze_backbone(freeze=True)

        # 只優化可訓練參數
        optimizer = optim.Adam(self.model_wrapper.get_trainable_params(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # 訓練
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch(optimizer)
            val_loss, val_acc = self.validate()

            # 更新學習率
            scheduler.step()

            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印結果
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    def train_fine_tuning(self, num_epochs=10, lr=0.0001):
        """
        微調策略

        解凍骨幹網絡，使用較小的學習率微調整個網絡

        Args:
            num_epochs: 訓練輪數
            lr: 學習率（通常較小）
        """
        print("\n" + "="*60)
        print("階段 2: 微調（解凍骨幹網絡）")
        print("="*60)

        # 解凍骨幹網絡
        self.model_wrapper.freeze_backbone(freeze=False)

        # 使用較小的學習率
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # 訓練
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)

            train_loss, train_acc = self.train_epoch(optimizer)
            val_loss, val_acc = self.validate()

            # 更新學習率
            scheduler.step()

            # 記錄歷史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印結果
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    def plot_history(self, save_path='transfer_learning_history.png'):
        """繪製訓練歷史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n訓練歷史已保存至: {save_path}")


# ==================== 第四部分：實戰案例 ====================

def demo_transfer_learning():
    """演示遷移學習"""
    print("遷移學習實戰演示")
    print("="*60)

    # 檢測設備
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用設備: {device}")

    # 使用 CIFAR-10 作為示例數據集
    print("\n加載數據集...")
    train_transform, val_transform = get_data_transforms(input_size=224, augment=True)

    # 加載 CIFAR-10（只使用貓和狗兩個類別）
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=train_transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                   transform=val_transform)

    # 只保留貓（類別3）和狗（類別5）
    cat_dog_indices_train = [i for i, (_, label) in enumerate(train_dataset)
                             if label in [3, 5]]
    cat_dog_indices_val = [i for i, (_, label) in enumerate(val_dataset)
                           if label in [3, 5]]

    train_dataset = torch.utils.data.Subset(train_dataset, cat_dog_indices_train)
    val_dataset = torch.utils.data.Subset(val_dataset, cat_dog_indices_val)

    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")

    # 創建模型
    print("\n創建模型...")
    model = TransferLearningModel(model_name='resnet18', num_classes=2, pretrained=True)
    print(f"總參數量: {model.count_parameters():,}")
    print(f"可訓練參數: {model.count_parameters(trainable_only=True):,}")

    # 創建訓練器
    trainer = TransferLearningTrainer(model, train_loader, val_loader, device)

    # 策略 1: 特徵提取
    trainer.train_feature_extraction(num_epochs=5, lr=0.001)

    # 策略 2: 微調
    trainer.train_fine_tuning(num_epochs=5, lr=0.0001)

    # 繪製歷史
    trainer.plot_history()

    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == '__main__':
    demo_transfer_learning()
