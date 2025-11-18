"""
AI 輔助學習工具
===============

這個模組提供 AI 驅動的學習輔助功能：
1. CNN 概念解釋器 - 用通俗易懂的語言解釋複雜概念
2. 代碼生成助手 - 根據需求生成代碼模板
3. 錯誤診斷工具 - 智能識別和解決常見錯誤
4. 學習路徑推薦 - 根據學習進度推薦下一步
5. 互動式問答系統

作者：AI Learning Notes
日期：2024-11
"""

import torch
import torch.nn as nn
import re
from typing import Dict, List, Tuple, Optional


class CNN Concept Explainer:
    """
    CNN 概念解釋器

    用簡單易懂的語言解釋 CNN 的各種概念
    """

    def __init__(self):
        """初始化概念庫"""
        self.concepts = {
            'convolution': {
                'title': '卷積運算',
                'simple': '卷積就像用一個小窗口在圖片上滑動，每次提取一小塊特徵',
                'detailed': '''
                卷積是 CNN 的核心操作：
                1. 濾波器（卷積核）在輸入上滑動
                2. 每個位置進行元素乘法並求和
                3. 生成一個新的特徵圖
                4. 可以檢測邊緣、紋理等特徵
                ''',
                'analogy': '就像用篩子篩麵粉，濾波器就是篩子，可以篩選出特定的特徵',
                'formula': 'Output[i,j] = Σ Σ Input[i+a,j+b] × Kernel[a,b]'
            },

            'pooling': {
                'title': '池化層',
                'simple': '池化就像把圖片縮小，保留最重要的信息',
                'detailed': '''
                池化的作用：
                1. 降低特徵圖的空間維度
                2. 減少參數和計算量
                3. 提供平移不變性
                4. 防止過擬合

                常見類型：
                - 最大池化：取窗口內最大值
                - 平均池化：取窗口內平均值
                ''',
                'analogy': '就像把高清照片壓縮成縮略圖，尺寸變小但重要內容都在',
                'formula': 'MaxPool: Output = max(Window), AvgPool: Output = mean(Window)'
            },

            'padding': {
                'title': '填充',
                'simple': '填充就是在圖片四周加一圈，避免邊緣信息丟失',
                'detailed': '''
                填充的作用：
                1. 保持輸出尺寸
                2. 保護邊緣信息
                3. 控制特徵圖大小

                常見策略：
                - Zero Padding：填充0
                - Same Padding：保持尺寸不變
                - Valid Padding：不填充
                ''',
                'analogy': '就像給相框加邊框，讓邊緣的圖案也能完整展示',
                'formula': 'Output_size = (Input_size + 2×Padding - Kernel_size) / Stride + 1'
            },

            'stride': {
                'title': '步幅',
                'simple': '步幅是卷積核每次移動的距離，步幅越大輸出越小',
                'detailed': '''
                步幅的影響：
                1. 控制輸出大小
                2. 影響感受野
                3. 減少計算量
                4. 實現下採樣

                選擇建議：
                - 步幅=1：保持細節
                - 步幅=2：快速降維
                - 步幅>2：激進下採樣
                ''',
                'analogy': '就像走路時的步子大小，大步走得快但可能錯過細節',
                'formula': 'Output_size = ⌊(Input_size - Kernel_size) / Stride⌋ + 1'
            },

            'receptive_field': {
                'title': '感受野',
                'simple': '感受野是神經元能"看到"的輸入區域範圍',
                'detailed': '''
                感受野的重要性：
                1. 決定了能捕捉的特徵尺度
                2. 深層神經元感受野更大
                3. 影響模型的表達能力
                4. 與網絡深度和核大小相關

                計算方法：
                - 每層感受野疊加
                - 考慮池化層的影響
                - 考慮步幅的影響
                ''',
                'analogy': '就像視野範圍，離得越遠（深層）能看到的範圍越大',
                'formula': 'RF(l) = RF(l-1) + (k-1) × Πstrides'
            },

            'batch_normalization': {
                'title': '批量歸一化',
                'simple': '批量歸一化讓每層的輸入保持在合適的範圍內',
                'detailed': '''
                批量歸一化的好處：
                1. 加速訓練收斂
                2. 允許更大的學習率
                3. 減少對初始化的依賴
                4. 有輕微的正則化效果

                工作原理：
                - 計算批次的均值和方差
                - 標準化到均值0方差1
                - 學習縮放和偏移參數
                ''',
                'analogy': '就像調音，確保每個音符都在合適的音域內',
                'formula': 'y = γ × (x - μ) / √(σ² + ε) + β'
            },

            'transfer_learning': {
                'title': '遷移學習',
                'simple': '遷移學習是用別人訓練好的模型來解決自己的問題',
                'detailed': '''
                遷移學習的優勢：
                1. 節省訓練時間
                2. 需要更少的數據
                3. 性能通常更好
                4. 適合小數據集

                兩種策略：
                1. 特徵提取：凍結預訓練層
                2. 微調：解凍並微調部分層
                ''',
                'analogy': '就像學會騎自行車後，學騎摩托車會容易很多',
                'formula': 'New_Model = Pretrained_Backbone + Custom_Classifier'
            }
        }

    def explain(self, concept: str, level: str = 'simple') -> str:
        """
        解釋概念

        Args:
            concept: 概念名稱
            level: 解釋級別 ('simple', 'detailed', 'analogy', 'formula')

        Returns:
            解釋文本
        """
        concept = concept.lower().replace(' ', '_')

        if concept not in self.concepts:
            return f"抱歉，我還不了解 '{concept}' 這個概念。\n可以嘗試: {', '.join(self.concepts.keys())}"

        info = self.concepts[concept]
        result = f"\n{'='*60}\n"
        result += f"{info['title']}\n"
        result += f"{'='*60}\n\n"

        if level == 'simple':
            result += f"💡 簡單理解：\n{info['simple']}\n"
        elif level == 'detailed':
            result += f"📚 詳細說明：\n{info['detailed']}\n"
        elif level == 'analogy':
            result += f"🎯 生活類比：\n{info['analogy']}\n"
        elif level == 'formula':
            result += f"📐 數學公式：\n{info['formula']}\n"
        else:
            # 顯示所有級別
            result += f"💡 簡單理解：\n{info['simple']}\n\n"
            result += f"📚 詳細說明：\n{info['detailed']}\n\n"
            result += f"🎯 生活類比：\n{info['analogy']}\n\n"
            result += f"📐 數學公式：\n{info['formula']}\n"

        return result

    def list_concepts(self) -> str:
        """列出所有可用的概念"""
        result = "\n可用的 CNN 概念:\n"
        result += "="*60 + "\n"
        for i, (key, value) in enumerate(self.concepts.items(), 1):
            result += f"{i}. {value['title']} ({key})\n"
        return result


class CodeGenerator:
    """
    代碼生成助手

    根據需求生成常用的 CNN 代碼模板
    """

    def __init__(self):
        """初始化代碼模板"""
        self.templates = {}

    def generate_simple_cnn(self, num_classes: int = 10, input_channels: int = 3) -> str:
        """生成簡單 CNN 模型"""
        code = f'''
import torch.nn as nn

class SimpleCNN(nn.Module):
    """簡單的 CNN 模型"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷積層
        self.features = nn.Sequential(
            nn.Conv2d({input_channels}, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 分類層
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, {num_classes})
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 使用示例
model = SimpleCNN()
print(model)
'''
        return code

    def generate_training_loop(self) -> str:
        """生成訓練循環代碼"""
        code = '''
def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向傳播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播
        loss.backward()
        optimizer.step()

        # 統計
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

# 使用示例
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f'Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
'''
        return code

    def generate_data_augmentation(self) -> str:
        """生成數據增強代碼"""
        code = '''
from torchvision import transforms

# 訓練集數據增強
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 測試集轉換
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
'''
        return code


class ErrorDiagnostic:
    """
    錯誤診斷工具

    智能識別和解決常見的 CNN 訓練錯誤
    """

    def __init__(self):
        """初始化錯誤模式庫"""
        self.error_patterns = {
            'dimension_mismatch': {
                'pattern': r'size mismatch|dimension|shape',
                'diagnosis': '維度不匹配錯誤',
                'causes': [
                    '1. 輸入圖像尺寸與模型期望不符',
                    '2. 卷積層輸出尺寸計算錯誤',
                    '3. 全連接層輸入維度設置錯誤'
                ],
                'solutions': [
                    '✓ 檢查輸入圖像的尺寸',
                    '✓ 使用 x.view(x.size(0), -1) 自動計算展平維度',
                    '✓ 打印每層輸出形狀進行調試'
                ]
            },

            'out_of_memory': {
                'pattern': r'out of memory|cuda|oom',
                'diagnosis': '內存不足錯誤',
                'causes': [
                    '1. 批量大小設置過大',
                    '2. 模型參數量過多',
                    '3. 圖像分辨率過高'
                ],
                'solutions': [
                    '✓ 減小批量大小（batch_size）',
                    '✓ 使用梯度累積技術',
                    '✓ 降低圖像分辨率',
                    '✓ 使用混合精度訓練'
                ]
            },

            'nan_loss': {
                'pattern': r'nan|inf',
                'diagnosis': 'NaN 或 Inf 損失',
                'causes': [
                    '1. 學習率設置過大',
                    '2. 梯度爆炸',
                    '3. 數值不穩定'
                ],
                'solutions': [
                    '✓ 降低學習率（建議從0.001開始）',
                    '✓ 使用梯度裁剪',
                    '✓ 檢查數據是否包含異常值',
                    '✓ 使用批量歸一化'
                ]
            },

            'no_learning': {
                'pattern': r'not converging|not learning',
                'diagnosis': '模型不學習',
                'causes': [
                    '1. 學習率設置過小',
                    '2. 權重初始化不當',
                    '3. 梯度消失'
                ],
                'solutions': [
                    '✓ 增大學習率',
                    '✓ 使用Xavier或He初始化',
                    '✓ 使用ReLU激活函數',
                    '✓ 添加批量歸一化',
                    '✓ 使用殘差連接'
                ]
            }
        }

    def diagnose(self, error_message: str) -> str:
        """
        診斷錯誤

        Args:
            error_message: 錯誤信息

        Returns:
            診斷結果和解決方案
        """
        result = "\n🔍 錯誤診斷\n"
        result += "="*60 + "\n\n"

        found = False
        for error_type, info in self.error_patterns.items():
            if re.search(info['pattern'], error_message, re.IGNORECASE):
                found = True
                result += f"📋 診斷：{info['diagnosis']}\n\n"
                result += "可能原因：\n"
                for cause in info['causes']:
                    result += f"  {cause}\n"
                result += "\n解決方案：\n"
                for solution in info['solutions']:
                    result += f"  {solution}\n"
                result += "\n"
                break

        if not found:
            result += "❌ 未能識別此錯誤類型\n"
            result += "建議：\n"
            result += "  1. 檢查錯誤堆棧trace\n"
            result += "  2. 搜索錯誤信息\n"
            result += "  3. 檢查 PyTorch 文檔\n"

        return result


class LearningPathRecommender:
    """
    學習路徑推薦器

    根據學習進度推薦下一步學習內容
    """

    def __init__(self):
        """初始化學習路徑"""
        self.path = {
            'beginner': {
                'completed': [],
                'current': '基礎理論',
                'next': [
                    '1. 理解卷積運算的原理',
                    '2. 學習填充和步幅',
                    '3. 實現簡單的卷積層',
                    '4. 理解池化層的作用'
                ],
                'resources': [
                    '📚 閱讀: 1_why-conv.ipynb',
                    '📚 閱讀: 2_conv-layer.ipynb',
                    '💻 實踐: 從零實現卷積運算'
                ]
            },

            'intermediate': {
                'completed': ['基礎理論'],
                'current': '經典架構',
                'next': [
                    '1. 實現 LeNet',
                    '2. 理解 AlexNet 的創新',
                    '3. 學習 VGG 的設計思想',
                    '4. 掌握 ResNet 的殘差結構'
                ],
                'resources': [
                    '📚 閱讀: 6_lenet.ipynb',
                    '💻 實踐: 訓練 LeNet',
                    '🎯 項目: CIFAR-10 分類'
                ]
            },

            'advanced': {
                'completed': ['基礎理論', '經典架構'],
                'current': '現代技術',
                'next': [
                    '1. 學習遷移學習',
                    '2. 掌握數據增強技術',
                    '3. 理解 Attention 機制',
                    '4. 探索輕量級網絡'
                ],
                'resources': [
                    '📚 閱讀: 10_transfer_learning.py',
                    '💻 實踐: 使用預訓練模型',
                    '🎯 項目: 自定義數據集分類'
                ]
            }
        }

    def recommend(self, level: str = 'beginner') -> str:
        """
        推薦學習內容

        Args:
            level: 學習水平 ('beginner', 'intermediate', 'advanced')

        Returns:
            推薦內容
        """
        if level not in self.path:
            return "未知的學習水平，請選擇: beginner, intermediate, advanced"

        info = self.path[level]
        result = f"\n📖 {level.upper()} 學習路徑\n"
        result += "="*60 + "\n\n"

        if info['completed']:
            result += "✅ 已完成：\n"
            for item in info['completed']:
                result += f"  • {item}\n"
            result += "\n"

        result += f"🎯 當前階段：{info['current']}\n\n"

        result += "📝 下一步學習：\n"
        for item in info['next']:
            result += f"  {item}\n"
        result += "\n"

        result += "📚 推薦資源：\n"
        for resource in info['resources']:
            result += f"  {resource}\n"

        return result


# ==================== 交互式 AI 助手 ====================

class CNNAssistant:
    """
    CNN 學習 AI 助手

    整合所有輔助功能的統一接口
    """

    def __init__(self):
        """初始化 AI 助手"""
        self.explainer = CNNConceptExplainer()
        self.code_gen = CodeGenerator()
        self.diagnostic = ErrorDiagnostic()
        self.recommender = LearningPathRecommender()

        print("🤖 CNN 學習 AI 助手已啟動")
        print("="*60)
        print("我可以幫你：")
        print("  1. 解釋 CNN 概念")
        print("  2. 生成代碼模板")
        print("  3. 診斷錯誤")
        print("  4. 推薦學習路徑")
        print("\n輸入 'help' 查看可用命令")
        print("="*60)

    def process_command(self, command: str) -> str:
        """
        處理用戶命令

        Args:
            command: 用戶輸入的命令

        Returns:
            響應內容
        """
        command = command.strip().lower()

        if command == 'help':
            return self._show_help()
        elif command.startswith('explain '):
            concept = command[8:].strip()
            return self.explainer.explain(concept)
        elif command == 'list concepts':
            return self.explainer.list_concepts()
        elif command.startswith('generate '):
            code_type = command[9:].strip()
            return self._generate_code(code_type)
        elif command.startswith('diagnose '):
            error = command[9:].strip()
            return self.diagnostic.diagnose(error)
        elif command.startswith('recommend '):
            level = command[10:].strip()
            return self.recommender.recommend(level)
        else:
            return "❌ 未知命令。輸入 'help' 查看可用命令。"

    def _show_help(self) -> str:
        """顯示幫助信息"""
        help_text = "\n📘 可用命令:\n"
        help_text += "="*60 + "\n"
        help_text += "  explain <concept>     - 解釋 CNN 概念\n"
        help_text += "  list concepts         - 列出所有可用概念\n"
        help_text += "  generate <type>       - 生成代碼模板\n"
        help_text += "  diagnose <error>      - 診斷錯誤\n"
        help_text += "  recommend <level>     - 獲取學習建議\n"
        help_text += "\n示例:\n"
        help_text += "  explain convolution\n"
        help_text += "  generate simple_cnn\n"
        help_text += "  diagnose size mismatch\n"
        help_text += "  recommend beginner\n"
        return help_text

    def _generate_code(self, code_type: str) -> str:
        """生成代碼"""
        if code_type == 'simple_cnn':
            return self.code_gen.generate_simple_cnn()
        elif code_type == 'training_loop':
            return self.code_gen.generate_training_loop()
        elif code_type == 'data_augmentation':
            return self.code_gen.generate_data_augmentation()
        else:
            return f"❌ 未知的代碼類型: {code_type}\n可用: simple_cnn, training_loop, data_augmentation"


# ==================== 主程序 ====================

def main():
    """主程序 - 交互式 AI 助手"""
    assistant = CNNAssistant()

    # 演示功能
    print("\n" + "="*60)
    print("演示 AI 助手功能")
    print("="*60)

    # 1. 解釋概念
    print(assistant.process_command('explain convolution'))

    # 2. 生成代碼
    print("\n生成簡單 CNN 模型:")
    print(assistant.process_command('generate simple_cnn'))

    # 3. 診斷錯誤
    print("\n診斷維度錯誤:")
    print(assistant.process_command('diagnose size mismatch error'))

    # 4. 學習建議
    print(assistant.process_command('recommend beginner'))


if __name__ == '__main__':
    main()
