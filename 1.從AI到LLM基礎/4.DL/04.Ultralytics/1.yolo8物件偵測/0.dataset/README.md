# YOLO 資料集準備指南（支援 YOLO11/v10/v9/v8）

> 📊 **資料集：** 自定義物件偵測資料集
> 🎯 **格式：** YOLO 格式（txt 標註）
> 🔧 **工具：** LabelImg, Roboflow, CVAT, Label Studio
> 🔄 **最後更新：** 2025-01

---

## 📖 簡介

本目錄包含用於訓練 **YOLO 系列模型**（YOLO11、YOLOv10、YOLOv9、YOLOv8）的資料集準備指南。良好的資料集是訓練高準確率模型的關鍵，所有 YOLO 版本使用相同的資料格式。

### 資料集品質檢查清單

- ✅ 圖像數量充足（建議每類 >500 張）
- ✅ 圖像多樣性（不同角度、光線、背景）
- ✅ 標註準確（邊界框緊貼物件）
- ✅ 類別平衡（各類別數量相近）
- ✅ 資料劃分（訓練:驗證:測試 = 70:20:10）

---

## 📁 資料集結構

### YOLO 標準格式

```
dataset/
├── images/
│   ├── train/              # 訓練集圖像
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/                # 驗證集圖像
│   │   ├── img101.jpg
│   │   └── ...
│   └── test/               # 測試集圖像（可選）
│       └── ...
├── labels/
│   ├── train/              # 訓練集標註
│   │   ├── img001.txt
│   │   ├── img002.txt
│   │   └── ...
│   ├── val/                # 驗證集標註
│   │   ├── img101.txt
│   │   └── ...
│   └── test/               # 測試集標註（可選）
│       └── ...
└── dataset.yaml            # 資料集配置檔案
```

### 標註檔案格式

每個 `.txt` 檔案對應一張圖像，格式如下：

```
<class_id> <x_center> <y_center> <width> <height>
```

**重要說明：**
- 所有座標值都是**相對值**（0-1 之間）
- `x_center`, `y_center`：邊界框中心點相對座標
- `width`, `height`：邊界框寬高相對尺寸
- `class_id`：類別編號（從 0 開始）

**範例：**
```txt
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
0 0.2 0.8 0.15 0.25
```

**計算方式：**
```python
# 假設圖像尺寸為 1920x1080，物件邊界框為 (x1=100, y1=200, x2=500, y2=600)
x_center = ((x1 + x2) / 2) / image_width   # (100+500)/2 / 1920 = 0.156
y_center = ((y1 + y2) / 2) / image_height  # (200+600)/2 / 1080 = 0.370
width = (x2 - x1) / image_width            # (500-100) / 1920 = 0.208
height = (y2 - y1) / image_height          # (600-200) / 1080 = 0.370

# 輸出到 txt：0 0.156 0.370 0.208 0.370
```

---

## 🛠️ 資料標註工具

### 1. LabelImg（推薦初學者）

**特點：**
- ✅ 簡單易用，GUI 介面
- ✅ 原生支援 YOLO 格式
- ✅ 跨平台（Windows/Mac/Linux）

**安裝：**
```bash
pip install labelImg

# 啟動
labelImg
```

**使用步驟：**
1. 點擊「Open Dir」選擇圖像資料夾
2. 點擊「Change Save Dir」選擇標註儲存位置
3. 選擇「YOLO」格式（View → Auto Save mode）
4. 按 `W` 鍵繪製邊界框
5. 輸入類別名稱
6. 按 `D` 鍵切換下一張圖

**快捷鍵：**
- `W`：建立邊界框
- `D`：下一張圖
- `A`：上一張圖
- `Del`：刪除選中的框
- `Ctrl+S`：儲存

---

### 2. Roboflow（推薦進階使用者）⭐

**特點：**
- ✅ 線上平台，無需安裝
- ✅ 自動資料增強
- ✅ 格式自動轉換（支援所有 YOLO 版本）
- ✅ 資料集版本管理
- ✅ 團隊協作
- ✅ AI 輔助標註
- ✅ 資料集健康檢查

**使用流程：**
1. 註冊 [Roboflow](https://roboflow.com/)
2. 建立新專案
3. 上傳圖像
4. 線上標註（支援 AI 輔助）
5. 應用資料增強
6. 導出 YOLO 格式（支援 YOLO11/v10/v9/v8）

**Python API：**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("YOUR_PROJECT")

# 導出為 YOLO 格式（通用於所有版本）
dataset = project.version(1).download("yolov8")  # 格式兼容 YOLO11/v10/v9/v8

# 直接訓練
from ultralytics import YOLO
model = YOLO('yolo11n.pt')  # 或 yolov10n.pt, yolov9c.pt, yolov8n.pt
results = model.train(data=f"{dataset.location}/data.yaml", epochs=100)
```

---

### 3. CVAT（企業級）

**特點：**
- ✅ 功能強大
- ✅ 支援影片標註
- ✅ 多人協作
- ✅ 自動標註（AI 輔助）

**安裝（Docker）：**
```bash
git clone https://github.com/opencv/cvat
cd cvat
docker-compose up -d
```

訪問：`http://localhost:8080`

---

### 4. Label Studio

**特點：**
- ✅ 開源、免費
- ✅ 支援多種任務類型
- ✅ 可自訂標註介面

**安裝：**
```bash
pip install label-studio

# 啟動
label-studio start
```

---

## 📊 資料集品質提升

### 1. 資料收集建議

**數量要求：**
- 小型專案：每類 200-500 張
- 中型專案：每類 500-2000 張
- 大型專案：每類 2000+ 張

**多樣性要求：**
- ✅ 不同角度（正面、側面、俯視、仰視）
- ✅ 不同光線（白天、夜晚、室內、室外）
- ✅ 不同背景（簡單、複雜）
- ✅ 不同尺度（遠景、近景）
- ✅ 不同遮擋程度（完全可見、部分遮擋）

### 2. 標註品質檢查

**檢查要點：**
- ✅ 邊界框緊貼物件
- ✅ 無遺漏的物件
- ✅ 類別標註正確
- ✅ 無重複標註

**自動檢查腳本：**
```python
import os
from pathlib import Path

def check_labels(label_dir, image_dir):
    """檢查標註檔案"""
    label_files = set(Path(label_dir).glob('*.txt'))
    image_files = set(Path(image_dir).glob('*.jpg'))

    # 檢查是否有圖像沒有對應的標註
    image_names = {f.stem for f in image_files}
    label_names = {f.stem for f in label_files}

    missing_labels = image_names - label_names
    if missing_labels:
        print(f"⚠️  缺少標註的圖像: {len(missing_labels)}")
        print(list(missing_labels)[:5])

    # 檢查標註格式
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"❌ {label_file.name} 第 {i+1} 行格式錯誤")
                    continue

                # 檢查座標範圍
                class_id, x, y, w, h = map(float, parts)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"❌ {label_file.name} 座標超出範圍 [0,1]")

    print("✅ 標註檢查完成")

# 使用範例
check_labels('labels/train', 'images/train')
```

### 3. 資料劃分

**建議比例：**
- 訓練集（Train）：70-80%
- 驗證集（Val）：15-20%
- 測試集（Test）：10-15%

**自動劃分腳本：**
```python
import os
import shutil
import random
from pathlib import Path

def split_dataset(image_dir, label_dir, output_dir, split_ratio=(0.7, 0.2, 0.1)):
    """
    劃分資料集
    Args:
        image_dir: 原始圖像目錄
        label_dir: 原始標註目錄
        output_dir: 輸出目錄
        split_ratio: (train, val, test) 比例
    """
    # 建立輸出目錄
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    # 獲取所有圖像檔案
    image_files = list(Path(image_dir).glob('*.jpg')) + \
                  list(Path(image_dir).glob('*.png'))

    # 隨機打亂
    random.shuffle(image_files)

    # 計算分割點
    total = len(image_files)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    # 分割資料
    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    # 複製檔案
    for split_name, files in splits.items():
        for img_file in files:
            # 複製圖像
            shutil.copy(img_file, f"{output_dir}/images/{split_name}/")

            # 複製標註
            label_file = Path(label_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, f"{output_dir}/labels/{split_name}/")

    print(f"✅ 資料集劃分完成:")
    print(f"   訓練集: {len(splits['train'])} 張")
    print(f"   驗證集: {len(splits['val'])} 張")
    print(f"   測試集: {len(splits['test'])} 張")

# 使用範例
split_dataset(
    image_dir='raw_images',
    label_dir='raw_labels',
    output_dir='dataset',
    split_ratio=(0.7, 0.2, 0.1)
)
```

---

## 🔄 資料增強

資料增強可以提升模型的泛化能力。YOLOv8 訓練時會自動應用一些增強，但也可以手動預處理。

### YOLOv8 內建增強

在訓練時自動應用：
- Mosaic（馬賽克）
- MixUp
- HSV 調整
- 隨機翻轉
- 隨機旋轉
- 隨機縮放

### 手動資料增強

使用 Albumentations 庫：

```python
import albumentations as A
import cv2

# 定義增強
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.1),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 讀取圖像
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 讀取標註（YOLO 格式）
with open('image.txt', 'r') as f:
    lines = f.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        class_id, x, y, w, h = map(float, line.strip().split())
        bboxes.append([x, y, w, h])
        class_labels.append(int(class_id))

# 應用增強
transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
augmented_image = transformed['image']
augmented_bboxes = transformed['bboxes']
```

---

## 📝 建立 dataset.yaml

每個資料集都需要一個 YAML 配置檔案：

```yaml
# dataset.yaml

# 資料集路徑（可使用絕對路徑或相對路徑）
path: /path/to/dataset  # 資料集根目錄
train: images/train     # 訓練集圖像路徑（相對於 path）
val: images/val         # 驗證集圖像路徑
test: images/test       # 測試集圖像路徑（可選）

# 類別資訊
nc: 2                   # 類別數量
names: ['cat', 'dog']   # 類別名稱列表（索引對應 class_id）
```

**完整範例：**

```yaml
# My Custom Dataset
path: /home/user/datasets/my_dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 4
names:
  0: person
  1: car
  2: bicycle
  3: motorcycle

# 可選資訊
download: https://example.com/dataset.zip
```

---

## 🔍 資料集統計分析

建議在訓練前分析資料集統計資訊：

```python
import os
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_dataset(label_dir, class_names):
    """分析資料集統計資訊"""
    label_files = list(Path(label_dir).glob('*.txt'))

    class_counts = Counter()
    bbox_sizes = []

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                class_counts[int(class_id)] += 1
                bbox_sizes.append((w, h))

    # 列印統計資訊
    print(f"總圖像數: {len(label_files)}")
    print(f"總物件數: {sum(class_counts.values())}")
    print(f"平均每張圖 {sum(class_counts.values())/len(label_files):.2f} 個物件")
    print("\n類別分布:")
    for class_id, count in class_counts.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        print(f"  {class_name}: {count} ({count/sum(class_counts.values())*100:.1f}%)")

    # 視覺化
    plt.figure(figsize=(10, 4))

    # 類別分布
    plt.subplot(1, 2, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title('Class Distribution')

    # 邊界框大小分布
    plt.subplot(1, 2, 2)
    widths, heights = zip(*bbox_sizes)
    plt.scatter(widths, heights, alpha=0.3)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Bounding Box Size Distribution')

    plt.tight_layout()
    plt.savefig('dataset_analysis.png')
    print("\n✅ 統計圖表已儲存到 dataset_analysis.png")

# 使用範例
analyze_dataset('labels/train', ['cat', 'dog', 'bird'])
```

---

## 📚 常見問題

### Q: 圖像格式有限制嗎？

**A:** 支援常見格式：
- ✅ JPG/JPEG
- ✅ PNG
- ✅ BMP
- ✅ TIFF

建議使用 **JPG** 格式（檔案小、載入快）

### Q: 圖像大小需要統一嗎？

**A:** 不需要。YOLOv8 會自動調整到訓練尺寸（預設 640x640）

### Q: 標註時邊界框可以超出圖像嗎？

**A:** 不可以。所有座標值必須在 [0, 1] 範圍內。

### Q: 如何處理小物件？

**A:**
- 使用更高的訓練解析度（如 1280）
- 增加小物件的訓練樣本
- 使用資料增強
- 調整模型架構（添加 P2 層）

### Q: 資料不平衡怎麼辦？

**A:**
- 對少數類別進行資料增強
- 使用類別權重
- 過採樣少數類別
- 欠採樣多數類別

---

## 🎯 資料集品質檢查清單

在開始訓練前，確認以下項目：

- [ ] 圖像和標註檔案命名一致
- [ ] 所有圖像都有對應的標註檔案
- [ ] 標註格式正確（5 個值，座標在 [0,1]）
- [ ] 資料集已正確劃分（train/val/test）
- [ ] dataset.yaml 配置正確
- [ ] 類別數量和名稱正確
- [ ] 執行統計分析，確認無異常
- [ ] 視覺化檢查部分標註，確認準確性

---

## 🔗 相關資源

- [YOLOv8 數據格式文檔](https://docs.ultralytics.com/datasets/)
- [Roboflow 公開資料集](https://universe.roboflow.com/)
- [COCO 資料集](https://cocodataset.org/)
- [Open Images 資料集](https://storage.googleapis.com/openimages/web/index.html)

---

**準備好高品質的資料集，是成功訓練模型的第一步！** 🎯
