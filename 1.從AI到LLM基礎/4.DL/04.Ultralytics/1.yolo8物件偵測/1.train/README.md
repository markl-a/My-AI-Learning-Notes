# YOLO 模型訓練指南（YOLO11/v10/v9/v8）

> 🎓 **訓練方式：** 手動訓練、AutoDistill 自動訓練
> 🔧 **框架：** Ultralytics YOLO（支援所有版本）
> ⚡ **特色：** 完整的訓練、驗證、推理流程
> 🔄 **最後更新：** 2025-01

---

## 📖 簡介

本目錄包含 **YOLO 系列模型**（YOLO11、YOLOv10、YOLOv9、YOLOv8）訓練的完整教學和範例程式碼。我們提供兩種訓練方式：

1. **手動訓練** - 完全掌控訓練過程，支援所有 YOLO 版本
2. **AutoDistill 自動訓練** - 使用大模型自動標註並訓練

### 版本選擇建議

| 版本 | 訓練速度 | 準確率潛力 | 推薦場景 |
|------|----------|------------|----------|
| **YOLO11** | 快 | 最高 | 新項目首選 |
| **YOLOv10** | 最快 | 高 | 需要快速迭代 |
| **YOLOv9** | 中 | 最高 | 追求極致準確率 |
| **YOLOv8** | 快 | 高 | 穩定生產環境 |

---

## 📁 目錄內容

```
1.train/
├── README.md                                              # 本文件
├── how_to_auto_train_yolov8_model_with_autodistill修改版.ipynb     # AutoDistill 訓練（修改版）
├── how_to_auto_train_yolov8_model_with_autodistill官方demo的中譯版本.ipynb  # 官方範例（中文版）
├── train_yolov8.py                                        # 基礎訓練腳本（即將添加）
├── train_advanced.py                                      # 進階訓練腳本（即將添加）
└── utils/                                                 # 工具函數（即將添加）
```

---

## 🚀 方法一：手動訓練（推薦）

### 1. 環境準備

```bash
# 安裝 Ultralytics
pip install ultralytics

# 安裝額外依賴
pip install tensorboard
pip install matplotlib
pip install pandas
```

### 2. 準備資料集

確保你的資料集結構如下：

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

**dataset.yaml 範例：**
```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 2
names: ['class1', 'class2']
```

### 3. 基礎訓練

#### Python API

```python
from ultralytics import YOLO

# 載入預訓練模型（選擇您需要的版本）
# YOLO11（推薦）- 最新、最快、最準確
model = YOLO('yolo11n.pt')  # n, s, m, l, x

# 也可以使用其他版本
# model = YOLO('yolov10n.pt')  # 超低延遲
# model = YOLO('yolov9c.pt')   # 最高準確率
# model = YOLO('yolov8n.pt')   # 穩定可靠

# 訓練模型
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU ID，使用 CPU 設為 'cpu'
    project='runs/detect',
    name='my_experiment',
)

# 驗證模型
metrics = model.val()

# 查看結果
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
```

**不同版本的特定優化：**

```python
# YOLO11 - 平衡型配置
model = YOLO('yolo11s.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer='AdamW',  # YOLO11 推薦
    lr0=0.01,
    patience=50
)

# YOLOv10 - 速度優化配置
model = YOLO('yolov10s.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,  # 可以用更大的 batch
    amp=True,  # 自動混合精度
    cache=True  # 快取資料
)

# YOLOv9 - 準確率優化配置
model = YOLO('yolov9c.pt')
results = model.train(
    data='dataset.yaml',
    epochs=200,  # 更多訓練輪數
    imgsz=640,
    batch=16,
    patience=100,  # 更長的耐心值
    mosaic=1.0,
    mixup=0.15  # 使用 MixUp
)

# YOLOv8 - 生產環境配置
model = YOLO('yolov8s.pt')
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    save_period=10,
    plots=True
)
```

#### 命令列

```bash
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=runs/detect \
    name=my_experiment
```

---

### 4. 進階訓練配置

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    # ===== 基本參數 =====
    data='dataset.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    device=0,

    # ===== 優化器參數 =====
    optimizer='AdamW',      # SGD, Adam, AdamW, RMSProp
    lr0=0.01,               # 初始學習率
    lrf=0.01,               # 最終學習率（lr0 * lrf）
    momentum=0.937,         # SGD 動量/Adam beta1
    weight_decay=0.0005,    # 權重衰減
    warmup_epochs=3.0,      # 熱身輪數
    warmup_momentum=0.8,    # 熱身初始動量
    warmup_bias_lr=0.1,     # 熱身初始偏置學習率

    # ===== 資料增強 =====
    hsv_h=0.015,            # HSV-Hue 增強（0.0-1.0）
    hsv_s=0.7,              # HSV-Saturation 增強
    hsv_v=0.4,              # HSV-Value 增強
    degrees=0.0,            # 旋轉角度（+/- deg）
    translate=0.1,          # 平移（+/- fraction）
    scale=0.5,              # 縮放（gain）
    shear=0.0,              # 剪切（+/- deg）
    perspective=0.0,        # 透視變換（0.0-0.001）
    flipud=0.0,             # 垂直翻轉機率
    fliplr=0.5,             # 水平翻轉機率
    mosaic=1.0,             # Mosaic 增強機率
    mixup=0.0,              # MixUp 增強機率
    copy_paste=0.0,         # Copy-Paste 增強機率

    # ===== 進階參數 =====
    patience=50,            # EarlyStopping 耐心值（輪數）
    save=True,              # 儲存檢查點
    save_period=10,         # 每 N 輪儲存一次
    cache=False,            # True/ram/disk - 快取圖像
    workers=8,              # 資料載入器工作線程數
    pretrained=True,        # 使用預訓練權重
    verbose=True,           # 詳細輸出
    seed=0,                 # 隨機種子

    # ===== 驗證參數 =====
    val=True,               # 訓練期間驗證
    plots=True,             # 生成訓練圖表
    rect=False,             # 矩形訓練
    cos_lr=False,           # 使用餘弦學習率調度器
    close_mosaic=10,        # 最後 N 輪關閉 Mosaic

    # ===== 推理參數 =====
    conf=0.001,             # 物件置信度閾值
    iou=0.7,                # NMS IoU 閾值
    max_det=300,            # 每張圖最大檢測數

    # ===== 損失函式權重 =====
    box=7.5,                # box loss gain
    cls=0.5,                # cls loss gain
    dfl=1.5,                # dfl loss gain

    # ===== 輸出 =====
    project='runs/detect',
    name='experiment',
    exist_ok=False,         # 覆蓋現有專案
)
```

---

### 5. 訓練腳本範例

建立 `train_yolov8.py`：

```python
#!/usr/bin/env python3
"""
YOLOv8 訓練腳本
使用方式: python train_yolov8.py --data dataset.yaml --model yolov8n.pt
"""

import argparse
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--project', type=str, default='runs/detect', help='project name')
    parser.add_argument('--name', type=str, default='train', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    return parser.parse_args()

def main():
    args = parse_args()

    # 載入模型
    if args.resume:
        model = YOLO(f'{args.project}/{args.name}/weights/last.pt')
        print(f"📦 恢復訓練: {args.project}/{args.name}")
    else:
        model = YOLO(args.model)
        print(f"📦 載入模型: {args.model}")

    # 訓練
    print(f"🚀 開始訓練...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.resume,

        # 進階參數
        optimizer='AdamW',
        lr0=0.01,
        patience=50,
        save=True,
        save_period=10,
        plots=True,
    )

    # 驗證
    print(f"📊 驗證模型...")
    metrics = model.val()

    # 輸出結果
    print(f"\n✅ 訓練完成！")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"\n模型儲存於: {args.project}/{args.name}/weights/best.pt")

if __name__ == '__main__':
    main()
```

**使用方式：**

```bash
# 基礎訓練
python train_yolov8.py --data dataset.yaml --model yolov8n.pt --epochs 100

# 自定義參數
python train_yolov8.py \
    --data dataset.yaml \
    --model yolov8s.pt \
    --epochs 300 \
    --batch 32 \
    --imgsz 1280 \
    --device 0 \
    --name my_experiment

# 恢復訓練
python train_yolov8.py --data dataset.yaml --resume
```

---

## 🤖 方法二：AutoDistill 自動訓練

AutoDistill 使用大型基礎模型（如 DINO、SAM）自動標註資料，然後訓練 YOLOv8 模型。

### 1. 安裝 AutoDistill

```bash
pip install autodistill
pip install autodistill-grounded-sam
pip install autodistill-yolov8
```

### 2. 自動標註和訓練

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# 定義類別
ontology = CaptionOntology({
    "cat": "cat",
    "dog": "dog"
})

# 使用 Grounded-SAM 自動標註
base_model = GroundedSAM(ontology=ontology)

# 標註圖像資料夾
base_model.label(
    input_folder="./raw_images",
    output_folder="./dataset"
)

# 訓練 YOLOv8
target_model = YOLOv8("yolov8n.pt")
target_model.train("./dataset/data.yaml", epochs=100)

# 評估
metrics = target_model.model.val()
print(f"mAP50: {metrics.box.map50}")
```

### 3. Notebook 範例

本目錄包含兩個 AutoDistill Notebook：

1. **官方 demo 中譯版本** - 原始官方範例的中文翻譯
2. **修改版** - 針對實際使用場景的優化版本

**使用建議：**
- 初學者：先閱讀官方範例理解流程
- 實戰使用：使用修改版進行實際訓練

---

## 📊 訓練監控

### 1. TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir runs/detect

# 訪問 http://localhost:6006
```

### 2. 查看訓練曲線

訓練完成後，在 `runs/detect/experiment/` 會生成：

- `results.png` - 訓練曲線（loss, mAP, precision, recall）
- `confusion_matrix.png` - 混淆矩陣
- `F1_curve.png` - F1 曲線
- `PR_curve.png` - Precision-Recall 曲線
- `P_curve.png` - Precision 曲線
- `R_curve.png` - Recall 曲線

### 3. 程式化讀取結果

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取訓練結果
results = pd.read_csv('runs/detect/experiment/results.csv')

# 繪製損失曲線
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(results['train/box_loss'], label='Train Box Loss')
plt.plot(results['val/box_loss'], label='Val Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Box Loss')

plt.subplot(2, 2, 2)
plt.plot(results['metrics/mAP50(B)'])
plt.xlabel('Epoch')
plt.ylabel('mAP50')
plt.title('mAP50')

plt.subplot(2, 2, 3)
plt.plot(results['metrics/precision(B)'], label='Precision')
plt.plot(results['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.title('Precision & Recall')

plt.subplot(2, 2, 4)
plt.plot(results['lr/pg0'])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')

plt.tight_layout()
plt.savefig('training_curves.png')
```

---

## 🎯 訓練技巧

### 1. 選擇合適的模型大小

| 場景 | 推薦模型 | 理由 |
|------|----------|------|
| 邊緣裝置 | YOLOv8n | 速度最快，體積最小 |
| 移動應用 | YOLOv8s | 平衡性能與速度 |
| GPU 伺服器 | YOLOv8m/l | 高準確率 |
| 研究/競賽 | YOLOv8x | 最高準確率 |

### 2. 學習率調整

```python
# 找最佳學習率
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.tune(data='dataset.yaml', iterations=300)
```

### 3. 處理過擬合

**症狀：** 訓練 loss 下降，驗證 loss 上升

**解決方法：**
- 增加資料增強
- 使用 Dropout
- 減小模型大小
- 增加訓練資料
- Early Stopping

```python
results = model.train(
    data='dataset.yaml',
    epochs=300,
    patience=50,  # Early stopping

    # 增強資料增強
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,

    # 正則化
    weight_decay=0.001,
)
```

### 4. 處理欠擬合

**症狀：** 訓練和驗證 loss 都很高

**解決方法：**
- 使用更大的模型
- 增加訓練輪數
- 提高學習率
- 減少正則化

```python
results = model.train(
    data='dataset.yaml',
    model='yolov8l.pt',  # 使用更大模型
    epochs=500,
    lr0=0.02,  # 提高學習率
    weight_decay=0.0001,  # 減少正則化
)
```

### 5. 小物件檢測

```python
results = model.train(
    data='dataset.yaml',
    imgsz=1280,  # 提高解析度
    mosaic=1.0,
    copy_paste=0.3,  # 複製貼上小物件
)
```

---

## 🔧 常見問題

### Q: CUDA out of memory？

**A:**
```python
# 減小批次大小
batch=8  # 或 4, 2

# 使用較小模型
model = YOLO('yolov8n.pt')

# 降低解析度
imgsz=480
```

### Q: 訓練速度太慢？

**A:**
```python
# 使用多 GPU
device=[0,1,2,3]

# 快取資料集到 RAM
cache='ram'

# 增加 workers
workers=16

# 使用 AMP（自動混合精度）
amp=True
```

### Q: mAP 不上升？

**A:**
- 檢查資料集標註品質
- 增加訓練輪數
- 調整學習率
- 檢查資料增強是否過度
- 使用預訓練模型

---

## 📚 學習資源

### 官方文檔
- [Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/)
- [Hyperparameter Tuning](https://docs.ultralytics.com/guides/hyperparameter-tuning/)

### 教學文章
- [YOLOv8 訓練最佳實踐](https://docs.ultralytics.com/guides/model-training-tips/)

### 相關工具
- [Weights & Biases](https://wandb.ai/) - 實驗追蹤
- [Comet ML](https://www.comet.ml/) - 模型管理
- [ClearML](https://clear.ml/) - MLOps 平台

---

## 🎓 下一步

訓練完成後：

1. **模型評估** - 在測試集上評估性能
2. **模型優化** - 量化、剪枝、蒸餾
3. **模型部署** - 導出並部署到目標平台
4. **持續改進** - 收集失敗案例，改進資料集

**相關目錄：**
- 資料集準備：`../0.dataset/`
- 模型部署：`../2.deploy/`

---

**祝訓練順利！** 🚀
