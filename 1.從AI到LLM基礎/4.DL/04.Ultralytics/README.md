# Ultralytics YOLOv8 物件偵測實戰

> 🔄 **最後更新：** 2025-01
> 📊 **完成度：** 良好
> 🎯 **YOLO 版本：** YOLOv8（最新版）
> ⭐ **特色：** 端到端實戰項目，從資料準備到模型部署

---

## 📖 簡介

本目錄包含使用 **Ultralytics YOLOv8** 進行物件偵測的完整實戰教學，從自製資料集準備、模型訓練、到實際部署的完整流程。

### 什麼是 YOLO？

**YOLO (You Only Look Once)** 是一種即時物件偵測演算法，以其速度快、準確率高而聞名。YOLOv8 是 Ultralytics 公司推出的最新版本。

### 為什麼選擇 YOLOv8？

- ✅ **速度極快：** 即時物件偵測（>30 FPS）
- ✅ **準確率高：** COCO 資料集上表現優異
- ✅ **易於使用：** 簡潔的 Python API
- ✅ **全面支援：** 檢測、分割、分類、姿態估計
- ✅ **部署便利：** 支援多種平台（iOS、Android、Web）

---

## 📁 專案結構

```
04.Ultralytics/
├── README.md                    # 本文件
└── 1.yolo8物件偵測/
    ├── 1.train/                 # 訓練相關
    │   ├── 資料集準備
    │   ├── 模型訓練
    │   └── 模型評估
    ├── 2.deploy/                # 部署相關
    │   ├── Android/             # Android 部署
    │   └── iOS/                 # iOS 部署（包含完整範例）
    └── 訓練結果與模型
```

---

## 🚀 快速開始

### 安裝 Ultralytics

```bash
# 使用 pip 安裝
pip install ultralytics

# 驗證安裝
yolo version

# 或使用 Python
python -c "from ultralytics import YOLO; print('Success!')"
```

### 基本使用

```python
from ultralytics import YOLO

# 1. 載入預訓練模型
model = YOLO('yolov8n.pt')  # n (nano), s (small), m (medium), l (large), x (xlarge)

# 2. 訓練模型
results = model.train(
    data='coco128.yaml',  # 資料集配置
    epochs=100,           # 訓練輪數
    imgsz=640,            # 圖像大小
    device=0              # GPU ID（-1 為 CPU）
)

# 3. 驗證模型
metrics = model.val()

# 4. 進行預測
results = model('image.jpg')
results[0].show()  # 顯示結果

# 5. 導出模型
model.export(format='onnx')  # 導出為 ONNX 格式
```

---

## 📚 完整教學流程

### 1️⃣ 資料集準備

#### 資料集格式

YOLOv8 使用 **YOLO 格式** 的標註：

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt
```

#### 標註檔案格式

每個 `.txt` 檔案包含該圖像的所有物件標註：

```
# 格式：<class_id> <x_center> <y_center> <width> <height>
# 所有座標都是相對值（0-1）
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```

#### 資料集配置檔案（YAML）

```yaml
# dataset.yaml
path: /path/to/dataset  # 資料集根目錄
train: images/train     # 訓練圖像路徑
val: images/val         # 驗證圖像路徑

# 類別
nc: 2                   # 類別數量
names: ['cat', 'dog']   # 類別名稱
```

#### 推薦標註工具

- [LabelImg](https://github.com/heartexlabs/labelImg)（YOLO 格式直接支援）
- [Roboflow](https://roboflow.com/)（線上標註，自動格式轉換）
- [CVAT](https://github.com/opencv/cvat)（功能強大）
- [Label Studio](https://labelstud.io/)（多功能）

---

### 2️⃣ 模型訓練

#### 選擇模型大小

| 模型 | 參數量 | 速度 | 準確率 | 適用場景 |
|------|--------|------|--------|----------|
| YOLOv8n | 3.2M | 最快 | 中等 | 邊緣裝置、即時應用 |
| YOLOv8s | 11.2M | 快 | 良好 | 平衡性能與速度 |
| YOLOv8m | 25.9M | 中等 | 高 | 準確率優先 |
| YOLOv8l | 43.7M | 慢 | 很高 | 高準確率需求 |
| YOLOv8x | 68.2M | 最慢 | 最高 | 競賽、研究 |

#### 訓練範例

```python
from ultralytics import YOLO

# 載入預訓練模型
model = YOLO('yolov8n.pt')

# 訓練
results = model.train(
    data='dataset.yaml',      # 資料集配置
    epochs=100,               # 訓練輪數
    imgsz=640,                # 輸入圖像大小
    batch=16,                 # 批次大小
    device=0,                 # GPU ID
    workers=8,                # 資料載入器線程數

    # 進階參數
    optimizer='AdamW',        # 優化器
    lr0=0.01,                 # 初始學習率
    weight_decay=0.0005,      # 權重衰減
    warmup_epochs=3,          # 熱身輪數

    # 資料增強
    hsv_h=0.015,              # HSV-Hue 增強
    hsv_s=0.7,                # HSV-Saturation 增強
    hsv_v=0.4,                # HSV-Value 增強
    degrees=0.0,              # 旋轉角度
    translate=0.1,            # 平移
    scale=0.5,                # 縮放
    mosaic=1.0,               # Mosaic 增強

    # 儲存選項
    project='runs/detect',    # 儲存專案資料夾
    name='my_experiment',     # 實驗名稱
    save=True,                # 儲存檢查點
    save_period=10,           # 每 N 輪儲存一次
)
```

#### 使用命令列訓練

```bash
yolo detect train \
    data=dataset.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

---

### 3️⃣ 模型評估

```python
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO('runs/detect/my_experiment/weights/best.pt')

# 在驗證集上評估
metrics = model.val()

# 查看指標
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

#### 評估指標說明

- **mAP50：** IoU 閾值為 0.5 的平均精確度
- **mAP50-95：** IoU 閾值從 0.5 到 0.95 的平均精確度
- **Precision：** 精確率（預測為正的樣本中實際為正的比例）
- **Recall：** 召回率（實際為正的樣本中被正確預測為正的比例）

---

### 4️⃣ 推理與預測

```python
from ultralytics import YOLO
import cv2

# 載入模型
model = YOLO('best.pt')

# 單張圖像
results = model('image.jpg')
results[0].show()

# 多張圖像
results = model(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# 影片
results = model('video.mp4', save=True)

# 即時攝影機
results = model(source=0, show=True)  # 0 為預設攝影機

# 獲取預測結果
for result in results:
    boxes = result.boxes          # 邊界框
    masks = result.masks          # 分割遮罩
    keypoints = result.keypoints  # 關鍵點
    probs = result.probs          # 分類機率

    # 取得座標
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # 左上、右下座標
        conf = box.conf[0]             # 信心分數
        cls = box.cls[0]               # 類別
        print(f"Class: {cls}, Confidence: {conf:.2f}")
```

#### 進階推理選項

```python
results = model.predict(
    source='image.jpg',
    conf=0.25,              # 信心閾值
    iou=0.7,                # NMS IoU 閾值
    max_det=300,            # 最大檢測數量
    classes=[0, 1],         # 只檢測特定類別
    device=0,               # GPU
    save=True,              # 儲存結果
    save_txt=True,          # 儲存標籤
    save_conf=True,         # 儲存信心分數
    visualize=False,        # 視覺化特徵
    augment=False,          # 測試時增強
    agnostic_nms=False,     # 類別無關的 NMS
)
```

---

### 5️⃣ 模型部署

#### 導出模型格式

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# 導出為不同格式
model.export(format='onnx')          # ONNX
model.export(format='torchscript')   # TorchScript
model.export(format='coreml')        # CoreML (iOS)
model.export(format='tflite')        # TensorFlow Lite (Android)
model.export(format='engine')        # TensorRT
model.export(format='openvino')      # OpenVINO
```

#### 支援的導出格式

| 格式 | 檔案 | 適用平台 | 速度 | 大小 |
|------|------|----------|------|------|
| PyTorch | `.pt` | Python | 中等 | 大 |
| TorchScript | `.torchscript` | Python/C++ | 快 | 大 |
| ONNX | `.onnx` | 多平台 | 快 | 中等 |
| TensorRT | `.engine` | NVIDIA GPU | 最快 | 中等 |
| CoreML | `.mlmodel` | iOS/macOS | 快 | 小 |
| TFLite | `.tflite` | Android/邊緣 | 快 | 小 |

#### iOS 部署範例

本專案包含完整的 iOS 部署範例：

```
2.deploy/iOS/
├── YOLOv8App/             # iOS 應用程式
├── best.mlmodel           # CoreML 模型
└── README.md              # iOS 部署指南
```

**部署步驟：**
1. 導出 CoreML 模型：`model.export(format='coreml')`
2. 將 `.mlmodel` 加入 Xcode 專案
3. 使用 Vision 框架進行推理

**檔案位置：** `2.deploy/iOS/`

---

## 🎯 實戰技巧

### 提升模型準確率

1. **收集更多資料**
   - 至少每類 1000+ 張圖像
   - 涵蓋不同角度、光線、背景

2. **資料增強**
   - 使用 Mosaic、MixUp
   - 調整 HSV、旋轉、翻轉

3. **調整超參數**
   - 學習率、批次大小
   - IoU 閾值、信心閾值

4. **使用更大的模型**
   - YOLOv8n → YOLOv8s → YOLOv8m

5. **增加訓練輪數**
   - 至少 100 輪
   - 觀察損失曲線，避免過擬合

### 提升推理速度

1. **選擇適當模型**
   - 邊緣裝置使用 YOLOv8n
   - GPU 伺服器使用 YOLOv8s/m

2. **使用 TensorRT**
   - 速度提升 2-5 倍
   - 只支援 NVIDIA GPU

3. **降低輸入解析度**
   - 640 → 480 或 320
   - 犧牲少量準確率換取速度

4. **批次推理**
   - 同時處理多張圖像

### 處理小物件

1. **增加輸入解析度**
   - 640 → 1280

2. **使用 P2 層**
   - 提升小物件檢測能力

3. **調整錨框大小**

---

## 📊 專案範例

### 自定義物件檢測

**應用場景：** 檢測自製資料集中的特定物件

**步驟：**
1. 標註資料集（使用 LabelImg）
2. 建立 `dataset.yaml`
3. 訓練模型
4. 評估並調優
5. 部署到應用程式

**參考：** `1.yolo8物件偵測/1.train/`

### iOS 應用部署

**應用場景：** 在 iPhone 上即時物件偵測

**步驟：**
1. 訓練並導出 CoreML 模型
2. 建立 iOS 專案
3. 整合模型
4. 實作即時推理

**完整範例：** `1.yolo8物件偵測/2.deploy/iOS/`

---

## 🔧 常見問題

### Q: 訓練時 GPU 記憶體不足？

**A:**
- 減小批次大小：`batch=8` 或 `batch=4`
- 使用較小模型：YOLOv8n
- 降低圖像大小：`imgsz=480`

### Q: 模型過擬合？

**A:**
- 增加資料增強
- 使用 Dropout
- 減少訓練輪數
- 收集更多資料

### Q: mAP 很低？

**A:**
- 檢查標註品質
- 增加訓練輪數
- 使用預訓練模型
- 調整資料增強參數

---

## 📚 學習資源

### 官方資源
- [Ultralytics 文檔](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics Hub](https://hub.ultralytics.com/)（線上訓練平台）

### 社群資源
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com/)（公開資料集）
- [Papers with Code - Object Detection](https://paperswithcode.com/task/object-detection)

### 相關論文
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- YOLOv7: [YOLOv7: Trainable bag-of-freebies](https://arxiv.org/abs/2207.02696)

---

## 🎓 學習建議

### 對於初學者：
1. 從預訓練模型開始（COCO）
2. 使用小資料集練習
3. 理解評估指標
4. 完成一個端到端專案

### 對於進階使用者：
1. 自定義架構
2. 多任務學習（檢測 + 分割）
3. 部署優化
4. 貢獻開源社群

---

## 🔗 相關章節

- **深度學習基礎：** 查看 `../00.DL_Path/`
- **PyTorch 基礎：** 查看 `../03.Pytorch/`
- **Transformer 應用：** 查看 `../05.Transformer_lib/`

---

**本專案展示了從資料準備到模型部署的完整物件偵測工作流程，是學習實戰技能的絕佳範例！** 🎯

---

**快樂檢測！** 🚀
