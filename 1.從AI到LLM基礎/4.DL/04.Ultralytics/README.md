# Ultralytics YOLO 物件偵測實戰指南

> 🔄 **最後更新：** 2025-01
> 📊 **完成度：** 優秀
> 🎯 **YOLO 版本：** YOLO11 / YOLOv10 / YOLOv9 / YOLOv8（全系列支援）
> ⭐ **特色：** 最新 YOLO 技術完整教學，從資料準備到模型部署

---

## 📖 簡介

本目錄包含使用 **Ultralytics YOLO 系列**（YOLO11、YOLOv10、YOLOv9、YOLOv8）進行物件偵測的完整實戰教學，從自製資料集準備、模型訓練、到實際部署的完整流程。

### 什麼是 YOLO？

**YOLO (You Only Look Once)** 是一種即時物件偵測演算法，以其速度快、準確率高而聞名。Ultralytics 持續推出最新版本，包括 YOLO11（2024 最新）、YOLOv10、YOLOv9 和 YOLOv8。

### YOLO 版本選擇指南

#### 🚀 **YOLO11（2024 最新，推薦）**
- ✅ **最高性能：** 最佳的速度與準確率平衡
- ✅ **創新架構：** C3k2 和 C2PSA 模塊，增強特徵提取
- ✅ **參數更少：** 比 YOLOv8 更輕量，推理速度更快
- ✅ **多任務支援：** 檢測、分割、姿態估計、OBB
- ✅ **最佳選擇：** 新項目首選，全面超越前代

#### ⚡ **YOLOv10（2024，低延遲優化）**
- ✅ **無需 NMS：** 端到端架構，消除非極大值抑制
- ✅ **超低延遲：** 比 RT-DETR 快 1.8 倍
- ✅ **輕量設計：** 比 YOLOv9 少 25% 參數
- ✅ **適用場景：** 需要極致速度的邊緣設備

#### 🎯 **YOLOv9（2024，高準確率）**
- ✅ **最高 mAP：** 檢測準確率達 0.935
- ✅ **創新技術：** PGI 和 GELAN 架構
- ✅ **精確檢測：** 在複雜場景中表現最佳
- ✅ **適用場景：** 準確率優先的應用

#### 🔧 **YOLOv8（2023，穩定可靠）**
- ✅ **成熟穩定：** 大量實戰驗證
- ✅ **豐富生態：** 最多社群支援和範例
- ✅ **易於使用：** 簡潔的 Python API
- ✅ **適用場景：** 生產環境穩定性要求高的項目

### 為什麼選擇 Ultralytics YOLO？

- ✅ **速度極快：** 即時物件偵測（>60 FPS）
- ✅ **準確率高：** COCO 資料集上持續突破
- ✅ **易於使用：** 統一的 Python API
- ✅ **全面支援：** 檢測、分割、分類、姿態估計、OBB
- ✅ **部署便利：** 支援多種平台（iOS、Android、Web、邊緣設備）
- ✅ **持續更新：** Ultralytics 積極維護和改進

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
# YOLO11（推薦）
model = YOLO('yolo11n.pt')  # n (nano), s (small), m (medium), l (large), x (xlarge)

# 也可以使用其他版本
# model = YOLO('yolov10n.pt')  # YOLOv10
# model = YOLO('yolov9c.pt')   # YOLOv9
# model = YOLO('yolov8n.pt')   # YOLOv8

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

**YOLO11 系列（推薦）**

| 模型 | 參數量 | mAP50-95 | 速度 (ms) | 適用場景 |
|------|--------|----------|-----------|----------|
| YOLO11n | 2.6M | 39.5 | 2.4 | 邊緣裝置、移動端 |
| YOLO11s | 9.4M | 47.0 | 3.1 | 即時應用、平衡型 |
| YOLO11m | 20.1M | 51.5 | 5.3 | GPU 伺服器 |
| YOLO11l | 25.3M | 53.4 | 6.8 | 高準確率需求 |
| YOLO11x | 56.9M | 54.7 | 11.3 | 競賽、研究 |

**其他版本對比**

| 系列 | 特點 | 推薦場景 |
|------|------|----------|
| **YOLO11** | 最新架構，速度快、準確率高 | 新項目首選 |
| **YOLOv10** | 無需 NMS，延遲極低 | 邊緣設備、實時性要求高 |
| **YOLOv9** | mAP 最高（0.935），精確檢測 | 準確率優先 |
| **YOLOv8** | 成熟穩定，生態豐富 | 生產環境 |

#### 訓練範例

```python
from ultralytics import YOLO

# 載入預訓練模型（推薦使用 YOLO11）
model = YOLO('yolo11n.pt')  # 或 yolov10n.pt, yolov9c.pt, yolov8n.pt

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

**不同版本的訓練建議：**

```python
# YOLO11 - 最新推薦
model = YOLO('yolo11s.pt')
results = model.train(data='dataset.yaml', epochs=100, imgsz=640)

# YOLOv10 - 追求速度
model = YOLO('yolov10s.pt')
results = model.train(data='dataset.yaml', epochs=100, imgsz=640, amp=True)

# YOLOv9 - 追求準確率
model = YOLO('yolov9c.pt')
results = model.train(data='dataset.yaml', epochs=200, imgsz=640, patience=50)

# YOLOv8 - 穩定生產
model = YOLO('yolov8s.pt')
results = model.train(data='dataset.yaml', epochs=100, imgsz=640)
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

## 🆕 YOLO11 最新特性詳解

### 架構創新

#### 1. **C3k2 模塊**
- 取代 YOLOv8 的 C2f 模塊
- 更快的特徵聚合過程
- Cross Stage Partial with kernel size 2
- 減少計算量同時提升性能

#### 2. **C2PSA 模塊（空間注意力）**
- Convolutional block with Parallel Spatial Attention
- 增強空間注意力機制
- 專注於圖像關鍵區域
- 提升小物件和遮擋物件的檢測能力

#### 3. **SPPF 保留**
- 保留 Spatial Pyramid Pooling - Fast
- 多尺度特徵提取
- 與前代版本保持一致性

### 性能優勢

**速度對比（Snapdragon 888）：**
- YOLO11n: 2.4ms（最快）
- YOLOv10n: 5.5ms
- YOLOv8n: 4.1ms
- YOLOv9-s: 11.5ms

**準確率對比（COCO mAP50-95）：**
- YOLO11x: 54.7
- YOLOv9-e: 55.6（最高，但速度慢）
- YOLOv10x: 54.4
- YOLOv8x: 53.9

**綜合評價：**
- YOLO11 提供最佳的速度與準確率平衡
- 比 YOLOv8 參數更少，速度更快
- 在檢測卡車等大型車輛方面優於 YOLOv10
- 對小型、遠距離物件的檢測有明顯改進

### 支援任務

YOLO11 支援多種計算機視覺任務：

1. **物件檢測（Object Detection）**
   ```python
   model = YOLO('yolo11n.pt')
   results = model('image.jpg')
   ```

2. **實例分割（Instance Segmentation）**
   ```python
   model = YOLO('yolo11n-seg.pt')
   results = model('image.jpg')
   ```

3. **姿態估計（Pose Estimation）**
   ```python
   model = YOLO('yolo11n-pose.pt')
   results = model('image.jpg')
   ```

4. **定向邊界框（Oriented Bounding Box）**
   ```python
   model = YOLO('yolo11n-obb.pt')
   results = model('image.jpg')
   ```

5. **分類（Classification）**
   ```python
   model = YOLO('yolo11n-cls.pt')
   results = model('image.jpg')
   ```

### 使用建議

**何時選擇 YOLO11：**
- ✅ 新項目開發
- ✅ 需要最佳性能平衡
- ✅ 邊緣設備部署
- ✅ 實時檢測應用
- ✅ 多任務需求（檢測+分割+姿態）

**何時考慮其他版本：**
- YOLOv10：需要絕對最低延遲
- YOLOv9：準確率是首要考量
- YOLOv8：需要穩定的生產環境

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
- [Ultralytics 文檔](https://docs.ultralytics.com/)（最新最全面）
- [YOLO11 官方介紹](https://www.ultralytics.com/yolo)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Ultralytics Hub](https://hub.ultralytics.com/)（線上訓練平台）
- [Model Comparison Tool](https://docs.ultralytics.com/compare/)（版本對比）

### 社群資源
- [Ultralytics Discord](https://discord.gg/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com/)（公開資料集）
- [Papers with Code - Object Detection](https://paperswithcode.com/task/object-detection)

### 相關論文與文章
- **YOLO11 (2024)**
  - [YOLOv11: An Overview of the Key Architectural Enhancements](https://arxiv.org/abs/2410.17725)
  - [Ultralytics YOLO Evolution Overview](https://arxiv.org/abs/2510.09653)
  - [YOLO11 vs YOLOv10 Comparison](https://docs.ultralytics.com/compare/yolo11-vs-yolov10/)

- **YOLOv10 (2024)**
  - [YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
  - [Official YOLOv10 Docs](https://docs.ultralytics.com/models/yolov10/)

- **YOLOv9 (2024)**
  - [YOLOv9: Learning What You Want to Learn](https://arxiv.org/abs/2402.13616)
  - [Official YOLOv9 Docs](https://docs.ultralytics.com/models/yolov9/)

- **YOLOv8 (2023)**
  - [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
  - [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)

### 教學文章
- [YOLO11 Redefining Real-Time Object Detection](https://learnopencv.com/yolo11/)
- [YOLO Model Comparison Guide](https://www.ultralytics.com/blog/comparing-ultralytics-yolo11-vs-previous-yolo-models)
- [YOLO Evolution 2015-2024](https://viso.ai/computer-vision/yolo-explained/)

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

**本專案展示了從資料準備到模型部署的完整物件偵測工作流程，涵蓋最新的 YOLO11 技術及完整的版本對比，是學習實戰技能的絕佳範例！** 🎯

---

## 🔮 未來展望

### YOLO26（即將推出）
Ultralytics 正在開發下一代 YOLO26 模型：
- 針對邊緣部署優化
- 端到端無需 NMS 推理
- 更好、更快、更小
- 為實際部署而設計

**保持關注 Ultralytics 官方更新！**

---

**快樂檢測！探索最新的 YOLO 技術！** 🚀
