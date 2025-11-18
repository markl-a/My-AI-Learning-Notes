# YOLO - You Only Look Once

> **論文**: You Only Look Once: Unified, Real-Time Object Detection
>
> **作者**: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
>
> **發表**: CVPR 2016
>
> **論文鏈接**: [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
>
> **官方網站**: [pjreddie.com/darknet/yolo](https://pjreddie.com/darknet/yolo/)

---

## 🎯 簡介

**YOLO (You Only Look Once)** 是目標檢測領域的革命性工作，將目標檢測重新定義為單一回歸問題，實現了**實時**目標檢測。與傳統的兩階段檢測器（如 R-CNN）不同，YOLO 在單次前向傳播中同時預測邊界框和類別概率。

### 核心思想

```
傳統方法 (R-CNN系列):
1. 區域提議 (Region Proposals)
2. 特徵提取
3. 分類每個區域
→ 慢，難以優化

YOLO:
1. 將圖像劃分為網格
2. 每個網格預測邊界框和類別
3. 單次前向傳播完成檢測
→ 快，端到端訓練
```

---

## 💡 核心創新

### 1. 統一檢測框架

**將檢測視為回歸問題**:
- 輸入: 整張圖像
- 輸出: 邊界框坐標 + 類別概率
- 單個神經網絡，端到端訓練

### 2. 網格劃分

```
圖像 → S×S 網格（論文中 S=7）

每個網格單元預測:
- B 個邊界框（論文中 B=2）
- 每個邊界框: (x, y, w, h, confidence)
- C 個類別概率（PASCAL VOC: C=20）

最終輸出: S×S×(B*5 + C) = 7×7×30 張量
```

### 3. 實時性能

| 模型 | FPS | mAP |
|------|-----|-----|
| Fast R-CNN | 0.5 | 70.0% |
| Faster R-CNN | 7 | 73.2% |
| **YOLO** | **45** | 63.4% |
| **Fast YOLO** | **155** | 52.7% |

---

## 🏗️ 架構

### 網絡結構

```
輸入: 448×448×3

卷積層 (24層):
Conv 7×7×64-s-2
MaxPool 2×2-s-2
Conv 3×3×192
MaxPool 2×2-s-2
Conv 1×1×128
Conv 3×3×256
Conv 1×1×256
Conv 3×3×512
MaxPool 2×2-s-2
[Conv 1×1×256 + Conv 3×3×512] × 4
Conv 1×1×512
Conv 3×3×1024
MaxPool 2×2-s-2
[Conv 1×1×512 + Conv 3×3×1024] × 2
Conv 3×3×1024
Conv 3×3×1024-s-2
Conv 3×3×1024
Conv 3×3×1024

全連接層 (2層):
FC 4096
FC 7×7×30

輸出: 7×7×30 張量
```

### 損失函數

```python
# YOLO 損失包含5部分
loss = (
    λ_coord * 坐標損失 +  # 邊界框中心坐標
    λ_coord * 尺寸損失 +  # 邊界框寬高
    包含物體的置信度損失 +
    不包含物體的置信度損失 +
    分類損失
)

# λ_coord = 5, λ_noobj = 0.5
```

---

## 🚀 快速開始

### 使用 Ultralytics YOLOv8

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

# 載入模型
model = YOLO('yolov8n.pt')  # n, s, m, l, x

# 預測
results = model('image.jpg')

# 顯示結果
for result in results:
    boxes = result.boxes  # 邊界框
    result.show()  # 顯示圖像
```

### 從零實現（簡化版）

```python
class YOLO(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S  # 網格大小
        self.B = B  # 每個網格的邊界框數
        self.C = C  # 類別數

        # 特徵提取（簡化）
        self.features = nn.Sequential(
            # ... 卷積層
        )

        # 檢測頭
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S*S*(B*5 + C))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        # 重塑為 (S, S, B*5 + C)
        x = x.view(-1, self.S, self.S, self.B*5 + self.C)
        return x
```

---

## 📊 實驗結果

### PASCAL VOC 2007

| 模型 | mAP | FPS |
|------|-----|-----|
| DPM v5 | 33.7% | 0.07 |
| R-CNN | 54.2% | 0.02 |
| Fast R-CNN | 70.0% | 0.5 |
| Faster R-CNN | 73.2% | 7 |
| **YOLO** | **63.4%** | **45** |

### YOLO 系列演進

| 版本 | 年份 | 主要改進 | mAP | FPS |
|------|------|---------|-----|-----|
| YOLOv1 | 2016 | 統一檢測框架 | 63.4% | 45 |
| YOLOv2 | 2017 | Batch Norm, Anchor Boxes | 78.6% | 67 |
| YOLOv3 | 2018 | 多尺度預測, FPN | 57.9% (COCO) | 35 |
| YOLOv4 | 2020 | CSPDarknet, Mish | 43.5% (COCO) | 65 |
| YOLOv5 | 2020 | PyTorch, 易用性 | 50.7% (COCO) | 140 |
| YOLOv8 | 2023 | 最新架構優化 | 53.9% (COCO) | 280 |

---

## 🎯 應用場景

### 1. 實時視頻檢測

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 視頻流檢測
cap = cv2.VideoCapture(0)  # 攝像頭

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 檢測
    results = model(frame)

    # 繪製結果
    annotated_frame = results[0].plot()
    cv2.imshow('YOLO', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
```

### 2. 自動駕駛

- 行人檢測
- 車輛檢測
- 交通標誌識別

### 3. 安防監控

- 異常行為檢測
- 人流統計
- 物品遺留檢測

### 4. 工業檢測

- 缺陷檢測
- 產品計數
- 質量控制

---

## 🌟 優勢與局限

### 優勢

- ✅ **速度快**: 實時檢測（45 FPS）
- ✅ **端到端**: 統一訓練，簡單高效
- ✅ **全局推理**: 看到整張圖像，減少背景誤檢
- ✅ **泛化能力**: 在藝術作品等新領域表現好

### 局限

- ❌ **小物體**: 對小物體和密集物體檢測較差
- ❌ **精度**: mAP 低於 Faster R-CNN
- ❌ **邊界框**: 每個網格只能檢測有限數量物體
- ❌ **長寬比**: 對不常見的長寬比泛化較差

---

## 📚 參考資源

### 論文系列

1. **YOLOv1** (2016): [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
2. **YOLOv2/YOLO9000** (2017): [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
3. **YOLOv3** (2018): [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
4. **YOLOv4** (2020): [arXiv:2004.10934](https://arxiv.org/abs/2004.10934)

### 代碼實現

- 🔥 **Ultralytics YOLOv8**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- 📦 **Darknet (原始)**: [pjreddie/darknet](https://github.com/pjreddie/darknet)
- 🐍 **PyTorch YOLO**: [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

### 學習資源

- 📺 **YOLO 論文精讀**: [bilibili](https://www.bilibili.com/video/BV1VZ4y1m7BV)
- 📖 **目標檢測綜述**: [Object Detection in 20 Years](https://arxiv.org/abs/1905.05055)

---

## 📝 引用

```bibtex
@inproceedings{redmon2016you,
  title={You only look once: Unified, real-time object detection},
  author={Redmon, Joseph and Divvala, Santosh and Girshick, Ross and Farhadi, Ali},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={779--788},
  year={2016}
}
```

---

<div align="center">
  <p><strong>⭐ YOLO: 讓實時目標檢測成為可能！</strong></p>
  <p>🚀 從 YOLOv1 到 YOLOv8 | 📹 實時應用 | 🌍 廣泛影響</p>
  <p><i>最後更新: 2024-11-18</i></p>
</div>
