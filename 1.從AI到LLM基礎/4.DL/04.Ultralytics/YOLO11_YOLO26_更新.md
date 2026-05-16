> 本檔補完 04.Ultralytics 既有 YOLOv8 內容到 2026 SOTA。對應 [`../CV_全景_2024-2026.md`](../CV_全景_2024-2026.md) §1

# YOLO11 → YOLO26 更新筆記

本資料夾的 `1.yolo8物件偵測/` 仍以 YOLOv8 為主軸,本檔接續補完從 v8 之後到 2026 年 1 月 YOLO26 釋出為止的演進、實務升級流程、以及部署層面的關鍵變化。

## 1. YOLO 演進線(2023 → 2026)

| 版本 | 釋出 | 維護者 | 關鍵字 |
|---|---|---|---|
| **YOLOv8** | 2023 | Ultralytics | C2f、Anchor-free、Decoupled head、DFL |
| **YOLOv9** | 2024 初 | WongKinYiu (社群) | **PGI**(Programmable Gradient Information)+ **GELAN**,解決深層梯度丟失 |
| **YOLOv10** | 2024 中 | 清華 THU-MIG | **端到端 NMS-free**(consistent dual assignments)、輕量分類頭 |
| **YOLO11** | 2024/09 | Ultralytics | **C3k2** 取代 C2f、**C2PSA** 空間注意力、22% 參數減量 |
| **YOLOv12** | 2025/02 | 學術 | **Area Attention(A²)** + **R-ELAN** + **FlashAttention**,attention-centric |
| **YOLOv13** | 2025 中 | 社群 (非官方) | **HyperACE**(hypergraph)+ **FullPAD**,高階特徵融合 |
| **YOLO26** | 2026/01 | Ultralytics | **原生 NMS-free**、**移除 DFL**、**ProgLoss + STAL + MuSGD**,nano CPU 推論最多快 43% |

> 命名跳號:Ultralytics 跳過 v13–25,直接命名為 YOLO26,主要為了對齊年份語意(2026)並與社群分支 YOLOv13 區隔。

## 2. YOLO26 核心改進

YOLO26 不再走 v12 的「加更多 attention」路線,反而 **針對 edge 部署做減法**:

- **NMS-free 原生支援**:像 v10 一樣 end-to-end,但整合進 Ultralytics 主線。Inference graph 不再有 NMS post-processing,TensorRT / CoreML / TFLite 編譯時不需要插入 plugin。
- **移除 DFL(Distribution Focal Loss)分支**:DFL 雖然提升精度,但在低算力裝置上是額外的 reshape + softmax 負擔。YOLO26 直接回歸到 scalar regression。
- **ProgLoss(Progressive Loss Balancing)**:訓練過程動態調整 cls / box / aux 的權重,避免後期被 easy negatives 或大物件主導。
- **STAL(Small-Target-Aware Label assignment)**:對小物件、遮擋、低對比樣本給更高的空間容忍度與正樣本分配機率,提升 small-object recall。
- **MuSGD optimizer**:SGD 與 Muon optimizer 的混合,靈感來自 Moonshot AI Kimi K2 的 LLM 訓練方案,把 LLM 訓練技術遷移到 CV。
- **效能**:nano 模型在 CPU 上比 YOLO11n **最多快 43%**,small-object mAP 提升明顯。

## 3. YOLOv13(社群分支)

YOLOv13 並非 Ultralytics 官方,是學術社群延續 v12 的 attention 路線:

- **HyperACE**:用 hypergraph 建模 cross-location / cross-scale 的高階關聯。
- **FullPAD**:Full-Pipeline Aggregation-and-Distribution,讓特徵流貫穿整個 pipeline。
- 在 COCO 上 YOLOv13-N 比 YOLO11-N 高 3.0% mAP、比 YOLOv12-N 高 1.5% mAP。

**選型建議**:研究 / 比賽用 v13;生產 / 邊緣部署選 YOLO26。

## 4. YOLO11 vs YOLO26 對比

| 項目 | YOLO11n | YOLO26n |
|---|---|---|
| 後處理 | 仍需 NMS | **原生 end-to-end,無 NMS** |
| DFL | 有 | **移除** |
| CPU 推論延遲 | 基準 | **快至 43%** |
| Small-object mAP | 基準 | 明顯提升(STAL) |
| Edge 部署便利度 | 需 export 後手動處理 NMS plugin | 直接 ONNX/TFLite 即可跑 |
| 訓練穩定性 | 標準 SGD | MuSGD 收斂更穩 |

## 5. 從 YOLOv8 升級到 YOLO11 / YOLO26 的 Migration Guide

Ultralytics 主線 API 幾乎沒有 breaking change,大部分情境下:

```bash
pip install -U ultralytics
```

然後改一行模型名:

```python
# 舊
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# 新(YOLO11)
model = YOLO("yolo11n.pt")

# 新(YOLO26)
model = YOLO("yolo26n.pt")
```

- **dataset.yaml 格式不變**:`path` / `train` / `val` / `names` 都相容。
- **export 不需要客製 plugin**:`model.export(format="onnx")` / `engine`(TensorRT)/ `coreml` / `tflite` 全部官方支援。
- **預訓練權重**:首次使用會自動從 Ultralytics CDN 下載。
- **Augmentation default 微調**:YOLO26 的 mosaic / mixup 概率與 v8 略有不同,自訂訓練超參時建議用官方預設先 baseline。

## 6. 完整訓練 + Export Pipeline 範例

從資料標註到 Jetson 部署的最小可行流程:

```python
# ---------- 0. 準備 ----------
# Roboflow 標完 → export 成 YOLO 格式 → 拿到 data.yaml
# pip install ultralytics roboflow onnx onnxruntime-gpu

from ultralytics import YOLO
from roboflow import Roboflow

# 1. (選用) Roboflow 拉資料
rf = Roboflow(api_key="YOUR_KEY")
dataset = rf.workspace("ws").project("ppe").version(3).download("yolov8")
# dataset.location 內含 data.yaml

# 2. 訓練 YOLO26
model = YOLO("yolo26n.pt")  # 從 nano 預訓練開始
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer="MuSGD",   # YOLO26 預設
    cos_lr=True,
    patience=20,
    device=0,
    project="runs/ppe",
    name="yolo26n_v1",
)

# 3. 驗證
metrics = model.val()
print(f"mAP50-95: {metrics.box.map:.3f}, mAP50: {metrics.box.map50:.3f}")

# 4. 推論測試
pred = model.predict(
    source="test_images/",
    conf=0.25,
    save=True,
)

# 5. Export 到部署格式(NMS-free,export 出來直接可用)
model.export(format="onnx", opset=17, dynamic=True, simplify=True)
# 產生 yolo26n_v1.onnx

# 6. Jetson 上 TensorRT 編譯
# $ trtexec --onnx=yolo26n_v1.onnx --saveEngine=yolo26n_v1.engine --fp16
# 因為 NMS 已內嵌進 graph,不需要 --plugins 或客製化 EfficientNMS_TRT

# 7. (選用) 也可以直接從 Python 一步轉 engine
model.export(format="engine", half=True, device=0, dynamic=True)
```

## 7. NMS-free 對部署的影響

傳統 YOLO 部署最痛的地方:

- ONNX export 出來 NMS 是空的,要靠 runtime 處理(ORT 沒有 NMS op 時要自己寫)。
- TensorRT 要插入 `EfficientNMS_TRT` plugin,版本對不上會崩。
- CoreML / TFLite 各自有自己的 NMS quirk。

YOLO26 的 end-to-end 設計把 NMS 邏輯透過 dual-assignment 訓練吸收進主幹預測,**inference 直接吐出 final boxes**。實務上:
- TensorRT engine 編譯指令更短,不用 plugin。
- 行動端(TFLite / CoreML)可以直接拿 graph 跑,不必在 Swift / Kotlin 那邊重寫 NMS。
- Throughput 因為少一段 host-side post-processing 通常還會再提升 10–20%。

## 8. 與 RT-DETR / RF-DETR 對比 — 何時選 Transformer Detector

| 場景 | 建議 |
|---|---|
| 邊緣裝置 / 即時(>60 FPS) | **YOLO26** |
| 多類別、複雜場景、大 batch GPU 推論 | RT-DETR / RF-DETR |
| 需要 open-vocabulary | Grounding DINO 系列 |
| Small-object 高 recall 為主 | YOLO26 + STAL,通常已夠用 |
| 重疊密集(人群、車流)| RT-DETR 在 query-based assignment 上略勝 |

RT-DETR / RF-DETR 走的是 set prediction,不需 NMS 是與生俱來的,但 backbone 普遍較重,nano-size 還是 YOLO 系贏。

## 9. 真實 Case

- **工地 PPE 偵測**:類別少(helmet / vest / no-helmet),YOLO26n 在 Jetson Orin Nano 上輕鬆 60 FPS,STAL 對遠距離小人物很有感。
- **零售貨架空缺(Out-of-Stock)**:用 YOLO26s 做格位偵測 + 後端規則判斷,部署到店端 mini-PC(無 GPU),CPU 推論 8–12 FPS 已可接受。
- **無人機目標追蹤**:YOLO26 + ByteTrack / BoT-SORT 是 2026 的標配組合,小物件 recall 提升讓追蹤 ID switch 顯著降低。

## 10. 與 Grounding DINO + SAM 3 結合

YOLO26 雖強,但仍是 **封閉類別**(closed-vocabulary),需要先標資料。實務上的分工:

- **Bootstrap 階段(無標註)**:用 Grounding DINO 給定文字 prompt → 自動產生 pseudo box → 過濾 → 訓練 YOLO26。
- **遮罩需求**:YOLO26 偵測完 box → 丟給 **SAM 3** 拿到 instance mask(SAM 3 支援 prompt-aware 細分割)。
- **長尾類別**:罕見類別保留給 Grounded-SAM 線上推論;高頻常見類別才轉成 YOLO26 跑即時。

這個 hybrid pipeline 是 2026 年量產 CV 系統最常見的架構:**YOLO26 做 hot path(快、便宜),Grounded-SAM 做 cold path(慢、彈性)**。

---

延伸閱讀:
- Ultralytics YOLO26 docs: <https://docs.ultralytics.com/models/yolo26>
- YOLO26 arXiv: <https://arxiv.org/abs/2509.25164>
- YOLOv12 arXiv: <https://arxiv.org/abs/2502.12524>
- YOLOv13: <https://arxiv.org/html/2506.17733v2>
- Ultralytics YOLO Evolution 綜覽: <https://arxiv.org/pdf/2510.09653>
