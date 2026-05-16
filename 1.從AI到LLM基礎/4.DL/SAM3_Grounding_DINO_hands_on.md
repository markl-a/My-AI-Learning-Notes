# SAM 3 與 Grounding DINO 實戰筆記

> 對應 [`./CV_全景_2024-2026.md`](./CV_全景_2024-2026.md) §1;搭配 [`./04.Ultralytics/YOLO11_YOLO26_更新.md`](./04.Ultralytics/YOLO11_YOLO26_更新.md)

---

## 1. SAM 3 是什麼(Meta, 2025/11)

Meta 在 2025 年 11 月 19 日釋出 **Segment Anything Model 3 (SAM 3)**,把 SAM 系列從「點/框 → 單一 mask」的 **Promptable Visual Segmentation (PVS)** 推進到 **Promptable Concept Segmentation (PCS)**。

簡單一句話:**SAM 2 一次 prompt 出一個 mask;SAM 3 一次 prompt 出「整張影像或整段影片裡該概念的所有實例 mask」**。Prompt 可以是:

- 短文字片語(noun phrase),例如 `"yellow school bus"`、`"person wearing helmet"`
- 影像範例(image exemplar):圈一個物件當參考
- 兩者混合

模型有 **848M 參數**,由共用 backbone 的「影像級偵測器 + 記憶體驅動的影片追蹤器」組成,新增 **presence head** 把「物件存在嗎」與「在哪裡」解耦,顯著提升偵測準確率。SAM 3 在 PCS 上的準確率**為 SAM 2 + 開放詞彙偵測組合的兩倍**,並在傳統 PVS 任務上同時超越 SAM 2。

## 2. SAM 3.1:Object Multiplex(2026/03)

SAM 3.1 在 2026 年 3 月 27 日釋出,作為 SAM 3 的 drop-in replacement,核心新增 **Object Multiplex**:用共享記憶體做**多物件聯合追蹤**,一次 forward pass 最多同時追蹤 **16 個物件**。

效果:中等物件數量影片 throughput 從 16 → 32 FPS(H100 單卡),多物件場景下最高 **7× 推理加速**,VOS benchmark 7 項中 6 項刷新。這讓即時影片應用(體育、監控、無人機)第一次能在單卡 GPU 上跑「真正的多目標追蹤」。

## 3. 訓練資料 SA-Co

**Segment Anything with Concepts (SA-Co)** 是 Meta 為了訓練 PCS 自建的資料集,核心數字:

- **4M 個獨特概念標籤**(含 hard negatives)
- **214K 個獨特 phrase** 跨 **126K 張圖片與影片**
- 概念數量是 COCO/LVIS 等傳統 benchmark 的 **50 倍**

SA-Co 透過可擴展的資料引擎(human + AI 混合標註)生成,這也是為什麼 SAM 3 能處理「戴橘色安全帽的工人」這種長尾、組合性 prompt。

## 4. SAM 1 / SAM 2 / SAM 3 對比

| 版本 | 發布 | Prompt | 輸出 | 影片 | 開放詞彙 |
|------|------|--------|------|------|----------|
| SAM 1 | 2023/04 | 點、框、粗 mask | 單一 mask | ❌ | ❌ |
| SAM 2 | 2024/07 | 點、框 + memory bank | 單一 mask(可跨 frame) | ✅ | ❌ |
| **SAM 3** | **2025/11** | **文字 / 範例 / 點框** | **該概念所有實例 mask** | ✅ | ✅ |
| SAM 3.1 | 2026/03 | 同上 + 多物件聯合追蹤 | 同上,multiplex | ✅ 16 物件 / pass | ✅ |

關鍵躍進:**SAM 3 把「偵測 + 分割 + 追蹤」三件事合進同一個模型**,不必再串 Grounding DINO + SAM 2 兩段。

## 5. Grounding DINO(ECCV 2024)

在 SAM 3 之前,**Grounding DINO**(IDEA-Research, ECCV 2024)是「文字 → bbox」開放詞彙偵測的事實標準。架構要點:

- 雙 backbone:image backbone(Swin) + text backbone(BERT)
- **feature enhancer**、**language-guided query selection**、**cross-modality decoder** 三段融合
- 預設輸出 900 個 box,每個 box 對所有輸入字詞都有相似度分數
- COCO zero-shot **52.5 AP**,ODinW zero-shot mean **26.1 AP**(刷新紀錄)

訓練資料涵蓋 detection、visual grounding、image-text pair,超過 1000 萬張圖。

## 6. Grounded-SAM Pipeline:2026 開放詞彙分割標配

雖然 SAM 3 已整合偵測+分割,但 **Grounded-SAM(Grounding DINO + SAM)** 在 2026 年仍是業界主流,原因:

1. **解耦**:可換 Grounding DINO → Florence-2 → DINO-X,或換 SAM → SAM 2 → SAM 3
2. **License 友善**:Grounding DINO + SAM 都是 Apache-2.0,SAM 3 需要申請存取
3. **輕量**:SAM 3 比 SAM 2 重不少,部分邊緣場景仍偏好「小偵測 + SAM 2」

典型 pipeline:

```
text prompt ─▶ Grounding DINO ─▶ bbox ─▶ SAM 3 / SAM 2 ─▶ mask
                                              │
                                              └─▶ memory bank ─▶ 影片追蹤
```

## 7. 完整 inference 範例(Grounded-SAM 3)

以下 ~70 行示範「找出所有戴安全帽的工人」,輸出 mask 與視覺化。前置:`pip install transformers torch pillow matplotlib numpy`,並在 HF 取得 `facebook/sam3` 與 `IDEA-Research/grounding-dino-base` 存取權限。

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    Sam3Processor, Sam3Model,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 載入 Grounding DINO(開放詞彙偵測)
gd_id = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(gd_id)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(device)

# 2) 載入 SAM 3(概念分割)
sam_proc = Sam3Processor.from_pretrained("facebook/sam3")
sam_model = Sam3Model.from_pretrained("facebook/sam3").to(device)

# 3) 讀圖 + 文字 prompt(工地照片)
image = Image.open("worksite.jpg").convert("RGB")
text_query = "a worker wearing a safety helmet."   # Grounding DINO 需句點結尾

# 4) Grounding DINO 出 bbox
gd_inputs = gd_proc(images=image, text=text_query, return_tensors="pt").to(device)
with torch.no_grad():
    gd_out = gd_model(**gd_inputs)
results = gd_proc.post_process_grounded_object_detection(
    gd_out, gd_inputs.input_ids,
    box_threshold=0.35, text_threshold=0.25,
    target_sizes=[image.size[::-1]],
)[0]
boxes = results["boxes"].cpu().numpy()
print(f"Grounding DINO 偵測到 {len(boxes)} 個候選")

# 5) SAM 3 直接用文字 prompt 找概念所有實例(也可只走這條)
sam_inputs = sam_proc(images=image, text="safety helmet worker",
                      return_tensors="pt").to(device)
with torch.no_grad():
    sam_out = sam_model(**sam_inputs)
masks = sam_proc.post_process_masks(
    sam_out.pred_masks.cpu(), sam_inputs["original_sizes"].cpu()
)[0]   # shape: [N, H, W],N = 偵測到的實例數

# 6) 視覺化
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image)
for m in masks:
    color = np.concatenate([np.random.random(3), [0.5]])
    h, w = m.shape[-2:]
    overlay = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(overlay)
for box in boxes:
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               fill=False, edgecolor="lime", linewidth=2))
ax.axis("off")
plt.tight_layout()
plt.savefig("output.png", dpi=120)
```

實務上 SAM 3 已涵蓋 Grounding DINO 的偵測能力,但保留兩段 pipeline 可以**用 Grounding DINO 的 bbox 當 SAM 的精緻 prompt**,常在小物件、密集場景準確率更穩。

## 8. 與 YOLO 系列對比

| 維度 | YOLO11 / YOLO26 | Grounded-SAM / SAM 3 |
|------|------|------|
| 詞彙 | 封閉(固定 N 類) | 開放(任意文字) |
| 速度(640×640, T4) | 1–5 ms | 60–300 ms |
| 訓練成本 | 自己標數據 | 零樣本即可用 |
| 精度(in-domain) | 高 | 中高 |
| 精度(novel class) | 0(看不到) | 高 |
| 適用 | 即時推論、邊緣 | 標註輔助、長尾、低樣本 |

**選型原則**:**「類別固定且要快」走 YOLO 26;「類別開放或標註成本高」走 Grounded-SAM**。實務常見組合:先用 SAM 3 自動標 1000 張 → 訓 YOLO26 → 上線。

## 9. 影片追蹤:Memory Bank 與 SAM 3.1 Multiplex

SAM 2 引入 **memory bank**:每個 frame 的 feature 與 mask 存進記憶體,後續 frame 透過 attention 取回,實現跨 frame 一致性。SAM 3 沿用此架構,並加入概念級理解 — 你只要在第一 frame 給「所有戴安全帽的人」,後續會自動追蹤新進場/暫時遮擋的同類物件。

SAM 3.1 的 **Object Multiplex** 進一步把 16 個物件的 memory 共享:不再每個物件跑一次 forward,而是**一次 batch 內聯合解碼**,memory bottleneck 大幅降低。對應 API 上幾乎無痛升級。

## 10. 真實 case

- **工地違規偵測**:`"worker without helmet"`、`"worker without safety vest"` 兩條 prompt 直接出警報,不需重新訓練類別。
- **自動標註輔助**:Roboflow / CVAT 已整合 SAM 3,標一張 → 自動框出整批同類物件,標註效率提升 5–10×。
- **農業精準噴藥**:`"weed between rice rows"` 找雜草,搭配 GPS 控制噴頭,農藥用量降低 30–60%。
- **無人機監測**:森林火災早期煙霧、違章建築、海岸漂流物;SAM 3.1 multiplex 讓 4K 即時推論可行。
- **影視後製**:rotoscoping(去背、合成)從幾小時縮短到幾分鐘。

## 11. 生產考量

**推理速度**:SAM 3 比 SAM 2 慢約 1.5–2×(848M 參數 + 概念解碼),H100 上單張圖 ~150–300 ms。要上線請務必:

- **Batch inference**:SA-Co 設計時就考慮 batch,單卡可同時處理 4–8 張
- **Pre-compute text embeddings**:固定 prompt(如「helmet」)可只算一次,後續所有圖共用,延遲降一半
- **Quantization**:HF Transformers 已支援 SAM 3 的 INT8 / FP16,精度損失 <1 mAP,速度 1.5–2×
- **TensorRT / vLLM-vision**:生產環境建議轉 TensorRT,延遲可再砍 30–50%
- **Roboflow 整合**:Roboflow 已上 SAM 3 hosted API,適合 PoC 與小規模生產,免自己管 GPU

**選版**:即時影片必選 SAM 3.1(multiplex);單張圖標註用 SAM 3;邊緣裝置且詞彙固定還是 YOLO26 + 知識蒸餾最划算。

---

**Sources**:
- [SAM 3: Segment Anything with Concepts (arXiv 2511.16719)](https://arxiv.org/abs/2511.16719)
- [Meta AI: SAM 3 / SAM 3.1 blog](https://ai.meta.com/blog/segment-anything-model-3/)
- [facebookresearch/sam3 GitHub](https://github.com/facebookresearch/sam3)
- [Hugging Face: facebook/sam3](https://huggingface.co/facebook/sam3)
- [Grounding DINO (arXiv 2303.05499, ECCV 2024)](https://arxiv.org/abs/2303.05499)
- [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)
- [Roboflow: What is SAM 3](https://blog.roboflow.com/what-is-sam3/)
