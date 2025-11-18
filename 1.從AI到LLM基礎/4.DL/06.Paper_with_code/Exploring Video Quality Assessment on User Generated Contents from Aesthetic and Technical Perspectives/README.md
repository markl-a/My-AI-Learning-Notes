# DOVER - 用戶生成內容視頻質量評估

> **論文**: Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives
> **作者**: Wu et al.
> **發表**: CVPR 2023
> **論文鏈接**: [arXiv:2211.04894](https://arxiv.org/pdf/2211.04894v3)
> **官方代碼**: [GitHub - DOVER](https://github.com/VQAssessment/DOVER)

---

## 📋 目錄

- [簡介](#簡介)
- [核心創新](#核心創新)
- [文件說明](#文件說明)
- [快速開始](#快速開始)
- [模型架構](#模型架構)
- [實驗結果](#實驗結果)
- [應用場景](#應用場景)
- [參考資源](#參考資源)

---

## 🎯 簡介

DOVER (Disentangled Objective Video quality EvaluatoR) 是一個創新的視頻質量評估框架，專門針對用戶生成內容（UGC）設計。與傳統方法不同，DOVER 將視頻質量評估分解為**美學**和**技術**兩個獨立的維度，更準確地反映人類對視頻質量的真實感知。

### 為什麼需要 DOVER？

在 YouTube、TikTok、Instagram 等平台上，每天有數億用戶上傳視頻。這些 UGC 視頻的質量差異巨大：
- 有些視頻技術質量很高（清晰、流暢），但內容平淡無趣
- 有些視頻內容精彩、構圖優美，但畫質較差
- 傳統的質量評估方法無法區分這兩個維度

DOVER 通過將質量評估分解為美學和技術兩個角度，解決了這個問題。

---

## 💡 核心創新

### 1. DIVIDE-3k 數據集

- **首個雙維度標註的 UGC-VQA 數據集**
- 包含 **3,590** 個視頻樣本
- 每個視頻都有三個評分：
  - 🎨 **美學評分** (Aesthetic Score)
  - 🔧 **技術評分** (Technical Score)
  - ⭐ **整體評分** (Overall Score)
- 來源：YFCC-100M 和 Kinetics-400

### 2. DOVER 模型

DOVER 採用**雙分支架構**：

```
                    輸入視頻
                       |
        ┌──────────────┴──────────────┐
        ▼                             ▼
    美學分支                       技術分支
    ├─ 下採樣 (128×128)            ├─ 保持分辨率 (224×224)
    ├─ 稀疏幀採樣                  ├─ 連續幀採樣
    ├─ ConvNeXt Backbone          ├─ ConvNeXt Backbone
    └─ 語義特徵提取                └─ 失真特徵提取
        |                             |
        ▼                             ▼
    美學評分 (SA)                 技術評分 (ST)
        |                             |
        └──────────┬──────────────────┘
                   ▼
        整體評分 = 0.428×SA + 0.572×ST
```

**關鍵技術**：
- ✅ **視角分解策略**：分離美學和技術特徵
- ✅ **跨尺度正則化**：確保美學評估不受技術失真影響
- ✅ **弱監督學習**：僅使用整體質量標籤訓練兩個分支
- ✅ **主觀啟發融合**：基於人類感知的加權融合

### 3. DOVER++

DOVER++ 進一步擴展了 DOVER，支持：
- 🎯 **個性化質量評估**：根據用戶偏好調整權重
- 📊 **單維度質量預測**：僅從美學或技術角度評估
- 🔄 **靈活的應用場景適配**

---

## 📁 文件說明

本目錄包含以下文件：

### 1. `影片品質評估：從美學與技術角度-內容簡介.md`
**論文詳細解說**，包含：
- 📖 論文各章節的深入分析
- 🔬 技術實現細節
- 💻 實作指南和代碼示例
- 📚 相關資源和延伸閱讀
- 🚀 未來研究方向

**適合對象**：想要深入理解論文原理的讀者

### 2. `DOVER_復現與實作.ipynb`
**完整的實作教程**，包含：
- 🛠️ 環境設置和依賴安裝
- 🏗️ DOVER 模型架構實現
- 🎬 視頻預處理工具
- 📊 結果視覺化
- 🔄 批量評估功能
- 🎯 個性化質量評估

**適合對象**：想要動手實踐的開發者

### 3. `TWCDOVER復現.ipynb` (舊版)
早期的復現嘗試，包含一些問題和錯誤。建議使用新版的 `DOVER_復現與實作.ipynb`。

### 4. `README.md` (本文件)
項目概述和快速導航。

---

## 🚀 快速開始

### 環境需求

```bash
# Python 版本
Python 3.8+

# 主要依賴
pip install torch torchvision
pip install decord opencv-python
pip install timm matplotlib seaborn
pip install scipy scikit-learn tqdm
```

### 基本使用

```python
import torch
from dover import DOVER

# 1. 載入模型
model = DOVER(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 2. 評估視頻
video_path = "sample_video.mp4"
aesthetic_score, technical_score, overall_score = model.evaluate(video_path)

# 3. 查看結果
print(f"美學評分: {aesthetic_score:.3f}")
print(f"技術評分: {technical_score:.3f}")
print(f"整體評分: {overall_score:.3f}")

# 4. 個性化評估（可選）
# 藝術導向（強調美學）
artistic_score = 0.7 * aesthetic_score + 0.3 * technical_score

# 專業導向（強調技術）
professional_score = 0.3 * aesthetic_score + 0.7 * technical_score
```

### 詳細教程

請參考 `DOVER_復現與實作.ipynb` 獲取完整的實作教程。

---

## 🏗️ 模型架構

### 美學分支 (Aesthetic Branch)

**目標**：評估視頻的語義內容、構圖、色彩等美學因素

**技術特點**：
- 📐 **空間下採樣**：降低到 128×128 訓練，224×224 推理
- ⏱️ **稀疏幀採樣**：每 4 幀抽取 1 幀
- 🧠 **Backbone**：ConvNeXt-Tiny (ImageNet-1K 預訓練)
- 🔧 **特殊技術**：跨尺度正則化

**為什麼這樣設計？**
- 下採樣減少對技術細節的敏感度
- 稀疏採樣關注內容語義而非時序失真
- 跨尺度正則化確保評估的一致性

### 技術分支 (Technical Branch)

**目標**：評估視頻的失真程度、清晰度等技術指標

**技術特點**：
- 📐 **保持分辨率**：使用原始或較高分辨率 (224×224)
- ⏱️ **連續幀採樣**：密集採樣以捕捉時序失真
- 🧠 **Backbone**：ConvNeXt-Tiny
- 🔧 **特殊技術**：隨機裁剪拼接

**為什麼這樣設計？**
- 高分辨率保留失真細節
- 連續採樣捕捉壓縮、卡頓等時序問題
- 隨機裁剪破壞美學連貫性，專注失真

### 融合策略

基於主觀研究，DOVER 使用加權融合：

```
整體評分 = 0.428 × 美學評分 + 0.572 × 技術評分
```

這個權重比例是通過大規模人類主觀評估實驗得出的，反映了人們對 UGC 視頻質量的真實感知。

---

## 📊 實驗結果

### 在主要數據集上的性能

| 數據集 | SRCC | PLCC | 說明 |
|--------|------|------|------|
| **LSVQ** | 0.876 | 0.894 | 大規模測試集（最具挑戰性） |
| **KoNViD-1k** | 0.895 | 0.907 | 中等規模數據集 |
| **YouTube-UGC** | 0.812 | 0.826 | 跨平台泛化能力 |
| **DIVIDE-3k** | 0.912 | 0.921 | 新提出的數據集 |

> **SRCC** (Spearman Rank Correlation Coefficient): 斯皮爾曼等級相關係數
> **PLCC** (Pearson Linear Correlation Coefficient): 皮爾遜線性相關係數

### 與其他方法的比較

DOVER 在所有主要 UGC-VQA 基準測試中均達到了 **最先進 (SOTA)** 的性能，顯著優於：
- ✅ BRISQUE (2012)
- ✅ VSFA (2019)
- ✅ MDTVSFA (2021)
- ✅ SimpleVQA (2022)

### 計算效率

- ⚡ **推理速度**：約 0.5 秒/視頻（使用單個 GPU）
- 💾 **模型大小**：約 56M 參數（兩個分支各 28M）
- 🔋 **記憶體需求**：約 2GB GPU 記憶體

---

## 🎯 應用場景

### 1. 視頻平台質量控制
**場景**：YouTube、TikTok 等平台需要評估用戶上傳內容

**應用方式**：
```python
# 評估上傳視頻
if overall_score < threshold:
    # 提示用戶提升視頻質量
    if aesthetic_score < technical_score:
        suggest("改善視頻構圖和內容呈現")
    else:
        suggest("提高視頻拍攝或編碼質量")
```

### 2. 個性化內容推薦
**場景**：根據用戶偏好推薦視頻

**應用方式**：
```python
# 根據用戶歷史行為調整權重
if user_prefers_artistic_content:
    score = 0.7 * aesthetic_score + 0.3 * technical_score
else:
    score = 0.3 * aesthetic_score + 0.7 * technical_score
```

### 3. 視頻壓縮優化
**場景**：評估不同壓縮參數對質量的影響

**應用方式**：
```python
# 測試不同壓縮等級
for compression_level in [high, medium, low]:
    compressed_video = compress(original, compression_level)
    _, technical_score, _ = model.evaluate(compressed_video)

    # 找到技術質量可接受的最高壓縮率
    if technical_score > acceptable_threshold:
        optimal_compression = compression_level
```

### 4. 內容創作輔助
**場景**：為視頻創作者提供質量反饋

**應用方式**：
```python
# 分析視頻並提供改進建議
aesthetic_score, technical_score, _ = model.evaluate(user_video)

feedback = []
if aesthetic_score < 3.5:
    feedback.append("🎨 建議：改善構圖和色彩搭配")
if technical_score < 3.5:
    feedback.append("🔧 建議：提高視頻清晰度和穩定性")

return feedback
```

### 5. 廣告投放優化
**場景**：評估廣告素材質量，優化投放策略

**應用方式**：
- 僅投放高質量廣告（overall_score > 4.0）
- 根據平台特性調整權重（專業平台強調技術，社交平台強調美學）

---

## 📚 參考資源

### 論文與代碼

- 📄 **原始論文**: [arXiv:2211.04894](https://arxiv.org/pdf/2211.04894v3)
- 💻 **官方代碼**: [GitHub - VQAssessment/DOVER](https://github.com/VQAssessment/DOVER)
- 🏆 **Papers with Code**: [DOVER on Papers with Code](https://paperswithcode.com/paper/exploring-video-quality-assessment-on-user)

### 數據集

- **DIVIDE-3k**: 本論文提出的新數據集（需聯繫作者獲取）
- **LSVQ**: [Large-Scale Video Quality Database](https://github.com/baidut/PatchVQ)
- **KoNViD-1k**: [KoNViD-1k Dataset](http://database.mmsp-kn.de/konvid-1k-database.html)
- **YouTube-UGC**: [YouTube UGC Dataset](https://media.withyoutube.com/)

### 相關論文

1. **BRISQUE** (2012): "No-Reference Image Quality Assessment in the Spatial Domain"
2. **VSFA** (2019): "Quality Assessment of In-the-Wild Videos"
3. **MDTVSFA** (2021): "Multi-Dimensional Video Quality Assessment with Multi-Task Learning"
4. **SimpleVQA** (2022): "Simple Video Quality Assessment with Standard CNN"

### 工具與框架

- 🔥 **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- 🎬 **Decord**: [https://github.com/dmlc/decord](https://github.com/dmlc/decord)
- 🖼️ **Timm**: [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- 🎥 **FFmpeg**: [https://ffmpeg.org/](https://ffmpeg.org/)

### 學習資源

- 📺 **視頻質量評估綜述**: [Video Quality Assessment: A Survey](https://arxiv.org/abs/2207.02595)
- 🎨 **圖像美學評估**: [Deep Learning for Image Aesthetics Assessment](https://arxiv.org/abs/1907.11985)
- 🔍 **失真檢測技術**: [Blind Image Quality Assessment: A Survey](https://arxiv.org/abs/1907.02665)

---

## 🤝 貢獻

歡迎提出問題、建議或改進！

如果您發現任何錯誤或有改進建議，請：
1. 提交 Issue
2. 提交 Pull Request
3. 聯繫維護者

---

## 📝 引用

如果您在研究中使用了 DOVER，請引用原始論文：

```bibtex
@inproceedings{wu2023dover,
  title={Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives},
  author={Wu, Haoning and Zhang, Erli and Liao, Liang and Chen, Chaofeng and Hou, Jingwen and Wang, Annan and Sun, Wenxiu and Yan, Qiong and Lin, Weisi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={},
  year={2023}
}
```

---

## 📄 授權

本教程遵循原論文和官方代碼的授權協議。請遵守相關的使用條款。

---

## 🔖 版本歷史

- **v1.0** (2024-11-18): 初始版本，包含論文解說和基礎實作
- **v1.1** (2024-11-18): 新增完整的 Jupyter Notebook 教程
- **v1.2** (2024-11-18): 增強文檔，添加更多示例和應用場景

---

**最後更新**: 2024-11-18

---

<div align="center">
  <p><strong>⭐ 如果這個項目對您有幫助，請給它一個 Star！</strong></p>
  <p>📚 持續學習 | 💡 分享知識 | 🚀 共同進步</p>
</div>
