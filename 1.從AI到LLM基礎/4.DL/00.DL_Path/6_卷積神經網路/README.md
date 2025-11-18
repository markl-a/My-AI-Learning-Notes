# 卷積神經網路 (CNN) 完整教程

> 從基礎到進階，從理論到實踐的CNN學習路徑

## 📖 課程概述

本資料夾包含完整的卷積神經網路（CNN）學習資源，涵蓋從基礎概念到現代架構，從理論推導到實戰應用的全方位內容。

### 🎯 適合對象

- 深度學習初學者
- 想系統學習CNN的開發者
- 需要實戰經驗的工程師
- 準備面試的求職者

### 💡 學習成果

完成本課程後，你將能夠：
- ✅ 深入理解CNN的工作原理
- ✅ 實現經典和現代CNN架構
- ✅ 掌握正則化和數據增強技術
- ✅ 使用可視化工具分析模型
- ✅ 應用遷移學習解決實際問題
- ✅ 使用AI工具提升開發效率

---

## 📚 課程內容

### 第一部分：基礎篇（原有內容）

#### [1. 從全連接層到卷積](1_why-conv.ipynb)
- **內容**：為什麼需要卷積層
- **關鍵概念**：平移不變性、局部性
- **實踐**：卷積運算推導
- **時長**：2-3小時

#### [2. 圖像卷積](2_conv-layer.ipynb)
- **內容**：互相關運算、卷積層實現
- **關鍵概念**：卷積核、特徵圖
- **實踐**：手動實現卷積、邊緣檢測
- **時長**：2-3小時

#### [3. 填充和步幅](3_padding-and-strides.ipynb)
- **內容**：控制輸出大小的方法
- **關鍵概念**：padding、stride、輸出形狀計算
- **實踐**：不同參數配置的效果
- **時長**：1-2小時

#### [4. 多輸入多輸出通道](4_channels.ipynb)
- **內容**：處理彩色圖像和多特徵圖
- **關鍵概念**：通道、1×1卷積
- **實踐**：多通道卷積實現
- **時長**：2-3小時

#### [5. 池化層](5_pooling.ipynb)
- **內容**：降採樣技術
- **關鍵概念**：最大池化、平均池化
- **實踐**：池化層實現和效果比較
- **時長**：1-2小時

#### [6. LeNet](6_lenet.ipynb)
- **內容**：第一個成功的CNN網路
- **關鍵概念**：完整的CNN架構
- **實踐**：在Fashion-MNIST上訓練LeNet
- **時長**：3-4小時

**基礎篇總時長：12-17小時**

---

### 第二部分：進階篇（新增內容）

#### [7. 現代CNN架構](7_modern-cnn-architectures.ipynb) 🆕
- **內容**：
  - AlexNet：深度學習革命的開端
  - VGG：更深的網路，3×3卷積的威力
  - Network in Network：1×1卷積的應用
  - GoogLeNet/Inception：多尺度特徵提取
  - ResNet：殘差連接突破深度限制
  - 架構對比與選擇指南
- **實踐**：
  - 實現所有主要架構
  - 參數量和性能對比
  - 在不同數據集上測試
- **時長**：6-8小時
- **難度**：⭐⭐⭐⭐

#### [8. 正則化與數據增強](8_regularization-and-augmentation.ipynb) 🆕
- **內容**：
  - **正則化技術**：
    - Dropout：原理、實現、使用建議
    - Batch Normalization：數學推導、PyTorch實現
    - Layer Normalization
    - Weight Decay、Early Stopping
  - **數據增強技術**：
    - 基礎變換：翻轉、旋轉、裁剪、顏色調整
    - Cutout：隨機遮蔽
    - Mixup：圖像混合
    - CutMix：結合Cutout和Mixup
  - **完整訓練流程**：結合多種技術的訓練循環
- **實踐**：
  - 手動實現Dropout和BatchNorm
  - 實現Cutout和Mixup
  - 完整的訓練pipeline
  - 對比實驗
- **時長**：5-7小時
- **難度**：⭐⭐⭐⭐

#### [9. CNN可視化](9_cnn-visualization.ipynb) 🆕
- **內容**：
  - **特徵圖可視化**：Hook機制、各層特徵分析
  - **卷積核可視化**：理解學到的模式
  - **Grad-CAM**：類激活映射，理解模型決策
  - **特徵空間可視化**：t-SNE降維
  - **訓練過程可視化**：損失和準確率曲線
  - **混淆矩陣和錯誤分析**
- **實踐**：
  - GradCAM完整實現
  - 特徵提取和可視化pipeline
  - 交互式分析工具
  - 錯誤樣本分析
- **時長**：4-6小時
- **難度**：⭐⭐⭐⭐⭐

#### [10. 遷移學習與微調](10_transfer-learning.ipynb) 🆕
- **內容**：
  - **遷移學習基礎**：原理和適用場景
  - **特徵提取**：固定預訓練模型
  - **微調策略**：部分解凍、差異化學習率
  - **逐步解凍**：多階段訓練
  - **數據預處理**：ImageNet歸一化
  - **最佳實踐**：模型選擇、學習率策略
- **實踐**：
  - 載入和修改預訓練模型
  - 在CIFAR-10上實戰
  - 對比不同策略的效果
  - 完整的訓練流程
- **時長**：5-7小時
- **難度**：⭐⭐⭐⭐
- **實用性**：⭐⭐⭐⭐⭐

#### [11. AI輔助開發指南](11_AI-assisted-development.md) 🆕
- **內容**：
  - **AI工具**：Copilot、ChatGPT、Claude、Cursor使用技巧
  - **實驗管理**：Weights & Biases、TensorBoard
  - **自動化**：
    - Optuna超參數優化
    - Ray Tune分佈式調優
    - AutoKeras自動模型搜索
  - **調試技巧**：使用AI助手、梯度檢查、激活監控
  - **性能優化**：混合精度、數據加載、編譯優化
  - **部署實踐**：量化、剪枝、ONNX、TorchScript
  - **實戰Checklist**：完整的開發流程清單
- **實踐**：
  - AI提示詞模板
  - 完整的代碼示例
  - 工具配置和使用
  - 最佳實踐指南
- **時長**：4-5小時
- **難度**：⭐⭐⭐
- **實用性**：⭐⭐⭐⭐⭐

**進階篇總時長：24-33小時**

---

## 🗺️ 學習路線圖

### 路線1：快速入門（適合有時間限制的學習者）
```
1_why-conv → 2_conv-layer → 6_lenet → 7_modern-cnn → 10_transfer-learning
總時長：約15-20小時
```

### 路線2：系統學習（適合深度學習初學者）
```
按順序學習1-11所有章節
完成每章的練習題
實現完整的項目
總時長：約40-50小時
```

### 路線3：實戰導向（適合有基礎的工程師）
```
6_lenet（復習基礎）→ 7_modern-cnn → 8_regularization → 10_transfer-learning → 11_AI-assisted
完成實戰項目
總時長：約25-30小時
```

### 路線4：面試準備（適合求職者）
```
重點：1-6（基礎概念）+ 7（架構理解）+ 9（可視化）
刷題：實現各種CNN架構
準備：常見面試題目
總時長：約30-35小時
```

---

## 💻 環境配置

### 基本要求
```bash
Python >= 3.8
PyTorch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.19.0
matplotlib >= 3.3.0
```

### 完整依賴（包含進階內容）
```bash
# 核心依賴
pip install torch torchvision torchaudio

# 數據處理
pip install numpy pandas pillow opencv-python

# 可視化
pip install matplotlib seaborn

# 機器學習工具
pip install scikit-learn

# 實驗管理
pip install tensorboard
pip install wandb

# 超參數優化
pip install optuna
pip install ray[tune]

# AutoML
pip install autokeras

# 模型部署
pip install onnx onnxruntime

# 可視化工具
pip install grad-cam
```

### 硬體建議
- **CPU訓練**：可以完成所有教程，但較慢
- **GPU訓練**（推薦）：
  - 最低：4GB顯存（GTX 1650）
  - 推薦：8GB顯存（RTX 3060）
  - 理想：12GB+顯存（RTX 3080/4080）

---

## 📊 數據集

本教程使用以下數據集：

| 數據集 | 大小 | 圖像尺寸 | 類別數 | 用途 |
|--------|------|----------|--------|------|
| Fashion-MNIST | 70K | 28×28 | 10 | 基礎訓練 |
| CIFAR-10 | 60K | 32×32 | 10 | 中級訓練 |
| ImageNet (預訓練) | 1.2M | 224×224 | 1000 | 遷移學習 |

所有數據集會自動下載（通過torchvision）。

---

## 🎓 練習題

每個章節都包含：
- **理論題**：鞏固概念理解
- **編程題**：實踐能力提升
- **項目題**：綜合應用

### 推薦項目

#### 初級項目
1. **手寫數字識別**（MNIST）
   - 實現LeNet
   - 達到99%+準確率

2. **服裝分類**（Fashion-MNIST）
   - 嘗試不同架構
   - 對比性能

#### 中級項目
3. **CIFAR-10圖像分類**
   - 使用遷移學習
   - 應用數據增強
   - 達到90%+準確率

4. **貓狗分類**
   - 使用預訓練模型
   - 實現Grad-CAM可視化
   - 部署模型

#### 高級項目
5. **自定義數據集分類**
   - 收集自己的數據
   - 完整的訓練流程
   - 使用AI工具優化
   - 部署到生產環境

6. **迷你ImageNet挑戰**
   - 設計自己的架構
   - 超參數優化
   - 與經典架構對比

---

## 📝 學習建議

### 學習方法
1. **理論與實踐結合**
   - 先理解原理
   - 然後動手實現
   - 最後做實驗驗證

2. **循序漸進**
   - 不要跳過基礎章節
   - 確保理解了再往下學
   - 完成練習題

3. **做筆記**
   - 記錄關鍵概念
   - 總結常見問題
   - 整理最佳實踐

4. **實踐為主**
   - 運行所有代碼
   - 修改參數觀察效果
   - 嘗試自己的想法

5. **使用AI助手**
   - 遇到問題先思考
   - 使用AI幫助理解
   - 驗證AI的回答

### 常見問題

**Q: 我需要多長時間完成？**
- 快速入門：2-3週（每天2-3小時）
- 系統學習：6-8週（每天2-3小時）
- 完全精通：3-6個月（包括大量實踐）

**Q: 我沒有GPU怎麼辦？**
- 可以使用Google Colab（免費GPU）
- 可以使用Kaggle Notebooks
- 基礎內容CPU也能完成

**Q: 我應該從哪裡開始？**
- 有深度學習基礎：從第6章開始
- 完全新手：從第1章開始
- 只想快速應用：直接學第10章（遷移學習）

**Q: 如何獲得幫助？**
- 使用AI助手（ChatGPT、Claude）
- 參考第11章的調試技巧
- 查看PyTorch官方文檔
- 在GitHub Issues提問

---

## 🔗 相關資源

### 官方文檔
- [PyTorch官方文檔](https://pytorch.org/docs/stable/index.html)
- [torchvision模型](https://pytorch.org/vision/stable/models.html)

### 論文
- [LeNet](http://yann.lecun.com/exdb/lenet/)
- [AlexNet](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [VGG](https://arxiv.org/abs/1409.1556)
- [GoogLeNet](https://arxiv.org/abs/1409.4842)
- [ResNet](https://arxiv.org/abs/1512.03385)

### 課程和書籍
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [CS231n: CNN for Visual Recognition](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### 工具和庫
- [Weights & Biases](https://wandb.ai/)
- [Optuna](https://optuna.org/)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)

---

## 📈 更新計劃

### 已完成 ✅
- [x] 基礎CNN概念（1-6）
- [x] 現代CNN架構（7）
- [x] 正則化和數據增強（8）
- [x] CNN可視化（9）
- [x] 遷移學習（10）
- [x] AI輔助開發（11）

### 計劃中 🚧
- [ ] 目標檢測入門（YOLO, Faster R-CNN）
- [ ] 圖像分割基礎（FCN, U-Net）
- [ ] 輕量級網路（MobileNet, EfficientNet）
- [ ] Vision Transformer入門
- [ ] 更多實戰項目

---

## 🤝 貢獻

歡迎貢獻！如果你發現任何問題或有改進建議：
1. 提交Issue
2. 提交Pull Request
3. 分享學習心得

---

## 📜 版權聲明

本教程基於開源社區的資源整理和擴展而成。
- 基礎部分（1-6）：改編自經典教材
- 進階部分（7-11）：原創內容

僅供學習使用，禁止商業用途。

---

## 🌟 開始學習

準備好了嗎？讓我們從 [第1章：為什麼需要卷積](1_why-conv.ipynb) 開始吧！

或者，如果你想快速上手實戰，可以直接跳到 [第10章：遷移學習與微調](10_transfer-learning.ipynb)。

祝學習愉快！🚀
