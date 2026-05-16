> 對應 [`./Time_Series_Tabular_FM_2024-2026.md`](./Time_Series_Tabular_FM_2024-2026.md) §2

# TabPFN v2 上手筆記:不訓練、不調參、2.8 秒擊敗 4 小時的 XGBoost

## 1. TabPFN 是什麼:把「訓練」搬進 forward pass

TabPFN 的全名是 **Tabular Prior-Data Fitted Network**。它跟一般 ML 模型最大的差別在於:**它不對你的資料訓練模型**。整個流程顛倒過來——TabPFN 在預訓練階段就已經看過約 1.3 億份「合成資料集」,模型學到的是「面對任意一份小型表格,如何在 forward pass 中近似貝氏後驗推論」。

換句話說,傳統 sklearn pipeline 是「fit(X, y) → 更新模型權重 → predict」;TabPFN 是「把整份訓練資料當成 prompt 餵給 transformer → in-context learning → 一次 forward pass 直接吐出預測」。模型權重從頭到尾不動。這跟 LLM 的 in-context learning 是同一個機制,只是把 token 換成了表格的 row。

理論上的詮釋是:PFN 在預訓練時從一個資料生成的 prior 採樣大量任務,優化「給定 context、預測下一筆 label」的 loss,最終學到的就是該 prior 下的 Bayesian posterior predictive distribution 的近似。所以使用者拿到的不是一個「分類器」,而是一個「近似貝氏推論機」。

## 2. TabPFN v2(*Nature* 2025)的突破

Hollmann 等人 2025 年 1 月發表於 *Nature* 的論文是這條線的引爆點。v2 在 OpenML CC18、TabZilla 等基準上,**2.8 秒**(單次 forward pass)就贏過了 XGBoost / CatBoost / LightGBM 各自調參 4 小時、再做 ensemble 的成績。在 AutoML Benchmark 上,v2 也壓制了 AutoGluon。

這件事的意義不只是「快」,而是「徹底改寫小資料 baseline 的標準」。過去 Kaggle / Tabular 競賽的鐵則是「先試 GBDT、再考慮 stacking」;現在「先試 TabPFN」變成更合理的起手式。

## 3. 適用範圍(超出就降級)

v2 的舒適區是清楚的:

- **樣本 ≤ 10,000 列**
- **特徵 ≤ 500 維**
- **類別 ≤ 10**(分類任務)
- 同時支援數值 / 類別特徵、缺失值、離群值

超過邊界會發生什麼?準確度下降、推論時間飆升(O(N²) 的 attention 成本)、GPU 記憶體爆炸。官方在 ≥ 10K 樣本時會自動 subsample 或拒絕運行(取決於 ignore_pretraining_limits 旗標)。後面會介紹 TabPFN-2.5 與 TabICL 把這個邊界推開。

## 4. 30 行 Python 上手

```bash
pip install tabpfn
# GPU 強烈建議;CPU 可跑但慢
```

```python
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

clf = TabPFNClassifier(device="cuda", n_estimators=8)  # 8 個內部 ensemble member
t0 = time.time()
clf.fit(Xtr, ytr)           # 幾乎不做事,只把資料搬到 device
proba = clf.predict_proba(Xte)[:, 1]
print(f"AUC = {roc_auc_score(yte, proba):.4f}  wall = {time.time()-t0:.2f}s")
```

API 完全 scikit-learn 兼容,所以可以無痛丟進 `Pipeline` 或 `cross_val_score`。

## 5. TabPFNRegressor 與 TabPFNClassifier

v2 新增了完整的迴歸支援:

```python
from tabpfn import TabPFNRegressor
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)
reg = TabPFNRegressor(device="cuda")
reg.fit(X[:5000], y[:5000])
# 不只給點估計,還能拿到完整 predictive distribution
output = reg.predict(X[5000:6000], output_type="full")
# output["mean"], output["median"], output["quantiles"], output["logits"]
```

迴歸器最大的賣點是它原生輸出 **後驗分位數**——做不確定性估計時不用再外掛 conformal prediction 或 quantile regression,直接拿。

## 6. TabPFN-TS(時序)與 TabICL(大資料)

**TabPFN-TS**(NeurIPS 2024 workshop)把時序預測重新表述為「(time features, lag features) → y」的 tabular regression 問題,直接用 v2 backbone,**11M 參數**的小模型在 GIFT-Eval、fev-bench 上贏過 Chronos-Large(710M)。對於 univariate 與帶 covariate 的 forecasting 都能 zero-shot。

**TabICL**(ICML 2025,Inria Soda)是另一條路線。它換掉 v2 的 row-token 結構,改成 **column-then-row attention** 兩階段架構,先壓出 row embedding 再做 ICL,把可用樣本量推到 **100K(訓練時)/ 500K(推論時透過 CPU offload)**。在 ≥ 10K 樣本的 53 個資料集上,TabICL 超越 TabPFNv2 和 CatBoost,同時在大表上比 TabPFN-2.5 快約 10×。

## 7. TabPFN-2.5(2025/11):再往上一階

Prior Labs 在 2025 年 11 月發表 TabPFN-2.5(arXiv 2511.08667)。重點:

- 設計目標放大到 **50,000 樣本 × 2,000 特徵**(對 v2 是 20× 資料 cell)
- 在 TabArena-Lite 對 default XGBoost **100% 勝率**(≤10K 樣本子集)
- 單次 forward pass 即匹配 AutoGluon 1.4 extreme mode 調 4 小時的 ensemble
- 提供 **Real-TabPFN-2.5**:在真實資料上 fine-tune 的版本,精度再上一階

實務上 2.5 已經是新的 small / medium tabular baseline 預設選項。

## 8. vs XGBoost 同條件對比

```python
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time

# UCI adult(二元分類,~48K 列;為了符合 v2 邊界,subsample 到 8K)
ds = fetch_openml("adult", version=2, as_frame=True)
X = ds.data.select_dtypes(include="number").fillna(0)
y = (ds.target == ">50K").astype(int)
X, y = X.sample(8000, random_state=0), y.loc[X.index]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

# --- XGBoost(default) ---
t0 = time.time()
xgb = XGBClassifier(n_estimators=500, tree_method="hist", eval_metric="logloss")
xgb.fit(Xtr, ytr)
auc_xgb = roc_auc_score(yte, xgb.predict_proba(Xte)[:, 1])
t_xgb = time.time() - t0

# --- TabPFN v2 ---
t0 = time.time()
clf = TabPFNClassifier(device="cuda", n_estimators=4)
clf.fit(Xtr, ytr)
auc_tab = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
t_tab = time.time() - t0

print(f"XGBoost: AUC={auc_xgb:.4f} wall={t_xgb:.1f}s")
print(f"TabPFN : AUC={auc_tab:.4f} wall={t_tab:.1f}s")
```

預期結果:TabPFN 的 wall time(含一次性 GPU 暖機)落在 5–15 秒,AUC 通常領先 default XGBoost 1–3 個百分點。若把 XGBoost 用 Optuna 跑 100 trial,可以追平,但時間成本是兩個量級的差距。

## 9. 限制與陷阱

- **資料量超界**:>10K(v2) / >50K(2.5)後要不就 subsample、要不就轉去 TabICL。盲目跑下去會 OOM。
- **特徵維度爆炸**:>500 維就要先做特徵選擇或 PCA,否則 attention 計算量無法承受。
- **極端類別不平衡**:正類 < 1% 時 ICL 的 prior 配適會偏弱,要搭配 sample weighting 或 threshold 校正。
- **超多類別**:>10 類分類就超出預訓練 prior,效果掉得很快;NLP-style 多 label 問題不適合。
- **強時間 / 空間結構**:純 tabular 假設 iid;若資料有 leakage 風險或 panel 結構,要自己處理 split。
- **可解釋性**:沒有 feature importance、沒有 tree path。要解釋只能事後接 SHAP(且要 KernelSHAP 因為沒有 tree 結構),成本高。

## 10. 何時用 TabPFN、何時用 GBDT

決策清單:

- **N ≤ 10K、d ≤ 500、要快、不想調參** → TabPFN v2 / 2.5。
- **N 介於 10K–100K** → TabPFN-2.5 或 TabICL,視 GPU 預算選擇。
- **N ≥ 100K** → LightGBM / XGBoost 仍是務實首選;TabICL 可作為對照組。
- **要 SHAP / 規則抽取 / 線上增量更新** → GBDT。
- **要 calibrated probability 與 quantile** → TabPFNRegressor 原生支援,GBDT 要外掛。
- **生產環境只有 CPU、單筆低延遲推論** → GBDT 仍占優,TabPFN 在 CPU 上吃力。
- **小樣本臨床 / 科學資料(N < 500)** → TabPFN 幾乎是免費的精度提升,強烈推薦。

## 11. 生產整合的現實

- **延遲**:單筆推論時 TabPFN 仍需把整份「context 訓練資料」放在 GPU memory 並重算 attention,單次推論在 N=8K 時 ~1–3 秒(A100)。對 < 100ms SLA 的線上服務不適合直接上。
- **Batch inference**:把多筆 query 拼成 batch 一次預測,單筆均攤可降到 ~10ms 等級,是目前最務實的部署形態。
- **ONNX 匯出**:官方目前未提供穩定 ONNX 路徑(注意 ICL 機制需要 context 一起送進去,跟一般靜態圖匯出邏輯不合)。實務上是用 TorchScript 包,或乾脆 Triton 上直接跑 PyTorch backend。
- **Context caching**:Prior Labs 在 2.5 版起提供「fit 完之後快取 context 表徵」的選項,後續 predict 不必重算 row encoder,可把單筆延遲再降一階。
- **License**:v2 採 Prior Labs Academic License(商用需洽談);TabICL 為 BSD-3,商用友善。選型時要看清楚。

## 12. 真實場景

- **臨床預測**:N 通常 200–2000,特徵 20–80。TabPFN v2 在多份 medical tabular benchmark(MIMIC subsets、UCI heart、Pima)上,沒做任何特徵工程就贏過調過的 XGBoost,且原生給出不確定性區間,對醫療決策直接可用。
- **Kaggle Tabular Playground Series**:作為第一個 baseline 跑 5 分鐘,通常能落在 leaderboard 前 30%。再用它的預測值當特徵餵 GBDT stacking,是 2025 年以後常見的奪牌組合。
- **A/B 測試後分析**:處理 heterogeneous treatment effect 時,TabPFNRegressor 的 posterior quantile 直接拿來估 CATE 的不確定性,省下接 BART / causal forest 的麻煩。
- **科學表格資料**:材料科學、化學、生物標記資料集普遍 N < 5000,正是 TabPFN 的甜蜜點。Hollmann 在 *Nature* 論文中已展示這類場景的廣泛優勢。

---

**結論**:對於小到中型表格任務,TabPFN 把「fit + tune + ensemble」三步驟壓縮成一次 forward pass,而且贏。它不是要取代 XGBoost,而是改寫了「baseline 應該長什麼樣」。2026 年起,任何 tabular 任務在動手做 feature engineering 之前,先丟一次 TabPFN-2.5,已經是新的職業 reflex。

Sources:
- [Accurate predictions on small data with a tabular foundation model (Nature 2025)](https://www.nature.com/articles/s41586-024-08328-6)
- [TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models (arXiv 2511.08667)](https://arxiv.org/abs/2511.08667)
- [TabICL: A Tabular Foundation Model for In-Context Learning on Large Data (arXiv 2502.05564)](https://arxiv.org/abs/2502.05564)
- [From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting (arXiv 2501.02945)](https://arxiv.org/abs/2501.02945)
- [PriorLabs/TabPFN GitHub](https://github.com/PriorLabs/TabPFN)
- [TabPFN-2.5 Model Report (Prior Labs)](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- [soda-inria/tabicl GitHub](https://github.com/soda-inria/tabicl)
