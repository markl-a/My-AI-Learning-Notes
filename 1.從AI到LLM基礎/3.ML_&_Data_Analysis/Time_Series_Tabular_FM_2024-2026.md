# 時序 / 表格 / 結構化資料 AI 全景 (2024-2026)

> 對應 [全景圖](../../2024-2026_AI完整領域全景圖.md) #9
> 企業真實資料中佔比最高,但長期被 LLM 風潮掩蓋的領域。2024-2026 是「基礎模型範式」首次真正攻入時序與表格的關鍵兩年。

---

## 1. 時序基礎模型 (Time Series Foundation Models)

2023 年 Nixtla **TimeGPT-1** 開啟商業 TSFM 序幕。2024 一年內 Google **TimesFM**(200M decoder-only)、Amazon **Chronos**(時序量化成 token 餵給 T5)、Salesforce **Moirai**(any-variate 多元時序原生支援)、ServiceNow **Lag-Llama** 全部開源。

**2025**:
- **Datadog Toto** — 2.36 兆 token,70% 來自其私有 telemetry,7× 更省參數
- **Moirai 2.0** — decoder-only,登上 GIFT-Eval #1
- **Amazon Chronos-2** (2025/10) — 單張 A10G > 300 forecasts/sec,zero-shot 原生支援多變量與外生變量
- **TabPFN-TS** — TabPFN 衍生

**突破**:**Chronos-Bolt** 比原版 Chronos 快 250×、記憶體少 20×,Base 版甚至比 Chronos-Large 更準且快 600×。

**選型**:新案子先用 Chronos-Bolt / Moirai 2.0 / TimesFM zero-shot 跑一輪當 baseline;發現 zero-shot 已 90% 達標就停下、寫進產線。

**警告**:GIFT-Eval 之外多個 benchmark 因把 TimesFM/UniTS/TTM 預訓練集當測試集,虛報 47–184% 的 MSE 優勢。

## 2. 表格基礎模型 (Tabular Foundation Models)

2022 **TabPFN**(Prior-Data Fitted Networks)是冷門突破,2025 **TabPFN v2** 登上 *Nature*,**2.8 秒內擊敗調了 4 小時的 XGBoost/CatBoost/LightGBM 集成**。

隨後:
- **TabICL** (2025 ICML, INRIA Soda):column-then-row embedding 把規模推到 500K 樣本,在 ≥10K 的 53 個資料集上**同時擊敗 TabPFNv2 和 CatBoost**
- **TabDPT**:用真實 OpenML 資料 ICL 預訓練,首次驗證 tabular FM 的 scaling law
- **TabPFN-2.5** (2025/11):對 default XGBoost 在 ≤10K 樣本 100% 勝率、≤100K 樣本 87% 勝率

**為什麼「重新震驚」傳統 ML 社群**:過去十年 SAINT、TabNet、NODE、FT-Transformer 等深度模型反覆嘗試擊敗 GBDT 都失敗。TabPFN 換了思路 ——**訓練一個能在 forward pass 中「貝氏推論整個資料集」的網路**,從 in-context learning 角度挑戰傳統認知的 GBDT 護城河。

**選型**:**樣本 ≤10K 直接用 TabPFN v2 / TabICLv2**;10K-100K 用 TabICL 或 TabPFN-2.5;**>100K 仍走 CatBoost/LightGBM**。

## 3. 經典 ML 仍主流:XGBoost / LightGBM / CatBoost

**2025 年 Kaggle 與工業界 80%+ 的表格冠軍方案仍是 GBDT**。

- **CatBoost** 在類別特徵密集場景超越 XGBoost 20%+
- **LightGBM** 大資料訓練速度仍無敵手
- **XGBoost** 是穩定中間值

NVIDIA Kaggle Grandmasters Playbook(2025)的七大手法仍以 GBDT 為核心。

**為何 deep learning 在表格上長期落後**:
1. 表格資料**沒有空間/序列局部性**
2. 特徵**異質**,GBDT 對缺值與類別原生友好
3. **小資料、低 SNR、強表格 noise**,神經網路嚴重 overfit
4. GBDT 對超參數**不敏感**
5. **可解釋性與部署成本**(SHAP、ONNX、樹模型 < 1ms inference)壓倒性勝出

## 4. 多變量時序與 Panel Data

Hyndman 的 hierarchical reconciliation(BottomUp/TopDown/MinT)仍是業界標準,**Nixtla HierarchicalForecast** 是事實 Python 實作。Moirai 是第一個**原生 any-variate** 的 TSFM,Chronos-2 在 2025 補上 covariate-informed 預測。

Panel data 的 cross-series transfer 由 **N-BEATS、N-HiTS、PatchTST、iTransformer** 三劍客在 fully supervised 場景仍最強。

**選型**:
- 單序列 → StatsForecast (auto-ARIMA/ETS/Theta)
- 百萬序列 → MLForecast + LightGBM with lag features
- 需 cross-series transfer 與外生變量 → NeuralForecast (TFT, N-HiTS, PatchTST) 或 Chronos-2 zero-shot
- 有層次結構 → HierarchicalForecast 做 reconcile

## 5. 異常偵測

基線:**PyOD**(統計+淺層學習)、**Isolation Forest / LOF / One-Class SVM**;時序專用深度方法:
- **DeepLog**(LSTM log)
- **Anomaly Transformer**(ICLR 2022)
- **TimesNet**(1D→2D 把週期攤平)
- **TranAD**

2025 新方向:**LogLLaMA** 用 LLaMA 做 log 異常偵測、TSFM (Toto, Moirai) 直接做 zero-shot 異常 scoring。

## 6. 時序 + LLM 混合

**Time-LLM**(ICLR 2024)透過 reprogramming 把時序映射成 text prototype,凍結 LLM 主幹做預測。**LLMTime** 把數字編成 digit token,讓 GPT-3/4 直接做 zero-shot 預測。

**心法**:**純預測 accuracy → 仍選 TSFM (Chronos-2/Moirai)** 而不是 Time-LLM(後者成本高、精度未明顯領先);**需「預測 + 解釋 + 互動」→ TSFM 給數,LLM 給故事** 的兩段式管線。

## 7. 資料工程與 Feature Store

- **Feast** — 開源領導者(Linux Foundation),模組化、unbundled
- **Tecton** — 託管金標(Uber Michelangelo 班底)
- **Hopsworks** — lakehouse + feature store 整合,受規管產業偏好
- **Featureform** — 輕量虛擬層

2025 三家都在加上 **vector / embedding feature** 以支援 RAG 與 LLM serving。

**選型**:新創/PoC → Feast + Postgres/Redis;百人以上 ML 團隊 → Tecton;金融/醫療 → Hopsworks。

## 8. 時序資料庫

四強分工:
- **TimescaleDB** = SQL 與關聯能力最強
- **InfluxDB 3.0** = IoT/監控最易上手,DataFusion 重寫解決基數問題
- **QuestDB** = ingest 與 query 速度王者(對 InfluxDB 3 快 12-36×)
- **ClickHouse** = 大規模 OLAP 分析最強

2026 受到廣泛採用的方式是**混合架構**:InfluxDB 即時告警 + ClickHouse 歷史分析 + TimescaleDB 事務控制。

## 9. 產業應用

- **零售需求預測**:GBDT 主場(LightGBM + MLForecast),Walmart/Amazon/Wayfair 試 Chronos/Moirai 做 cold-start
- **能源負載**:N-HiTS + 外生氣象變量,2025 TSFM zero-shot 在台電/北歐電網試點
- **金融量化**:仍保守(GBDT + transformer hybrid,謹防 lookahead)
- **雲端監控**:Toto 主場,Datadog/Grafana 集成
- **預測性維護**:市場 2030 達 $21.3B(27% CAGR)

## 10. 三條路線的決策樹

| 場景 | 首選 | 為何 |
|---|---|---|
| 表格 ≤ 10K 樣本 | **TabPFN v2 / TabICLv2** | 1 秒勝過 4 小時調的 XGBoost |
| 表格 10K-100K | TabICL or CatBoost | TabICL 已能匹敵 |
| 表格 > 100K 或極異質 | **CatBoost / LightGBM** | TSFM 仍受 context 限制 |
| 新時序、無歷史、要快 PoC | **Chronos-Bolt / Moirai 2.0 zero-shot** | 5 分鐘出 baseline |
| 百萬序列、長期穩定產線 | **MLForecast + LightGBM** | 成本低、可解釋、SLA 穩 |
| 跨序列 transfer、外生變量豐富 | **TFT / N-HiTS / Chronos-2** | 多變量原生 |
| 觀測性 / 監控時序 | **Datadog Toto + ClickHouse** | 為 telemetry 而生 |
| 需要「預測 + 解釋」 | TSFM 預測 + LLM 解讀(雙模型) | 各取所長 |

---

## 2026 結構化資料 AI 工程師地圖

**三層金字塔**:

1. **底層(80% 案子)**:精通 **LightGBM / CatBoost / XGBoost + Optuna + SHAP**,熟練 **Pandas / Polars / DuckDB** 做特徵工程,掌握 point-in-time correctness 與 leakage 偵測。
2. **中層(15% 案子)**:會用 **Nixtla 全家桶**(StatsForecast / MLForecast / NeuralForecast / HierarchicalForecast),熟 **PyOD / DeepOD** 做異常,能在 **TimescaleDB / ClickHouse** 上寫 SQL aggregation。
3. **頂層(5% 案子但未來成長最快)**:能 fine-tune 或 zero-shot 部署 **Chronos-2 / Moirai 2.0 / TimesFM / Toto / TabPFN v2 / TabICL**,理解 GIFT-Eval / BOOM 評測陷阱,會寫 **Feature Store(Feast/Tecton)** 與 **TSFM 做混合管線**。

**反 LLM-hype 提醒**:企業 70% 的可貨幣化 ML 仍在表格與時序;與其追每週新 LLM,不如把 TabPFN/Chronos 與你公司 PostgreSQL 裡的真實表整合起來 —— 那才是 2026 結構化資料 AI 工程師的真正護城河。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
