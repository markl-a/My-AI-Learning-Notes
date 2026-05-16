# Chronos / Moirai / TimesFM:時序基礎模型 zero-shot 實戰

> 對應 [`./Time_Series_Tabular_FM_2024-2026.md`](./Time_Series_Tabular_FM_2024-2026.md) §1

---

## 1. 時序基礎模型 (TSFM) 為何 2025 火起來

時序預測是企業最值錢的 ML 任務之一(供應鏈、能源、廣告、雲監控),但**長期被 NLP 與 CV 兩波 FM 浪潮繞過**。原因簡單:時序資料**沒有共享 token 空間**,每個感測器、每張電網表、每條 SKU 的尺度與週期都不同;傳統 ARIMA / Prophet / GBDT 個別 fit 個別預測,跨資料集 transfer 幾乎沒有。

2023 年 Nixtla **TimeGPT-1** 開了商業 TSFM 序幕,證明「一個模型 zero-shot 預測任何時序」在工程上可行。2024 一年內五家大廠把武器搬上開源貨架:

- **Google TimesFM** (2024/02, ICML 2024):200M decoder-only,Google Trends + Wikipedia pageviews 預訓練
- **Amazon Chronos** (2024/03):把時序量化成 token,直接餵預訓練好的 T5
- **Salesforce Moirai** (2024/02, ICML 2024):masked encoder,原生 any-variate 多元時序
- **ServiceNow Lag-Llama** (2024):decoder-only + lag features
- **IBM Granite TimeSeries / TTM**:tiny mixer 路線,百萬參數量級

2025 年是「**第二代 TSFM 大爆發**」:
- **Chronos-Bolt** (2024/11):**比原版 Chronos 快 250×**
- **Moirai 2.0** (2025/08):decoder-only,GIFT-Eval #1
- **TimesFM 2.5** (2025/09):200M、16K context、重奪 GIFT-Eval 首位
- **Chronos-2** (2025/10):zero-shot 原生支援多變量與外生變量
- **Datadog Toto** (2025):2.36 兆 telemetry token,專攻可觀測性

**核心范式變化**:從「個別 fit」走向「**一個 pretrained checkpoint + zero-shot 直接出預測**」,類似 GPT-3 對 NLP 的衝擊重演一遍。

---

## 2. Chronos-Bolt(Amazon, 2024/11):把時序當 token 餵 T5

原版 Chronos 把連續值 **量化成離散 token**(scale → quantize → vocabulary),把預訓練好的 T5(encoder-decoder)當「時序 LLM」,在近 1000 億 time-point 上預訓練。**架構幾乎沒改 NLP T5,只換 tokenizer**——這就是它能站在 Hugging Face 巨人肩膀上的關鍵。

**Chronos-Bolt 是 2024/11 升級版**:
1. 把連續時序切成 **patches**(多 time-point 一組)餵 encoder,而非一個一個 token
2. Decoder **直接出 quantile 多步預測**(direct multi-step),不再 autoregressive 逐步解
3. 結果:**比原版 Chronos 快 250×、省記憶體 20×、MASE 還降 5%**;Bolt-Base 甚至比 Chronos-Large 更準且快 600×

Chronos-Bolt 已內建在 **AutoGluon-TimeSeries** 與 **SageMaker JumpStart**,是目前生產線最常見的 zero-shot 預測選擇。

---

## 3. Chronos-2(Amazon, 2025/10):從 univariate 走向 universal

Chronos-2 不是更大,而是**更通用**。120M encoder-only,核心創新是 **group attention 機制**:在同一個 attention group 內可以跨多條序列共享 context,讓 zero-shot 直接吃下:

- **單變量**(原 Chronos 場景)
- **多變量**(多 sensor 共預測)
- **covariate-informed**(把促銷、氣象、節日當外生變量直接 in-context 餵入)

效能:**單張 AWS A10G 每秒 > 300 個 forecasts**,延遲 < 50ms。在 fev-bench(強調 covariate)上把基線甩開一截,在 GIFT-Eval 與 Chronos Benchmark II 也是 SOTA。

**對工程師意義**:過去需要 TFT、N-HiTS 等專門架構處理的多變量 + 外生變量問題,現在 zero-shot 一行 `pipeline.predict()` 就能跑出 baseline。

---

## 4. Moirai 2.0(Salesforce, 2025/11):decoder-only,GIFT-Eval #1

Moirai 1.0(2024/02)是第一個原生 any-variate TSFM(masked encoder),Moirai 2.0 全面重構:

- **架構**:masked encoder → **decoder-only transformer**(更貼合自回歸生成、易 scale)
- **訓練**:36M series 新語料 + GIFT-Eval Pretrain + Salesforce 內部 ops data
- **目標**:quantile forecasting + multi-token prediction
- **效能**:GIFT-Eval **MASE 最佳**(無 test 洩漏的 FM 之中)、CRPS 平 SOTA
- **效率**:**比 Moirai 1.0-Large 快 2×,參數小 30×**

Hugging Face: `Salesforce/moirai-2.0-R-small`,Python 套件 `uni2ts`。

---

## 5. TimesFM(Google, 2024 → 2025/09 v2.5):decoder-only + patch embedding

ICML 2024,**Google Research 第一個時序 FM**。設計理念明擺著抄 ViT:

- **Patch embedding**:32 個連續 time-point → 一個 token(residual MLP 投影)
- **Stacked decoder transformer**:標準 GPT 架構,只是輸入是時序 patch token
- **Output patch**:每步 decoding 出 128 個 time-point(比輸入長,出 horizon 更快)
- **參數**:200M(比 LLM 小很多)
- **預訓練資料**:100 億 time-point,主要是 Google Trends + Wikipedia pageviews

TimesFM 2.5(2025/09)把 **context 從 2K 拉到 16K**(8× 提升)、參數從 500M 縮回 200M,**重奪 GIFT-Eval 首位**。這也是 TSFM 圈跟 LLM 圈一樣開始「小而強」的訊號。

Hugging Face: `google/timesfm-2.5-200m`,套件 `pip install timesfm`。

---

## 6. 快速上手 Chronos-Bolt(~30 行)

電量 7 天預測:

```python
# pip install chronos-forecasting pandas
import pandas as pd
import torch
from chronos import ChronosBoltPipeline

# 1. 讀資料:假設小時粒度,target 是 kWh
df = pd.read_csv("electricity_hourly.csv", parse_dates=["ts"])
context = torch.tensor(df["kwh"].values[-512:])  # 最近 512 小時當 context

# 2. 載 Chronos-Bolt(Base 模型約 200M 參數)
pipeline = ChronosBoltPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# 3. zero-shot 預測未來 168 小時 = 7 天
forecast = pipeline.predict(
    context=context,
    prediction_length=168,
    quantile_levels=[0.1, 0.5, 0.9],  # 10/50/90 分位數
)
# forecast shape: [num_series=1, num_quantiles=3, prediction_length=168]

# 4. 取中位數與信賴區間
median = forecast[0, 1].cpu().numpy()
low, high = forecast[0, 0].cpu().numpy(), forecast[0, 2].cpu().numpy()
print(f"未來 7 天電量中位數: {median.sum():.1f} kWh, 90% CI: [{low.sum():.1f}, {high.sum():.1f}]")
```

**典型延遲**:單張 A10G < 30 ms。完全沒做 fine-tune,zero-shot。

---

## 7. 快速上手 Moirai 2.0(~30 行)

```python
# pip install uni2ts
import torch
import pandas as pd
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

SIZE = "small"   # small/base/large
PDT = 168        # prediction length = 168 小時
CTX = 512        # context length

module = Moirai2Module.from_pretrained(f"Salesforce/moirai-2.0-R-{SIZE}")
model = Moirai2Forecast(
    module=module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size=32,
    num_samples=100,           # Monte Carlo 抽樣數
    target_dim=1,              # 單變量;多變量改成 D
    feat_dynamic_real_dim=0,   # 動態外生變量維度
    past_feat_dynamic_real_dim=0,
)

df = pd.read_csv("electricity_hourly.csv", parse_dates=["ts"])
past = torch.tensor(df["kwh"].values[-CTX:]).unsqueeze(0).unsqueeze(-1)  # [1, CTX, 1]
past_observed = torch.ones_like(past, dtype=torch.bool)
past_is_pad = torch.zeros(1, CTX, dtype=torch.bool)

with torch.no_grad():
    forecast = model(past_target=past, past_observed_target=past_observed,
                     past_is_pad=past_is_pad)
# forecast shape: [num_samples=100, PDT, target_dim=1]
median = forecast.median(dim=0).values.squeeze().numpy()
print(f"未來 7 天中位數預測: {median.sum():.1f} kWh")
```

---

## 8. 多模型比較範例(同一份 NeurIPS competition data)

針對 [Make-It-Count / M5 / GIFT-Eval 子集] 同一份資料,用三模型 zero-shot 跑一輪比 MAE:

```python
# pip install chronos-forecasting uni2ts timesfm gluonts
from chronos import ChronosBoltPipeline
import timesfm, torch, numpy as np
from sklearn.metrics import mean_absolute_error

def eval_model(pred_fn, name, test_y):
    yhat = pred_fn()                            # 各模型給出 median forecast
    mae = mean_absolute_error(test_y, yhat)
    print(f"{name:<15s}  MAE = {mae:.3f}")
    return mae

# 假設 context (np.ndarray) 與 test_y (未來 168 步真值) 已準備好
ctx_t = torch.tensor(context, dtype=torch.float32)

bolt = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base",
                                           device_map="cuda")
eval_model(lambda: bolt.predict(ctx_t, 168, quantile_levels=[0.5])[0,0].numpy(),
           "Chronos-Bolt", test_y)

# Moirai 2.0(略,用上一節 forecast.median)
# TimesFM 2.5
tfm = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(backend="gpu",
                                                     per_core_batch_size=32,
                                                     horizon_len=168),
                      checkpoint=timesfm.TimesFmCheckpoint(
                          huggingface_repo_id="google/timesfm-2.5-200m"))
eval_model(lambda: tfm.forecast([context], freq=[0])[0][0], "TimesFM-2.5", test_y)
```

**真實 leaderboard 觀察**(2025 GIFT-Eval):三者差距通常在 **5% MASE 之內**;選型重點不是 0.5% 的精度,而是**延遲、是否支援外生變量、context 長度需求**。

---

## 9. 適用 vs 不適用

**適用**:
- **新時序、無歷史**:cold-start 新品、新感測器、新地區
- **要快 PoC**:30 行 zero-shot 出 baseline,先讓業務看到價值
- **短 horizon (≤ 168 步)**:大多 TSFM 訓練最佳化的範圍
- **少於幾百條序列**,但每條夠長(context ≥ 256)

**不適用**:
- **百萬序列穩定產線**:**GBDT (LightGBM + MLForecast lag features) 仍勝**——成本低 10×、可解釋、SHAP、ONNX 部署 < 1ms
- **強外生變量依賴**(促銷、氣象、節日):用 **Chronos-2** 或老牌 **TFT** / N-HiTS
- **長 horizon (> 1 年)**:TSFM 訓練資料分布偏短,衰減快
- **嚴格延遲 SLA < 5ms**:仍走 GBDT 或預編譯小模型
- **金融量化交易**:謹防 lookahead leakage,且預訓練資料常含 noise pattern

---

## 10. GIFT-Eval benchmark 與資料洩漏陷阱

**GIFT-Eval**(Salesforce 2024 提出)是目前最廣為接受的 TSFM 公開 benchmark,涵蓋 24 個資料集、144K 時序、177M 觀測點,**禁止把測試集用於預訓練**。Moirai 2.0 與 TimesFM 2.5 都在這個 leaderboard 競爭。

**但坊間多個聲稱「MSE 提升 47–184%」的論文,常因把 TimesFM/UniTS/TTM 的 pretrain 集當測試集而虛報**——所謂的「47-184% MSE 提升」就出自這類洩漏比較。**寫研究報告或選型決策時,只引用 GIFT-Eval / fev-bench / Chronos Benchmark II / BOOM 上的數字**,其餘表格一律當作 marketing 看待。

Datadog **BOOM**(2.36 兆 token、observability 領域)是 2025 開源的第二個重要 benchmark,專測雲監控時序。

---

## 11. Time-LLM 混合:TSFM 給數、LLM 給解讀

**Time-LLM**(ICLR 2024)走「reprogramming」:把時序映射成 LLM 詞表中的 text prototype,凍結 LLaMA / GPT 主幹做預測。**LLMTime** 更暴力——直接把數字編成 digit token 給 GPT-3/4。

**心法**:純預測精度,**TSFM (Chronos-2 / Moirai 2.0) 還是贏 Time-LLM**,且成本低 10–100×。Time-LLM 的真正甜蜜點是:

```
TSFM 出預測 → LLM 接收 (預測值, 歷史, 外部新聞)
            → 生成「為何這樣預測、要不要 override、風險點」自然語言報告
```

這種**兩段式管線**是 2025-2026 企業 BI / 風控 dashboard 的主流做法:不要試圖讓 LLM 直接做數字預測,讓它做它擅長的「故事敘述與決策支援」。

---

## 12. 真實 case

- **零售 cold-start 新品需求**:Walmart / Amazon / Wayfair 用 Chronos / Moirai zero-shot 對「上市 < 4 週、無歷史」的 SKU 做初期預測,等累積 12 週後再切回 LightGBM + MLForecast 產線管線
- **能源負載**:北歐電網 + 台電試點用 N-HiTS + 氣象外生變量;新區域、新變電站用 Chronos-2 zero-shot 立即上線
- **雲監控異常**:**Datadog Toto** 直接用 zero-shot forecast 與 Student-t mixture head 對 billions of ephemeral metrics 做 anomaly score,完全不需 per-series 訓練——這是目前最大規模的 TSFM 工業部署案例
- **金融 / 量化**:仍保守(GBDT + transformer hybrid),但 Bloomberg、Two Sigma 已在 sentiment + 多資產 panel 上實驗 Moirai 2.0

**結論**:TSFM 不是要取代 GBDT,而是**把「冷啟動 0 → 1」這段做到 1 行程式碼**;穩定產線 100 → 1000 還是 GBDT 的天下。把這兩條路線並列在你 2026 的 ML 工具帶上,才是務實的時序工程師心智地圖。

---

## Sources

- [Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon (AWS Blog)](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)
- [amazon-science/chronos-forecasting (GitHub)](https://github.com/amazon-science/chronos-forecasting)
- [Chronos-2: From Univariate to Universal Forecasting (arXiv 2510.15821)](https://arxiv.org/abs/2510.15821)
- [Introducing Moirai 2.0 (Salesforce Blog)](https://www.salesforce.com/blog/moirai-2-0/)
- [Moirai 2.0: When Less Is More for Time Series Forecasting (arXiv 2511.11698)](https://arxiv.org/abs/2511.11698)
- [A decoder-only foundation model for time-series forecasting (Google Research)](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [google-research/timesfm (GitHub)](https://github.com/google-research/timesfm)
- [Introducing Toto: A state-of-the-art time series foundation model by Datadog](https://www.datadoghq.com/blog/datadog-time-series-foundation-model/)
- [Toto and BOOM unleashed (Datadog Blog)](https://www.datadoghq.com/blog/ai/toto-boom-unleashed/)
