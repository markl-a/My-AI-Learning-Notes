# Time Series + LLM 混合管線:從預測到「會說人話」的時序系統

> 對應 [`./Time_Series_Tabular_FM_2024-2026.md`](./Time_Series_Tabular_FM_2024-2026.md) §6;搭配 [`./Chronos_Moirai_zero_shot_forecasting.md`](./Chronos_Moirai_zero_shot_forecasting.md);因果視角 [`../../17.Causal_ML/`](../../17.Causal_ML/)

---

## 1. 為何需要「時序 + LLM」混合

過去十年的時序系統只負責「給數字」:輸入歷史 → 輸出 forecast。但 2024 年起,業務端的提問變成「**為什麼**這週銷售下滑」「**如果**油價漲 10% 會怎樣」「下個峰值**何時**到、要不要先補貨」—— 這三類問題分別對應**預測、解釋、互動**,只給數字滿足不了。

LLM 不能取代時序模型(數值精度太差),時序模型不能取代 LLM(不會講故事、不會推理因果鏈)。**真正的生產系統是兩者串接**:時序基礎模型(TSFM)負責數,LLM 負責對話、解釋、what-if 推演。這正是 2024–2025 出現「Time-LLM / LLMTime / TimeCopilot / TimeOmni」這一整波研究的動機。

## 2. Time-LLM (ICLR 2024):reprogramming 路線

[Jin et al., 2024](https://arxiv.org/abs/2310.01728) 提出一個關鍵思路 —— **凍結 LLM 主幹(LLaMA / GPT-2),只訓練一個 reprogramming 層**,把時序 patch 投影到 LLM 詞表中的「text prototype」(可解釋的少量語意原型,如 “short up”、“steady”、“spike”),再餵進 LLM。Forecast 用一個輕量輸出層接出來。

優點:**參數量極小**、可繼承 LLM 的 few-shot / 跨域泛化能力,在 ETT、Weather、Traffic 多個 benchmark 上 SOTA。缺點:仍需 supervised fine-tune reprogramming 層,並非真 zero-shot;且面對 Chronos-2 / Moirai 2.0 等純時序 FM,純 accuracy 已不佔上風。

## 3. LLMTime (NeurIPS 2023):把數字當 token 直接餵 GPT

[Gruver et al., 2023](https://arxiv.org/abs/2310.07820) 更激進 —— **把時序的每個數字編成 digit token**(處理掉 BPE 切碎小數的問題),讓 GPT-3 / GPT-4 **完全 zero-shot** 直接 next-token 預測。結果令人意外:在 Darts、Monash、Informer benchmark 上,GPT-4 zero-shot 與當時最佳專用模型打平甚至超越。

意義:LLM 的序列建模能力**本身就涵蓋時序**,不需要任何 fine-tune。但實務缺點明顯:token 消耗極高(長序列直接燒錢)、長 horizon 易發散、tokenizer 對特殊數值不穩定。**LLMTime 啟發了後續一切 hybrid 思路**,但生產上很少直接用。

## 4. TimeOmni-1 (2025):時序推理

2025 出現的 TimeOmni / Time-R1 等線索是把 **chain-of-thought 推理**搬到時序領域 —— 不只給未來數字,而是輸出「**因為 t-7 有促銷尖峰、t-3 weekday 下滑、結合外生天氣回升,故預測 t+1 為 …**」的推理鏈。配合 RLHF / RLAIF,讓模型學會在時序語境下做多步推理,而非 black-box 出數。這條線目前還在學術階段,但已預告 2026 的方向。

## 5. 生產建議(實事求是)

| 需求 | 推薦 |
|------|------|
| **純 accuracy、追 MSE/MAPE** | **TSFM** —— Chronos-2 / Moirai 2.0 / TimesFM zero-shot,別繞路 |
| 需要「預測 **+** 自然語言解釋」 | **兩段式管線**:TSFM 給 forecast → LLM 給 narrative |
| 用戶要互動 / what-if | TSFM + LLM + 工具呼叫(TimeCopilot 範式) |
| 跨域、樣本極少 | Time-LLM reprogramming 仍有空間 |
| 全程文字介面 demo | LLMTime(注意 token 成本) |

**核心心法**:不要用 LLM 直接預測數字,讓 LLM 做它擅長的事(解釋、對話、調工具),數字交給 TSFM。

## 6. 可執行範例:Chronos-Bolt + GPT-4o-mini 兩段式管線

```python
import pandas as pd
from chronos import ChronosBoltPipeline
from openai import OpenAI
import json

# 1) TSFM 預測
pipe = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-base")
history = pd.read_csv("sales_daily.csv")["sales"].values[-180:]
forecast = pipe.predict(context=history, prediction_length=30,
                        quantile_levels=[0.1, 0.5, 0.9])

# 2) 組 prompt 餵 LLM
client = OpenAI()
prompt = f"""
你是時序分析師。歷史 180 天:{history.tolist()}
未來 30 天預測 (P10/P50/P90):{forecast.tolist()}
回傳 JSON,鍵為:peak_date, peak_value, anomalies, narrative。
"""
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": prompt}],
)
report = json.loads(resp.choices[0].message.content)
print(report["narrative"])  # 例:「預計第 12 天達峰值 ...，因週末效應 + 月初發薪」
```

50 行內、雙模型協作、輸出結構化 JSON,可直接接 BI 儀表板。

## 7. TimeCopilot 範式:對話式 what-if

[TimeCopilot](https://github.com/AzulGarza/TimeCopilot) 把上述管線包成 **agent + tool**:LLM 是 orchestrator,TSFM、外生資料 API、異常偵測都是 tool。使用者可問「**如果下週氣溫降 5°C,銷售會怎樣?**」,LLM 改寫 covariate、呼叫 TSFM 重跑、用自然語言回覆。這比靜態 forecast 更接近「分析師」體驗,也是 Snowflake / Databricks 2025 內建的方向。

## 8. LLM 警惕

- **Tokenizer 不穩**:`0.123` 可能被切成 `0`/`.`/`123`,不同模型行為差異大;務必走 LLMTime 的 digit tokenization 或讓 LLM 只讀「敘述」而非原始數列
- **長序列易出錯**:>500 點數列直接餵 LLM 通常崩,改餵摘要(均值、極值、季節性 + 最近 N 點)
- **Token 預算炸裂**:GPT-4 每次跑全量歷史成本爆炸,生產一定要先用 TSFM 壓縮成 forecast + 摘要
- **數值幻覺**:LLM 改寫數字會說謊,**最終數字一律以 TSFM 為準**、LLM 只負責文字
- **時區/單位**:LLM 常忽略時區、單位轉換,提示詞要顯式註明

## 9. 真實 case:零售長官「為什麼這週銷售下滑」

業務場景:CEO 不會去看 forecast 曲線,他在 Slack 打字問「為什麼這週下滑?下週會回來嗎?」

後端流程:
1. LLM 解析意圖 → 抓近 90 天 + 同期去年資料
2. 跑 Chronos-2 → 拿 forecast + prediction interval
3. 跑異常偵測(STL + IQR)→ 找 outlier 點
4. 拉外生資料:天氣、促銷檔期、競品價、節慶
5. LLM 把 (1)–(4) 編成 200 字 narrative,附 1–2 個關鍵圖
6. 回 Slack:「本週下滑 8%,主因週三豪雨 + 競品促銷;**TSFM 預測下週回升至基線 ±2%**;建議週五加碼會員折扣。」

這就是 hybrid 的真正價值 ——**把分析師日常 30 分鐘的事壓到 30 秒**。

## 10. 與 Causal ML 結合

純預測回答「會發生什麼」,因果回答「**為什麼**會發生 / **如果**改變 X 會怎樣」。時序 + LLM hybrid 天然適合接 causal:
- **DoWhy / EconML / CausalForest** 算 ATE / CATE → LLM 翻成人話
- **Synthetic Control** 評估某次促銷的反事實影響 → LLM 撰寫週報
- **Granger / PCMCI** 找滯後關係 → LLM 解釋「為何天氣領先銷售 3 天」

詳見 [`../../17.Causal_ML/`](../../17.Causal_ML/)。**2026 的 hybrid 一定會把 causal 納入 tool**,讓 LLM 不只解釋相關性、而能說因果。

## 11. 2026 走向:時序原生多模態 LLM

當前 hybrid 是「兩個模型串接」,2026 趨勢是**時序原生多模態 LLM**:
- 同時吃**時序數列、時序圖(視覺)、文字描述、外生事件**作為原生 modality
- 內建 forecast head + 文字 head,單一前向就同時出數字與解釋
- 跨域 transfer:在金融學的「峰值偵測」能力,直接遷到醫療生理訊號

線索:Google Gemini 2.5 已開始 native 處理「圖表圖像 + 數值」,Anthropic / Meta 也釋出多模態時序的 paper。當 vision-time-language 三模態打通,「拍一張 Grafana 截圖丟給 LLM,它就能說異常 + 預測 + 建議」會成為日常 —— 也是 hybrid pipeline 的終局。

---

**一句話總結**:純預測選 TSFM,要對話選 hybrid;LLM 是分析師、不是預測器。
