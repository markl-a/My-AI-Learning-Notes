# 22. 自動化 AI 研究 / Self-Improving AI / RSI (2024-2026)

> 對應 [全景圖](../2024-2026_AI完整領域全景圖.md) #16
> 當 AI 開始「研究 AI 自己」,我們處於曲線的哪一段?

> **⚠️ 前置 / Prerequisites**
> 本 deep-dive 為 2026 frontier briefing 風格(~150 行密集 briefing,每節列當代 SOTA + 選型建議),**預設讀者已掌握**:
> 1. **RL 基礎(value/policy gradient、PPO、reward model)**(對應 repo:[強化學習與LLM整合指南](../2.深入LLM模型工程與LLM運維/10.進階話題/強化學習與LLM整合指南.md))
> 2. **Constitutional AI / RLAIF 概念**(對應 repo:[2.深入LLM模型工程與LLM運維/6.偏好對齊 (Alignment) 技術](<../2.深入LLM模型工程與LLM運維/6.偏好對齊 (Alignment) 技術/>))
> 3. **MCTS / search 概念**(若 repo 內無,先看:[Sutton & Barto《Reinforcement Learning》Ch. 8.7](http://incompleteideas.net/book/the-book-2nd.html))
>
> 缺乏 RL 與 alignment 背景直接讀本檔,「reward hacking / scalable oversight / AI Scientist self-judging」等核心張力會無法 grok,建議先補前置。
>
> **延伸 / 反向連結**:[2.深入LLM模型工程與LLM運維/12.推理模型應用](../2.深入LLM模型工程與LLM運維/12.推理模型應用/) | [21.AI_Forecasting_Economics](../21.AI_Forecasting_Economics/README.md)

> **⚠️ 鮮度與可信度說明 / Freshness & Reliability**
> 本章涉及 **AI 自動化研究、AlphaEvolve、Sakana AI Scientist、AGI 時程、RSI 風險、ASL 級別、Anthropic vs OpenAI 比較** 等 2024-2026 frontier 議題,**特別容易混入媒體推測與業內傳聞**。本章內容混合三類來源:
> 1. **論文 / 官方 blog / 已發表 benchmark**(高信任度)
> 2. **媒體報導 / vendor 公告 / 工程師訪談**(屬「報導」非「事實」)
> 3. **AI agent 整理 + 我手動驗證**(可能有誤標,尤其 ICLR workshop 評分、AlphaEvolve 具體成果數字、ASL-X 預測時程)
>
> 任何要拿來做決策或公開引用的具體陳述(時程預測、AGI 倒數、ARR 對比、ICLR/NeurIPS 評分、AlphaEvolve compute 節省百分比),**請以一手 source 為準**。本章**未逐一回查 source**,對「RSI 是 frontier lab 監測門檻」這種 *結構性敘事* 較高信心,對「2026 末突破 ASL-3」這種 *時點預測* 較低信心。

---

## 1. 概念光譜:從工具到 RSI

四個遞進層次:
1. **Automated AI research(自動科研助手)**——AI 取代研究流程中的某些勞動環節。代表:OpenAI Deep Research、Gemini Deep Research、Sakana AI Scientist。**現狀:已可商用**。
2. **AI for AI / Closed-loop scientific discovery**——AI 主導完整科研循環。代表:Sakana AI Scientist-v2 通過 ICLR workshop peer review、Google AI co-scientist。**現狀:狹義領域 demo 級,通用領域不可靠**。
3. **Self-improvement**——模型用自身輸出改進自身權重 / 推理。代表:Constitutional AI / RLAIF、DeepSeek-R1-Zero(純 RL self-bootstrap reasoning)、AlphaEvolve(改進 Gemini 訓練 kernel)。**現狀:component-level 真實存在**。
4. **Recursive self-improvement (RSI)**——模型改進的模型再改進模型,進入正反饋飛輪,觸發「智能爆炸」。**現狀:未被觀察到,但 Anthropic / OpenAI / METR 都已視為需 1-3 年內監測的能力門檻**。

## 2. 為什麼 2024-2026 是關鍵窗口

三個獨立訊號同時對齊:
1. Sakana 證明 AI 撰寫的論文可通過真實 peer review
2. AlphaEvolve 證明 LLM-evolutionary loop 可優化生產級基礎設施
3. METR、Anthropic RSP v3、OpenAI Preparedness v2 同步把「AI R&D 自動化」列為需要監測的風險類別

業界共識:**RSI 不再是科幻**,而是 frontier lab roadmap 上 12-36 個月的問題。

## 3. 代表性系統

### Sakana AI Scientist v1 (2024) / v2 (2025)
**v1**(2024-08)首個端到端 LLM-driven 全自動科研系統。**v2**(2025-04, arxiv 2504.08066)拋棄人工模板,引入 **progressive agentic tree search** + 專屬 experiment manager agent。

**里程碑**:2025/03 v2 生成的論文 *"Compositional Regularization: Unexpected Obstacles in Enhancing Neural Network Generalization"* 在 ICLR 2025 workshop 通過 peer review,三位審稿人評分 6/7/6(平均 6.33)。**首次** 純 AI 論文通過 ICLR 流程,2025 由 Nature 報導。

**誠實評估**:這是 workshop(門檻低於主會)、且 Sakana 自報「v2 對 strong template 的 task 反而不如 v1」。屬概念驗證而非常態能力。

### DeepMind FunSearch (2023) → AlphaEvolve (2025)
**FunSearch** (Nature 2024/01):**LLM + 演化搜尋 + 自動評估器** 解決開放數學問題。cap set problem 20 年來最大的漸近下界改進。

**AlphaEvolve** (2025-05):**Gemini 2.0 Flash + Pro 雙模型 ensemble**。三大成就:
- 找到 4×4 複數矩陣乘法用 **48 次純量乘法**(突破 Strassen 1969 的 49 次)
- **TPU 硬體電路**已採用其建議
- 把 Gemini 訓練 kernel 加速 **23%**,使 Gemini 訓練時間縮短 **1%**——**目前最接近 RSI 的真實案例**

### Google DeepMind AI Co-Scientist (2025)
基於 Gemini 2.0 的多 agent 框架:**Generation → Reflection → Ranking → Evolution → Meta-review**。2025 開放給美國 DOE 全部 17 個國家實驗室。
**已驗證**:提出的肝纖維化藥物再利用候選經實驗室驗證有效;預測複雜抗生素抗藥性機制。

### OpenAI Deep Research + ChatGPT Agent (2024-2025)
基於 o3 變體,聚焦 **web research + 結構化報告**。**Humanity's Last Exam** 上 26.6%(當時 SOTA)。2025/07 併入 ChatGPT agent,獲得 visual browser。

### Anthropic 內部 AI for Alignment Research
**「用 AI 自動化對齊研究」是 Anthropic 核心議程**:
- **RLAIF**:用 AI 生成偏好標籤
- **Constitutional Classifiers**:2025/02 開放 7 天 jailbreaking 賞金,183 人花 3000+ 小時無人攻破 universal jailbreak
- Anthropic chief scientist Jared Kaplan(2025/12 *Guardian*):**「2027-2030 之間人類必須決定是否允許 AI 遞迴自我改進——這是領域內最大的決定。」**

## 4. 核心技術組件

### Tree Search over Ideation Space
Sakana v2 用 **agentic tree search**(ToT + MCTS 在 idea space)。AlphaEvolve 的 island-based evolutionary search 是同源思想。**通用模式**:LLM 提供「合理 prior」,自動 evaluator 提供「真實 signal」,搜尋演算法協調兩者。

### LLM-as-Reviewer / Reward Modeling
**問題**:LLM reviewer 是 distribution-内 critic,可能 reward hack。**Eureka** 用真實環境 reward(GPU sim 跑 RL)規避這個問題——這是 closed-loop AI for AI 能否成立的最大瓶頸:**evaluator 必須來自模型分佈之外**。

### Code Synthesis + Execute + Score Loop
**Eureka** (NVIDIA, ICLR 2024):LLM 生成 reward code → NVIDIA Isaac Gym 並行評估 → reward reflection 反饋 → 29 個 RL 環境中 **83% 任務超越人類專家**。

## 5. Evolutionary Algorithm 與 LLM 的合流

**LLM 變成 evolutionary algorithm 的「變異算子」**:
- **Promptbreeder** (DeepMind, ICML 2024):self-referential——演化 mutation prompts 而非僅 task prompts
- **FunSearch / AlphaEvolve**:程式碼即基因,LLM 是 mutation operator
- **Eureka**:RL reward function 演化
- **Lange et al.**:LLM 生成 architecture mutations

**為何重要**:傳統 EA 卡在「變異缺乏語意」,LLM 解決這個瓶頸;LLM 卡在「無 ground truth feedback」,EA 提供 fitness。

## 6. Self-Play 與自我蒸餾的 LLM 回歸

**DeepSeek-R1**(2025/01, Nature 2025):**R1-Zero 完全跳過 SFT**,純用 RL(GRPO)從 base model bootstrap 出 self-verification、reflection、long CoT 行為——重現 AlphaGo Zero「無人類資料的 self-play」精神於語言領域。

**意義**:reasoning 行為可以從可驗證 reward(數學、程式)中**湧現**。一旦推廣到「AI 研究」task 且有自動 evaluator,RSI 的鏈條原則上可閉合。

## 7. Anthropic / OpenAI / DeepMind 公開立場

| 機構 | 立場 | 文件 |
|---|---|---|
| **Anthropic** | RSI 是「最終風險」,2027-2030 為決策窗口。RSP v3 把「AI R&D 自動化」分兩級門檻:可自動化入門研究 → 可使 effective compute scale 加速 35×/年。後者觸發 ASL-4 | RSP v3.0、v3.1 |
| **OpenAI** | 2025/04 Preparedness Framework v2 將 **AI Self-Improvement 列為三大 Tracked Categories 之一** | Preparedness Framework v2 |
| **DeepMind** | AlphaEvolve / AI co-scientist / Genesis Mission 是最具體的 AI-for-AI 投資 | Frontier Safety Framework |

三家共識:**RSI 是 plausible、measurable、severe、irremediable 的風險類別**。

## 8. 與 AGI 時程的關係

### METR Time Horizon 與 RE-Bench
AI agent 能完成的「任務水平時長」每年 **10×** 增長。**RE-Bench**(2024/11):7 個 ML 研究工程環境 8 小時人類專家比較——當給定 2 小時預算,最佳 AI agent 得分 **4× 於人類**;但給人類 8 小時,人類仍勝。

METR 2025/08 *Forecasting Impacts of AI R&D Acceleration* + 2026/02 simpler timelines model 預測 **2032 年 99% AI R&D 自動化**。Aschenbrenner *Situational Awareness*:2027-2028 出現「AI 遠端員工」,進入 RSI。

### 誠實的時程評估
樂觀派(Aschenbrenner、AI 2027)與保守派(METR 2032)差距 5 年,**但兩派都同意 RSI 是這個十年內的事**。實際 bottleneck 不是模型能力本身,而是 (a) 自動 evaluator 設計、(b) 長 horizon 一致性、(c) 計算資源、(d) 安全護欄。

## 9. 風險面

**Alignment 在 RSI 場景的挑戰**:目前所有對齊技術都假設「人類能監督模型輸出」。一旦 AI 改進 AI,監督鏈條中可能出現人類無法理解的中間階段。Anthropic *Alignment Faking* 研究、OpenAI *Sandbagging* 類別都針對此情境。

**「智能爆炸」可信度**:強形式(數週內進入超智能)缺乏實證;弱形式(2-5 年加速 5-10×)有 AlphaEvolve 1% Gemini 訓練加速作為弱證據。

**Hype 與實質的差距**:多數媒體把「AI 寫了論文」當成 RSI 證據,實則 Sakana v2 連通用領域可靠性都未達標。

## 10. 學術 vs 工業差距

**開源側**:Sakana AI Scientist v1/v2、FunSearch、OpenEvolve、Eureka、Promptbreeder 都已開源。

**Frontier lab 側**:Anthropic、OpenAI、DeepMind 內部 AI-for-AI 使用程度**不透明**。已知碎片:AlphaEvolve 改進 Gemini 訓練、Anthropic 用 Claude 自動 red team、OpenAI Codex / Cursor 系工具大量用於內部研究。**「Claude 改 Claude 訓練程式」這種 RSI loop 是否在內部運轉,沒人公開**。

## 11. 可實作的 Mini AI Scientist

**最小可行架構**(LangGraph + Claude / GPT-4,單機):

```
[Ideate Agent] --topic--> [Search Agent (WebSearch)] --refs--> [Critic Agent]
       |                                                          |
       v                                                          v
[Code Agent] --script--> [Executor (sandbox)] --results--> [Review Agent]
       ^                                                          |
       |____________________ revise ______________________________|
                              (max N iters)
```

**實作要點**:
1. State schema 用 Pydantic 定義
2. Code executor 用 `e2b` / Docker sandbox,絕不直接 `exec()`
3. Evaluator 必須客觀:ML task 用 holdout test set,數學用 SymPy
4. Tree search:K=3-5 candidates,N=5 層
5. Cost guard:500K tokens / $20

**起點 repo**:fork `SakanaAI/AI-Scientist-v2` 用 small template(NanoGPT、MNIST)。

---

## 12. 動手範例 — 用 LangGraph 寫 Mini AI Scientist

把 §11 的架構圖翻成可跑的 LangGraph skeleton。**目標**:理解 state passing、node 邊界、cost guard,不是要產 SOTA 論文。**先決**:已讀 §11 + 完成過 LangChain hello-world。

### 架構圖

```
              +-------------+
              |   ideate    |    propose K=3 hypotheses
              +------+------+
                     |
                     v
              +-------------+
              |   search    |    WebSearch refs (literature check)
              +------+------+
                     |
                     v
              +-------------+
              |   critic    |    score & pick best hypothesis
              +------+------+
                     |
                     v
              +-------------+
              |    code     |    write training/eval script
              +------+------+
                     |
                     v
              +-------------+
              |   execute   |    sandbox run, capture metrics
              +------+------+
                     |
                     v
              +-------------+
              |   review    |--no--> back to ideate (revise, max N=3)
              +------+------+
                     |yes
                     v
                  [ done ]
```

### Pydantic State

```python
from pydantic import BaseModel, Field
from typing import Literal

class ScientistState(BaseModel):
    topic: str
    hypotheses: list[str] = Field(default_factory=list)
    references: list[dict] = Field(default_factory=list)
    selected_hypothesis: str | None = None
    code: str | None = None
    results: dict | None = None        # holdout metrics
    review: Literal["accept", "revise"] | None = None
    iter: int = 0
    tokens_used: int = 0               # cost guard
    cost_usd: float = 0.0
```

### 五個 node(skeleton)

```python
from langgraph.graph import StateGraph, END
from anthropic import Anthropic
client = Anthropic()
MAX_TOKENS, MAX_USD, MAX_ITERS = 500_000, 20.0, 3

def call_llm(prompt: str, state: ScientistState) -> str:
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    state.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
    state.cost_usd    += resp.usage.input_tokens*15e-6 + resp.usage.output_tokens*75e-6
    return resp.content[0].text

def ideate(state):
    out = call_llm(f"Propose 3 falsifiable ML hypotheses about: {state.topic}", state)
    state.hypotheses = [h.strip() for h in out.split("\n") if h.strip()][:3]
    return state

def search(state):                          # use Tavily / Exa / Brave
    state.references = [tavily.search(h) for h in state.hypotheses]
    return state

def critic(state):
    out = call_llm(f"Pick most novel & feasible from:\n{state.hypotheses}\nRefs:\n{state.references}", state)
    state.selected_hypothesis = out
    return state

def code(state):
    state.code = call_llm(f"Write a self-contained PyTorch script testing: {state.selected_hypothesis}. Output holdout test acc.", state)
    return state

def execute(state):                         # e2b sandbox — NEVER raw exec()
    state.results = e2b_run(state.code, timeout=600)
    return state

def review(state):
    # IMPORTANT: review uses holdout metrics, NOT LLM-as-judge
    state.review = "accept" if state.results.get("test_acc", 0) > state.results.get("baseline_acc", 0) else "revise"
    state.iter += 1
    return state

def gate(state):
    if state.tokens_used > MAX_TOKENS or state.cost_usd > MAX_USD: return END
    if state.iter >= MAX_ITERS or state.review == "accept":        return END
    return "ideate"
```

### 編譯 + 運行

```python
g = StateGraph(ScientistState)
for n, fn in [("ideate", ideate), ("search", search), ("critic", critic),
              ("code", code), ("execute", execute), ("review", review)]:
    g.add_node(n, fn)
g.set_entry_point("ideate")
g.add_edge("ideate", "search"); g.add_edge("search", "critic")
g.add_edge("critic", "code");   g.add_edge("code", "execute")
g.add_edge("execute", "review")
g.add_conditional_edges("review", gate, {"ideate": "ideate", END: END})
app = g.compile()
final = app.invoke(ScientistState(topic="label smoothing on tiny ViT"))
```

### 評估設計(critical)

**用 holdout test set,不要用 LLM-as-judge**。理由見 README §4「Evaluator 必須來自模型分佈之外」。
- ML task:把 dataset 在進入 graph **前**就切 train/val/test,test set 對 LLM 隱藏,只 review node 看 metric。
- 數學/邏輯 task:用 SymPy / Lean / unit test 當 evaluator。
- 若 task 沒有客觀 metric,**不要嘗試這個 task**——你會做出一個 reward-hacked toy,不是 mini scientist。

### 起點

不要從零寫——fork [`SakanaAI/AI-Scientist-v2`](https://github.com/SakanaAI/AI-Scientist-v2),挑 small template(NanoGPT、MNIST、grokking),把 backend 換成 Claude/Gemini,在 1 個 idea × 1 iter 上跑通,**先確認 token 預算正確再加 loop 層數**。

---

## 2026 自動 AI 研究觀察員地圖

| 軸線 | 關鍵指標 | 觀察點 | 信號意義 |
|---|---|---|---|
| **論文自動化** | AI 論文通過 main conference peer review | NeurIPS / ICML 2026-2027 main track | 第 2 層商業化臨界 |
| **演算法發現** | AlphaEvolve 類成果 | DeepMind / OpenAI 技術報告 | 弱 RSI 落地程度 |
| **能力 horizon** | METR Time Horizon doubling | METR 季度報告 | RSI 時程主指標 |
| **AI R&D 評估** | RE-Bench、SWE-bench、SciCode | 模型 release 時 third-party eval | 自動化研究是否可商用 |
| **政策觸發** | Anthropic ASL-4 / OpenAI High AI Self-Improvement | 各家 system card | 監管 / pause 風險 |
| **開源追趕** | DeepSeek / Qwen / Llama 是否複現 AlphaEvolve | arxiv + HF 開源 release | 能力擴散速度 |
| **內部 deployment** | 各 lab 揭露 AI 在自己訓練 pipeline 的貢獻 | Anthropic / OpenAI / DeepMind blog | 真實 RSI 進度 |

**最終判斷**:2024-2026 是「**AI 開始能科研、但還不能可靠地科研**」的時期。Sakana 證明可能性,AlphaEvolve 證明價值,DeepSeek-R1 證明 RL 路徑,Anthropic / OpenAI 證明風險被認真對待。**RSI 仍未發生**——沒有觀察到自我加速的飛輪——但所有零件已就位。下一個 18 個月的關鍵問題是 **自動 evaluator 設計**(這比模型本身更可能成為 bottleneck)。

---

## References & Sources

本檔由 2026-05 deep-research agent 產出,引用來源散見於各章。原始 agent 在研究階段曾使用以下類型來源:
- 學術論文(arXiv、Nature、Science、NeurIPS/ICML/ICLR proceedings)
- 廠商技術部落格(Anthropic、OpenAI、Google DeepMind、Meta AI、NVIDIA Developer Blog、Microsoft Research)
- 產業分析(SemiAnalysis、Epoch AI、Stratechery、The Information)
- 開源 repo 文件(Hugging Face、GitHub README)

**目前本檔的具體引用連結待補(下一輪 revision)**。讀者引用任何具體數字、發布日期、產品功能前,請以官方 source 為準。
