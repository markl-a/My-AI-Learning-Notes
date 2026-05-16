# 03 行為面試 STAR 範例

> **本檔定位**:20 題最常被問的行為面試題,每題給一個 STAR 範例答案。
> 至少 5 個範例取自 [phantom-mesh](https://github.com/markl-a/phantom-mesh) 真實
> 開發場景(provider fallback、SSE 解析、跨平台 build、multi-agent 協調、cost tracking),
> 確保你的答案經得起連環追問。
>
> **Cross-link**:
> - 職涯路徑 → [`01_AI_Engineer_vs_ML_Engineer_職涯路徑.md`](./01_AI_Engineer_vs_ML_Engineer_職涯路徑.md)
> - 履歷打造 → [`02_履歷與作品集打造.md`](./02_履歷與作品集打造.md)
> - 全景圖 → [`../../2024-2026_AI完整領域全景圖.md`](../../2024-2026_AI完整領域全景圖.md)

---

## STAR 框架速記

**S**ituation(背景 ~15 字)→ **T**ask(任務 ~15 字)→ **A**ction(行動 ~30 字)→ **R**esult(結果 ~20 字)
**口頭回答控制在 1.5–2 分鐘**。寫下來約 80 字一段,口語化展開到 200 字左右。

---

## Part A:技術衝突 / 決策(Q1–Q5)

### Q1. Tell me about a time you disagreed with a teammate on a technical decision.

> **S**:phantom-mesh 早期討論 LLM provider 抽象層,要不要做 unified message format。
> **T**:我主張統一,另一位 contributor 認為應該各 provider 直連、留原始 schema。
> **A**:我寫了一份 spike,實測三個 provider 真實差異(stream event 結構、tool call schema、refusal 格式),把資料攤在 PR description,提出 "thin adapter + 原 schema 透過 metadata 透傳" 折衷方案。
> **R**:他同意了,折衷方案進 main。後來新增 Gemini 時只動 80 行,驗證設計合理。學到 disagreement 用 code + data 解,不要用嘴。

### Q2. Tell me about a difficult technical decision you made.

> **S**:RAG 系統 latency p99 達 3.2s,SLA 是 2s。
> **T**:評估是否替換向量庫(Qdrant → pgvector)以簡化 stack,還是繼續優化現有 stack。
> **A**:做 capacity estimation + 2 天 spike,實測 pgvector + HNSW 在 5M chunks 下 p99 比 Qdrant 慢 200ms,但 ops 成本與 query join 能力勝出。畫一張 decision matrix(維護成本、latency、cost、團隊熟悉度)給 staff eng 與 EM,主張換 pgvector。
> **R**:換掉後 p99 1.6s(達標)、月維運成本省 40%。一個月後新功能要 join user metadata,證明選擇正確。

### Q3. Tell me about a time you pushed back on a product / PM request.

> **S**:PM 要求 chatbot 在所有頁面 widget 化、每頁自動 inject context。
> **T**:技術上可做,但 cost 會 5x、latency 雙倍,user 也不見得想要。
> **A**:做 7 天 A/B test(injected vs on-demand),量化 engagement 與 cost。同時做 5 個 user interview。
> **R**:engagement 只 +3%,cost +480%,user 反映「打擾」。PM 接受改成 on-demand。學到 push back 必須帶替代方案 + 數據,不是說 No。

### Q4. Tell me about a time you had to balance speed vs quality.

> **S**:demo day 前 5 天,我們的 agent 在 tool calling 時偶發 schema 不合(<1% trigger 但 demo 必中)。
> **T**:選 (a) 治標 hardcode 5 個 tool 強制 schema,3 hr 完成; (b) 重構 tool dispatcher,3 天完成。
> **A**:選 (a) 上 demo,但在 dispatcher 加 TODO + GH issue 排到下 sprint 第一順位。Demo 完隔週做 (b)。
> **R**:demo 順利,長期債也清掉。學到 short-term hack 不是錯,**沒記下來** 才是錯。

### Q5. Tell me about a time you made a mistake.

> **S**:第一次部署 vLLM 自架推論服務時,沒設 `max_num_seqs` 限制。
> **T**:某天遇到 traffic 突發,GPU OOM crash,影響 30 分鐘。
> **A**:rollback 後做 post-mortem:加 admission control、加 Prometheus alert(GPU mem > 85%)、寫 runbook 進 wiki。在團隊週會公開檢討,不甩鍋給 traffic。
> **R**:同類事故沒再發生。團隊把這個 post-mortem 模板化成 SRE 標準流程。學到事故報告坦白比甩鍋更建立信任。

---

## Part B:phantom-mesh 真實場景(Q6–Q10)

### Q6. Tell me about a time you debugged a really hard problem. **[phantom-mesh: SSE 串流解析]**

> **S**:phantom-mesh 接 Anthropic streaming API,在 Windows PowerShell 環境下偶發 stream 卡死,Mac / Linux 正常。
> **T**:debug 一個 OS + transport + parser 三層交織的問題。
> **A**:寫一個最小重現(50 行 Python + httpx),抓 wire-level packet,發現 PowerShell `Out-Host` 預設 buffering 把 SSE chunked encoding 邊界吃掉。改成 raw byte stream + manual chunk parser,並加跨平台 CI matrix(Windows / Mac / Ubuntu)防回歸。
> **R**:bug fix 進 main,加了 3 個跨平台 integration test。學到 streaming bug 絕大多數不在 LLM 端,而在 buffering / encoding / OS 差異。

### Q7. Tell me about a time you had to coordinate across teams. **[phantom-mesh: multi-agent 協調]**

> **S**:phantom-mesh 的 supervisor agent 要協調 3 個 specialist agent(code、search、reasoning),早期版本 supervisor 卡在「等所有 agent 回完才繼續」,長 task 動輒等 5 分鐘。
> **T**:設計 streaming + interruption 的協調機制。
> **A**:借 Erlang actor model 思路,把 agent 包成 async generator,supervisor 用 `asyncio.as_completed` 邊收邊決策。引入 cancel token 讓 supervisor 可提早終止無用 agent。和另一個負責 cost tracking 的同事協作:他 hook cancel 事件用來 refund cost budget。
> **R**:平均 task latency 從 4.8 分降到 1.7 分,cost 省 30%。學到 multi-agent 不是「平行 LLM call」,核心是 coordination semantics。

### Q8. Tell me about a time you had to learn a new tech fast. **[phantom-mesh: 跨平台 build]**

> **S**:phantom-mesh 要支援 macOS / Windows / Linux 三平台 + ARM/x86 雙架構,我之前只熟 Linux Docker。
> **T**:1 週內生 6 種 build,並進 CI。
> **A**:看 GitHub Actions matrix doc 1 hr,看 Tauri / PyInstaller doc 各 2 hr,先做 happy path 再補 edge case(Windows code signing、macOS notarization、Linux AppImage)。每個踩坑寫進 docs/build.md。
> **R**:6 種 artifact 自動 release。學到「快速學新技術」不是讀書,是 **先做出最小可行版本再回頭補理論**。

### Q9. Tell me about a project you're proud of. **[phantom-mesh: provider fallback layer]**

> **S**:LLM API 每家都會 downtime / rate limit。phantom-mesh 用戶遇到 OpenAI 503 就整個 agent 失敗,體驗極差。
> **T**:設計 multi-provider fallback,要 (a) 切換對使用者透明 (b) 不能傳壞 message history (c) cost 仍正確 tracking。
> **A**:設計 abstraction 三層:provider client(各家原生 SDK)→ normalizer(統一 message / tool / streaming event)→ router(strategy:fallback / round-robin / cost-optimized)。寫了 80+ integration test,模擬 5 種失敗模式(timeout、rate limit、auth fail、schema invalid、partial stream)。
> **R**:現在 phantom-mesh 在 4 家 provider 自動 fallback,demo 中故意拔網路測,user 看不出來。學到 abstraction 不是把 API 包一層,是 **想清楚哪些差異該透出去、哪些該吃進來**。

### Q10. Tell me about how you measure success in your work. **[phantom-mesh: cost tracking]**

> **S**:phantom-mesh 想做 cost-aware multi-agent。早期版本 cost 不透明,user 跑一個任務不知道花多少。
> **T**:把 cost 變成 first-class signal:可看、可警告、可中斷。
> **A**:設計 per-agent / per-task cost ledger,hook 每個 LLM call 的 usage,累積到 task tree。引入 cost budget(user 可設「這個 task 最多 $0.50」),超過自動 graceful stop。寫 dashboard 展示 cost breakdown。
> **R**:user 從「不敢開 agent 怕花錢」變「自信跑 agent」。我把成功定義從「能跑」改成「使用者願意每天跑」。學到 cost 在 LLM 應用是 1st class metric,不是 ops 細節。

---

## Part C:學習力 / 應對失敗 / 跨團隊(Q11–Q15)

### Q11. Tell me about a time you had to learn from a failure.

> **S**:我第一次帶 LLM 專案,只看 demo accuracy,沒做嚴謹 eval,上線後客訴湧入。
> **T**:重建 eval 體系。
> **A**:導入 RAGAS + 自建 100 題 golden set,每 PR 跑 eval,regression > 3% 不能 merge。建 dashboard。
> **R**:之後 3 個月,大客訴歸零。學到「沒 eval 就不是 production」。

### Q12. Tell me about a time you solved an ambiguous problem.

> **S**:CEO 說「我們也要做 AI」,沒 spec。
> **T**:把模糊需求變成 3 個月可執行 roadmap。
> **A**:做 10 個 user interview、列 25 個 use case、用 (impact × feasibility) 排優先級、選 3 個做 MVP,寫成 spec doc 給 leadership review。
> **R**:選中的 use case 上線後成為公司年度產品 highlight。學到 ambiguity 是機會,不是阻力。

### Q13. Tell me about a time you got tough feedback.

> **S**:第一份工作的 perf review,manager 說我「技術好但溝通讓 PM 抓不到重點」。
> **T**:改善表達。
> **A**:每次寫 design doc 都先寫 TL;DR、加 decision section、找 PM 同事互相 review 對方的 doc。每個月固定看一本溝通書(《Made to Stick》、《On Writing Well》)。
> **R**:半年後同 manager 主動推我帶跨團隊專案。學到 feedback 不舒服才有用,要當訊號不是當攻擊。

### Q14. Tell me about a time you mentored or were mentored.

> **S**:帶一個 new grad 做第一個 RAG 專案。
> **T**:讓他 3 個月 ship,不是我幫他寫。
> **A**:每週 1on1,用「我問問題、不給答案」模式;讓他先寫 design doc 再 code;code review 給 directional feedback 不直接改;遇到難 bug 一起 debug 但讓他打字。
> **R**:他第 11 週獨立 ship,perf review 拿 exceeds。學到 mentor 是 **忍住不寫 code 的能力**。

### Q15. Tell me about a time you had to work with a difficult person.

> **S**:一位資深同事每次 code review 都極度挑剔、語氣尖銳,新人不敢發 PR。
> **T**:不能直接 confront,但要保護團隊文化。
> **A**:1-on-1 喝咖啡,先承認他的 review 技術上常常對,再 frame 為「想討論怎麼讓新人 onboarding 更快」。具體建議:review 多用問句、把 critical 與 nit 分開標記。同時和 EM 同步,引入 review etiquette guideline。
> **R**:他開始用 "nit:" / "blocking:" 標記,新人 PR 速度回升。學到 difficult 通常不是壞人,是 **沒被告知 cost**。

---

## Part D:領導 / 影響力 / 倫理(Q16–Q20)

### Q16. Tell me about a time you took initiative without being asked.

> **S**:團隊每週花 4 小時手動跑 model eval。
> **T**:沒人 assign 我做自動化,但我覺得這浪費。
> **A**:週末花 6 小時寫 eval pipeline + GitHub Actions cron + Slack notification。週一給 EM 看 demo,問是否願意正式採用。
> **R**:採用後團隊每週省 4 hr,EM 在公司 all-hands 提到。學到 initiative 不是請求 permission,是 **做 prototype 換 permission**。

### Q17. Tell me about a time you made an unpopular decision.

> **S**:我推團隊全面用 type hints + mypy strict,初期 PR 速度下降 30%。
> **T**:頂住 2 個月,證明長期收益。
> **A**:寫 RFC 解釋 cost / benefit,設 2 個月 trial,trial 結束 retrospective 投票決定去留。trial 期間我親自幫人修 type error 降低門檻。
> **R**:retrospective 投票 8:2 留下。生產 bug 同期下降 40%。學到 unpopular 不代表錯,但 **要設止損點**。

### Q18. Tell me about an ethical concern you raised at work.

> **S**:某個 chatbot 用了 scraped 資料訓練 retriever,我擔心 PII 外洩風險。
> **T**:不能直接擋上線(老闆要 deadline),但要負責任地 escalate。
> **A**:寫 risk memo 列 5 種潛在違規場景 + 3 個 mitigation(PII scrubbing pipeline、legal review、redaction at retrieval),呈 EM 與 legal。建議延後 1 週上線。
> **R**:legal 同意 mitigation,1 週後安全上線,沒事故。學到 ethics 不是 say no,是 **say no AND propose path forward**。

### Q19. Tell me about a time you had to influence without authority.

> **S**:跨團隊有 5 個 service 都各自接 OpenAI,沒有統一 monitoring。
> **T**:推動共用 LLM gateway,但我不是 staff、沒權力 mandate。
> **A**:寫 1-pager pitch、做 demo、找 3 個友善的 team lead 試用、收 testimonial、把 cost saving 算出來($800/month → $400/month),拿到 architecture review。
> **R**:4 個 service 半年內遷移完成,我也順勢升 senior。學到 influence = 把「對的事」變成「容易做的事」。

### Q20. Tell me about a time you had to say "I don't know."

> **S**:面試客戶現場 demo,客戶問:「你們的 model 在 OOD(out-of-distribution) input 上行為?」
> **T**:我們其實沒系統做過 OOD eval。
> **A**:直接說:「我們目前沒有 systematic OOD benchmark,我可以說的是已有 X eval,但你問的點是 gap。我們可以下週帶 OOD eval result 回來。」會後 3 天內補了一個 quick OOD eval。
> **R**:客戶後來說當下 honesty 是讓他簽約的關鍵。學到 senior 的標誌不是什麼都知道,是 **知道自己不知道什麼**。

---

## Top 5 容易被問的 LLM 工程師問題(必背)

這 5 題是 2026 年面試實際出現頻率前 5 名,要能 30 秒切入 + 2 分鐘展開 + 5 分鐘 deep dive。

### TQ1. "Tell me about a recent LLM project you worked on."
- **30 秒切入**:Problem(1 句)→ 你的 role(1 句)→ 最 impressive 的數字(1 句)→ trade-off(1 句)。
- **避免**:時間軸流水帳。
- **金句範例**:"我主導 RAG over 5M docs,把 hallucination 從 38% 降到 9%,核心是 hybrid retrieval + RAGAS in CI,trade-off 是 query latency +250ms 但 faithfulness +0.24。"

### TQ2. "How do you stay updated on LLM / AI?"
- **不要說**:"我看推特" / "我看新聞"(太弱)。
- **要說**:具體訊源 + 處理方法 + 怎麼選擇深入。
- **金句範例**:"每週固定 Epoch AI、SemiAnalysis、Dwarkesh、TLDR AI;每月讀 1-2 篇 frontier paper 並寫筆記;每季看一場 AI 大會錄影。我用本 repo 全景圖維持廣度,每季選 2-3 主題深耕。"(對應 [`04_AI_工程師持續學習指南.md`](./04_AI_工程師持續學習指南.md))

### TQ3. "What's a controversial opinion you hold about AI?"
- **目的**:測你有沒有獨立思考,而不是 hype follower。
- **不要說**:重複名人觀點 / 政治敏感題。
- **建議方向**(挑一個你真的相信的):
  - "LLM benchmark 大多 overfit,實際 use case 應該自己造 eval。"
  - "agent autonomy 被高估,80% 場景 workflow 比 agent 更穩。"
  - "context window 變大不能取代 RAG,反而讓 retrieval quality 更重要。"
- **架構**:claim(1 句)→ 為什麼大家錯(1 句)→ 我的證據 / 經驗(2 句)→ 但 acknowledge 反方有道理(1 句)。

### TQ4. "How would you evaluate a LLM application?"
- 從 use case 倒推:user-perceived metric → proxy metric → automated eval → CI gate。
- 範例 stack:RAGAS(faithfulness / answer relevancy)+ DeepEval(custom rubric)+ LLM-as-judge(GPT-4 / Claude)+ golden set(100-500 題人工標)+ regression test in CI。
- 提到 trade-off:LLM-as-judge 有 bias、golden set 會 stale、自動化 metric 跟人類偏好相關但不完美。

### TQ5. "Why do you want to work here?"
- **致命錯誤**:"because you're a great company"(0 分)。
- **公式**:具體產品 / 技術細節(你做了研究)→ 對應你想學的 / 你能貢獻的 → 為什麼現在這個時機。
- **金句範例**:"看了你們近期的 X 部落格,提到自己 train SLM 做 routing,這正好是我做過的方向;我帶來 phantom-mesh 的 provider abstraction 經驗,可以加速你們 multi-model 策略。2026 是 SLM + agent 的爆發點,我想押這個方向。"

---

## 練習方法

1. **錄音法**:把 20 題每題錄一次 1-2 分鐘的口語回答,自己回放,你會發現:嗯嗯啊啊太多、句子太長、結尾沒收。
2. **互練法**:找朋友 / mentor 做 mock,他擔任 hostile 面試官,每題追問 3 層("為什麼 X 不選 Y? cost 你怎麼算? 換 Z 會怎樣?")。
3. **STAR 表**:做一張 20 行 × 4 欄(S/T/A/R)的表,每題寫滿,口語化時就有 anchor。

---

## 下一步

- 持續學習 → [`04_AI_工程師持續學習指南.md`](./04_AI_工程師持續學習指南.md)
- 履歷再強化 → [`02_履歷與作品集打造.md`](./02_履歷與作品集打造.md)
- 技術題庫 → [`../1.LLM面試題庫/`](../1.LLM面試題庫/)
- 系統設計 → [`../2.系統設計案例/`](../2.系統設計案例/)
