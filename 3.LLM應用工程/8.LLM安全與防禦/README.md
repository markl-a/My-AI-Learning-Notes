## LLM 安全與防禦

### 8.1 Prompt Injection、越獄與資料洩漏風險
### 8.2 OWASP LLM Top 10 資安議題
### 8.3 紅隊測試 (Red Teaming) 與防禦策略

## 📚 學習資源

### 理論文檔
- [2025_LLM安全挑戰與防禦.md](./2025_LLM安全挑戰與防禦.md) - 完整的 LLM 安全理論知識

### 實作示例

本資料夾包含完整的 LLM 安全防禦實作示例，涵蓋以下主題：

#### 1. [Prompt Injection 檢測器](./實作示例/1.prompt_injection_detector/)
- 多層次檢測系統（基於規則、FlipAttack、AI 輔助）
- 威脅等級分類（CRITICAL, HIGH, MEDIUM, LOW, SAFE）
- 常見攻擊模式識別
- **狀態**: ✅ 已完成並測試

#### 2. [安全的 LLM Pipeline](./實作示例/2.secure_llm_pipeline/)
- 多層防禦策略（Defense in Depth）
- 7 層安全檢查流程
- 速率限制、輸入清理、輸出驗證
- 審計日誌系統
- **狀態**: ✅ 已完成並測試

#### 3. [安全的 RAG 系統](./實作示例/3.secure_rag/)
- 文檔投毒防禦
- 向量數據庫安全包裝
- 間接 Prompt Injection 防禦
- 安全的上下文構建
- **狀態**: ✅ 已完成並測試

#### 4. [安全的 Agent 系統](./實作示例/4.secure_agent/)
- Meta's Rule of Two 實作
- 工具訪問控制
- 人機協作（Human-in-the-Loop）
- **狀態**: 📝 開發中

#### 5. [安全監控與審計](./實作示例/5.monitoring_and_audit/)
- 實時安全監控
- 異常行為檢測
- 告警系統
- **狀態**: 📝 開發中

#### 6. [紅隊測試工具](./實作示例/6.red_teaming/)
- 自動化攻擊測試
- 已知攻擊模式庫
- 防禦效果評估
- **狀態**: 📝 開發中

#### 7. [完整的安全聊天機器人](./實作示例/7.complete_chatbot/)
- 整合所有安全組件
- 生產就緒的示例
- **狀態**: 📝 開發中

## 🚀 快速開始

### 安裝依賴

```bash
cd 實作示例
pip install -r requirements.txt
```

### 運行示例

```bash
# Prompt Injection 檢測器
cd 1.prompt_injection_detector
python prompt_injection_detector.py

# 安全的 LLM Pipeline
cd ../2.secure_llm_pipeline
python secure_pipeline.py

# 安全的 RAG 系統
cd ../3.secure_rag
python secure_rag.py
```

## 📖 學習路徑

### 初學者路徑
1. 閱讀 [2025_LLM安全挑戰與防禦.md](./2025_LLM安全挑戰與防禦.md)
2. 運行 Prompt Injection 檢測器示例
3. 研究安全 Pipeline 的實作

### 進階路徑
1. 深入研究 RAG 安全
2. 實作 Agent 安全機制
3. 建立完整的安全監控系統

### 實戰路徑
1. 使用紅隊測試工具評估自己的系統
2. 整合所有安全組件到生產環境
3. 持續監控和改進

## 🛡️ 安全檢查清單

基於 OWASP LLM Top 10 的安全檢查清單：

- [x] LLM01: Prompt Injection 防禦
  - [x] 輸入驗證
  - [x] 模式檢測
  - [x] AI 輔助檢測

- [x] LLM02: 敏感信息洩露防禦
  - [x] 輸出驗證
  - [x] 系統提示保護

- [ ] LLM03: 供應鏈安全
  - [ ] 依賴審查
  - [ ] 模型驗證

- [ ] LLM04: 數據與模型投毒防禦
  - [x] 文檔驗證（RAG）
  - [ ] 訓練數據驗證

- [x] LLM05: 輸出處理
  - [x] 輸出驗證
  - [x] 內容過濾

- [ ] LLM06: 過度代理權限
  - [ ] 最小權限原則
  - [ ] 工具訪問控制

- [x] LLM07: 系統提示洩露
  - [x] 輸出檢測
  - [x] 安全的提示模板

- [x] LLM08: 向量與嵌入弱點
  - [x] 向量數據庫安全
  - [x] 嵌入驗證

- [x] LLM09: 錯誤信息
  - [x] 輸出驗證
  - [ ] 事實檢查

- [x] LLM10: 無界限消耗
  - [x] 速率限制
  - [x] 長度限制

## 🔧 自定義和擴展

所有示例都設計為可擴展的，你可以：

1. **集成真實的 LLM** - 替換 MockLLM 為 OpenAI、Anthropic 等
2. **使用真實的向量數據庫** - 集成 ChromaDB、Pinecone 等
3. **添加自定義驗證規則** - 根據業務需求添加檢測模式
4. **集成到現有系統** - 將安全組件集成到你的應用中

## 📊 性能考慮

- **輸入驗證** - 約 1-5ms
- **Prompt Injection 檢測** - 約 5-20ms
- **輸出驗證** - 約 1-5ms
- **總體開銷** - 約 10-30ms（相比 LLM 調用時間可忽略不計）

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📚 參考資源

### 官方資源
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [OpenAI Safety Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Anthropic's AI Safety](https://www.anthropic.com/index/core-views-on-ai-safety)

### 研究論文
- [FlipAttack Research](https://www.keysight.com/blogs/en/tech/nwvs/2025/05/20/prompt-injection-techniques-jailbreaking-large-language-models-via-flipattack)
- [Sugar-Coated Poison Attack](https://www.keysight.com/blogs/en/tech/nwvs/2025/08/07/sugar-coated-poison-prompt-injection-attack)
- [Meta's Agents Rule of Two](https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/)

### 工具和框架
- [LLM Guard](https://github.com/protectai/llm-guard)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [Rebuff](https://github.com/protectai/rebuff)

## ⚠️ 重要聲明

本資料夾中的所有攻擊示例僅供教育目的。請勿在未經授權的系統上使用。

- ✅ **允許**: 在自己的測試環境中學習和研究
- ✅ **允許**: 在授權的紅隊測試中使用
- ❌ **禁止**: 在生產環境或他人系統上惡意使用
- ❌ **禁止**: 用於任何非法活動

## 📝 更新日誌

### 2025-11-18
- ✅ 新增 Prompt Injection 檢測器
- ✅ 新增安全的 LLM Pipeline
- ✅ 新增安全的 RAG 系統
- 📝 Agent 安全系統開發中
- 📝 監控和審計系統開發中