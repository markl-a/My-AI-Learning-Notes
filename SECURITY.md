# 安全政策 / Security Policy

## 支援的版本 / Supported Versions

| 版本 / Version | 支援狀態 / Supported |
| -------------- | -------------------- |
| 1.x.x          | :white_check_mark:   |
| < 1.0          | :x:                  |

## 報告漏洞 / Reporting a Vulnerability

我們非常重視專案的安全性。如果您發現任何安全漏洞，請按照以下步驟報告：

We take the security of this project seriously. If you discover a security vulnerability, please follow these steps:

### 報告方式 / How to Report

1. **請勿**在公開的 Issue 中報告安全漏洞
2. 請通過以下方式私下報告：
   - 建立一個標題為 `[SECURITY]` 的私密 Issue
   - 或發送郵件至專案維護者

1. **Do NOT** report security vulnerabilities in public issues
2. Please report privately through:
   - Creating a private issue with `[SECURITY]` in the title
   - Or emailing the project maintainers

### 報告內容應包含 / What to Include

- 漏洞的詳細描述 / Detailed description of the vulnerability
- 重現步驟 / Steps to reproduce
- 潛在影響 / Potential impact
- 可能的修復建議（如有）/ Suggested fix (if any)

### 回應時間 / Response Timeline

- **確認收到**：48 小時內 / Within 48 hours
- **初步評估**：7 天內 / Within 7 days
- **修復計劃**：根據嚴重程度，14-30 天內 / 14-30 days depending on severity

## 安全最佳實踐 / Security Best Practices

使用本專案的程式碼時，請注意以下安全事項：

When using code from this project, please note the following:

### API 金鑰管理 / API Key Management

```bash
# 正確做法：使用環境變數
export OPENAI_API_KEY="your-key-here"

# 錯誤做法：硬編碼在程式碼中
# api_key = "sk-xxxxx"  # 永遠不要這樣做！
```

### 環境變數 / Environment Variables

- 使用 `.env` 文件管理敏感資訊
- 確保 `.env` 已加入 `.gitignore`
- 參考 `.env.example` 設置您的環境

### 依賴安全 / Dependency Security

```bash
# 定期檢查依賴安全
pip-audit
safety check
```

## 已知安全考量 / Known Security Considerations

### 本專案中的程式碼示例

- 示例程式碼主要用於**教育目的**
- 在生產環境中使用前，請進行適當的安全審查
- 特別注意 API 呼叫、文件操作和用戶輸入處理

### LLM 應用安全

使用 LLM 相關程式碼時，請注意：

- **Prompt Injection**：驗證和清理用戶輸入
- **敏感資料**：避免在 prompt 中包含敏感資訊
- **輸出驗證**：不要盲目信任 LLM 的輸出
- **Rate Limiting**：實施適當的請求限制

## 安全更新 / Security Updates

安全更新將通過以下方式發布：

- GitHub Releases
- README.md 中的更新日誌
- 重大安全問題會有專門的通知

## 致謝 / Acknowledgments

我們感謝所有負責任地報告安全問題的人。報告者將在修復發布後被列入致謝名單（如果他們願意）。

We thank everyone who responsibly reports security issues. Reporters will be credited in the acknowledgments after the fix is released (if they wish).

---

**安全是我們共同的責任。感謝您幫助保護這個專案！**

**Security is a shared responsibility. Thank you for helping keep this project safe!**
