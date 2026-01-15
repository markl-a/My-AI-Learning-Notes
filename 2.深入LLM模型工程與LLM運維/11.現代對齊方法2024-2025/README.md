# 現代 LLM 對齊方法 2024-2025

> **最後更新**: 2026-01
> **涵蓋範圍**: RLHF、DPO、IPO、SimPO、KTO、ORPO 等
> **難度**: 進階

---

## 目錄

1. [對齊概述](#1-對齊概述)
2. [RLHF 回顧與局限](#2-rlhf-回顧與局限)
3. [DPO: Direct Preference Optimization](#3-dpo-direct-preference-optimization)
4. [IPO: Identity Preference Optimization](#4-ipo-identity-preference-optimization)
5. [SimPO: Simple Preference Optimization](#5-simpo-simple-preference-optimization)
6. [KTO: Kahneman-Tversky Optimization](#6-kto-kahneman-tversky-optimization)
7. [ORPO: Odds Ratio Preference Optimization](#7-orpo-odds-ratio-preference-optimization)
8. [方法對比與選擇指南](#8-方法對比與選擇指南)

---

## 1. 對齊概述

### 1.1 什麼是 LLM 對齊？

LLM 對齊（Alignment）是指讓語言模型的行為符合人類偏好和價值觀的過程。

### 1.2 對齊方法演進

| 年份 | 方法 | 特點 |
|------|------|------|
| 2022 | RLHF | 需要獎勵模型 + PPO |
| 2023 | DPO | 直接偏好優化，無需獎勵模型 |
| 2024 | SimPO | 無需參考模型，長度歸一化 |
| 2024 | KTO | 支持二元標籤數據 |
| 2024 | ORPO | SFT + 對齊一步完成 |

---

## 2. RLHF 回顧與局限

### 2.1 RLHF 流程

1. **SFT 階段**: 監督微調
2. **RM 階段**: 訓練獎勵模型
3. **PPO 階段**: 強化學習優化

### 2.2 局限性

- 需要 4 個模型同時在內存
- PPO 訓練不穩定
- 可能產生獎勵駭客
- 資源消耗大

---

## 3. DPO: Direct Preference Optimization

### 3.1 核心思想

直接從偏好數據優化策略，無需訓練獎勵模型。

### 3.2 損失函數

```python
# DPO 損失
def dpo_loss(chosen_log_ratios, rejected_log_ratios, beta=0.1):
    diff = chosen_log_ratios - rejected_log_ratios
    losses = -F.logsigmoid(beta * diff)
    return losses.mean()
```

### 3.3 優缺點

| 優點 | 缺點 |
|------|------|
| 無需獎勵模型 | 需要高品質偏好數據 |
| 訓練穩定 | 對 β 參數敏感 |
| 內存效率高 | 可能有長度偏差 |

---

## 4. IPO: Identity Preference Optimization

解決 DPO 的過度擬合問題，使用恆等函數替代 sigmoid，添加正則化。

```python
def ipo_loss(chosen_log_ratios, rejected_log_ratios, beta=0.1):
    diff = chosen_log_ratios - rejected_log_ratios
    target = 1 / (2 * beta)
    return ((diff - target) ** 2).mean()
```

---

## 5. SimPO: Simple Preference Optimization

### 5.1 創新點

1. **無需參考模型**
2. **長度歸一化** - 解決長度偏差

### 5.2 損失函數

```python
def simpo_loss(chosen_log_probs, rejected_log_probs, 
               chosen_lengths, rejected_lengths,
               beta=2.0, gamma=0.5):
    chosen_rewards = beta * chosen_log_probs / chosen_lengths
    rejected_rewards = beta * rejected_log_probs / rejected_lengths
    return -F.logsigmoid(chosen_rewards - rejected_rewards - gamma).mean()
```

---

## 6. KTO: Kahneman-Tversky Optimization

### 6.1 特點

- 基於前景理論（損失厭惡）
- **只需二元標籤數據**，不需要成對偏好

### 6.2 數據格式

```json
{
    "prompt": "問題",
    "response": "回答",
    "label": true  // 好或壞
}
```

---

## 7. ORPO: Odds Ratio Preference Optimization

### 7.1 特點

結合 SFT 和偏好對齊為單一訓練階段。

### 7.2 損失函數

```
L_ORPO = L_SFT + λ · L_OR
```

---

## 8. 方法對比與選擇指南

### 8.1 對比表

| 方法 | 需要獎勵模型 | 需要參考模型 | 數據需求 |
|------|------------|------------|---------|
| RLHF | ✅ | ✅ | 偏好對 |
| DPO | ❌ | ✅ | 偏好對 |
| SimPO | ❌ | ❌ | 偏好對 |
| KTO | ❌ | ✅ | 二元標籤 |
| ORPO | ❌ | ❌ | 偏好對 |

### 8.2 選擇建議

- **資源有限**: SimPO
- **高品質對齊**: DPO
- **只有二元標籤**: KTO
- **從頭訓練**: ORPO

---

*本文檔持續更新中*
