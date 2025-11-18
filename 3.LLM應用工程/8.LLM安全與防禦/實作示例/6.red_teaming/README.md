# 紅隊測試工具

## 概述

這是一個自動化的紅隊測試工具，用於評估 LLM 系統的安全防禦能力。

## 功能特點

- **攻擊模式庫** - 包含多種已知的攻擊模式
- **自動化測試** - 自動運行所有攻擊並評估結果
- **詳細報告** - 生成包含統計和詳細結果的報告
- **可擴展** - 輕鬆添加新的攻擊模式

## 攻擊類型

1. **Prompt Injection** - 直接注入攻擊
2. **Jailbreak** - 越獄攻擊
3. **System Prompt Leak** - 系統提示洩露
4. **Encoding Bypass** - 編碼繞過
5. **Context Overflow** - 上下文溢出
6. **FlipAttack** - 反轉文本攻擊

## 使用方法

### 基本使用

```bash
# 測試無防禦系統（演示用）
python red_team_tester.py --mode vulnerable

# 測試安全 Pipeline
python red_team_tester.py --mode secure
```

### 測試自定義系統

```python
from red_team_tester import RedTeamTester

# 定義你的系統
def my_system(user_input: str):
    # 你的 LLM 系統邏輯
    return process(user_input)

# 創建測試器
tester = RedTeamTester(my_system)

# 運行測試
results = tester.run_tests()

# 生成報告
report = tester.generate_report()
print(report)
```

### 測試特定攻擊類型

```python
from red_team_tester import RedTeamTester, AttackType

# 只測試 Prompt Injection
results = tester.run_tests(
    attack_types=[AttackType.PROMPT_INJECTION]
)
```

## 報告示例

```
================================================================================
紅隊測試報告
================================================================================

總測試數: 12
被阻止的攻擊: 7 (58.3%)
成功防禦: 5
防禦失敗: 2
誤報: 0

================================================================================
詳細測試結果
================================================================================

### PROMPT_INJECTION ###

攻擊: 基礎 Prompt Injection
Payload: Ignore all previous instructions...
被阻止: 是
評估: ✓ 正確阻止
```

## 添加自定義攻擊

```python
from red_team_tester import AttackPattern, AttackType

# 添加新的攻擊模式
custom_attack = AttackPattern(
    name="我的自定義攻擊",
    type=AttackType.PROMPT_INJECTION,
    payload="你的攻擊 payload",
    description="攻擊描述",
    expected_behavior="應該被阻止"
)

# 將其添加到攻擊庫
attacks = AttackLibrary.get_all_attacks()
attacks.append(custom_attack)
```

## 最佳實踐

### 1. 定期測試

```bash
# 在 CI/CD 中運行
python red_team_tester.py --mode secure > report.txt
```

### 2. 追蹤改進

記錄每次測試的結果，追蹤防禦改進情況。

### 3. 更新攻擊庫

隨著新攻擊技術的出現，及時更新攻擊模式庫。

## 重要聲明

⚠️ **教育用途** - 此工具僅供教育和授權的安全測試使用。

- ✅ 在自己的系統上測試
- ✅ 在授權的紅隊演練中使用
- ❌ 不要攻擊他人的系統
- ❌ 不要用於惡意目的

## 參考資源

- [Red Teaming LLM Applications](https://www.anthropic.com/index/red-teaming-language-models)
- [OWASP LLM Testing Guide](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
