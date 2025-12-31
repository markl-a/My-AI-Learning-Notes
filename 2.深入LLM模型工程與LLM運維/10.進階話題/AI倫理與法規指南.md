# AI 倫理與法規完整指南

> 最後更新：2025-01

## 📋 概述

隨著 AI 技術的快速發展，AI 倫理和法規遵循變得越來越重要。本指南涵蓋主要法規框架、倫理原則和實務合規策略。

## 🌍 全球 AI 法規概覽

### 歐盟 AI 法案 (EU AI Act)

```
風險分級管理：

┌─────────────────────────────────────────┐
│  不可接受的風險 (Prohibited)             │
│  • 社會信用評分系統                      │
│  • 公共場所即時生物識別                  │
│  • 操縱性 AI 系統                        │
├─────────────────────────────────────────┤
│  高風險 (High-Risk)                      │
│  • 醫療診斷系統                          │
│  • 信用評估系統                          │
│  • 招聘與人力資源系統                    │
│  • 教育評估系統                          │
│  要求：透明度、人工監督、風險評估        │
├─────────────────────────────────────────┤
│  有限風險 (Limited Risk)                 │
│  • 聊天機器人                            │
│  • 情感識別系統                          │
│  要求：透明度揭露                        │
├─────────────────────────────────────────┤
│  最小風險 (Minimal Risk)                 │
│  • 垃圾郵件過濾                          │
│  • 遊戲 AI                               │
│  要求：無特殊要求                        │
└─────────────────────────────────────────┘
```

### 中國 AI 法規

```
主要法規：
1. 《生成式人工智能服務管理暫行辦法》
   - 適用於向公眾提供生成式AI服務
   - 要求內容安全審核
   - 演算法備案制度

2. 《互聯網信息服務深度合成管理規定》
   - 深度偽造技術規範
   - 標記要求

3. 《互聯網信息服務算法推薦管理規定》
   - 演算法透明度
   - 用戶權益保護
```

### 美國 AI 政策

```
主要框架：
1. AI 行政命令 (Executive Order on AI)
   - 安全與安保要求
   - 報告義務

2. NIST AI 風險管理框架
   - 自願性指導
   - 風險評估方法

3. 各州法律
   - 加州 CCPA 延伸
   - 紐約 AI 招聘法
```

## 🎯 核心倫理原則

### 1. 公平性 (Fairness)

```python
from sklearn.metrics import confusion_matrix
import numpy as np

class FairnessAnalyzer:
    """公平性分析器"""

    def __init__(self, predictions, labels, sensitive_attribute):
        self.predictions = predictions
        self.labels = labels
        self.sensitive = sensitive_attribute

    def demographic_parity(self) -> dict:
        """人口統計平等：各群體的正預測率應相近"""
        groups = np.unique(self.sensitive)
        rates = {}

        for group in groups:
            mask = self.sensitive == group
            positive_rate = np.mean(self.predictions[mask])
            rates[group] = positive_rate

        # 計算差異
        max_diff = max(rates.values()) - min(rates.values())

        return {
            "rates": rates,
            "max_difference": max_diff,
            "is_fair": max_diff < 0.1  # 10% 閾值
        }

    def equalized_odds(self) -> dict:
        """均等機會：各群體的 TPR 和 FPR 應相近"""
        groups = np.unique(self.sensitive)
        metrics = {}

        for group in groups:
            mask = self.sensitive == group
            tn, fp, fn, tp = confusion_matrix(
                self.labels[mask],
                self.predictions[mask]
            ).ravel()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            metrics[group] = {"TPR": tpr, "FPR": fpr}

        return metrics

    def individual_fairness(self, similarity_matrix) -> float:
        """個體公平：相似個體應得到相似對待"""
        n = len(self.predictions)
        violations = 0

        for i in range(n):
            for j in range(i + 1, n):
                # 如果相似但預測不同
                if similarity_matrix[i, j] > 0.9:  # 相似閾值
                    if self.predictions[i] != self.predictions[j]:
                        violations += 1

        return 1 - (violations / (n * (n - 1) / 2))

    def generate_report(self) -> str:
        """生成公平性報告"""
        dp = self.demographic_parity()
        eo = self.equalized_odds()

        report = ["=== AI 公平性分析報告 ===\n"]

        report.append("\n📊 人口統計平等分析：")
        for group, rate in dp["rates"].items():
            report.append(f"  群體 {group}: 正預測率 = {rate:.2%}")
        report.append(f"  最大差異: {dp['max_difference']:.2%}")
        report.append(f"  是否公平: {'✅ 是' if dp['is_fair'] else '❌ 否'}")

        report.append("\n📊 均等機會分析：")
        for group, metrics in eo.items():
            report.append(f"  群體 {group}: TPR={metrics['TPR']:.2%}, FPR={metrics['FPR']:.2%}")

        return "\n".join(report)


# 使用範例
analyzer = FairnessAnalyzer(
    predictions=model_predictions,
    labels=true_labels,
    sensitive_attribute=gender_data
)

print(analyzer.generate_report())
```

### 2. 透明度與可解釋性 (Transparency & Explainability)

```python
import shap
from lime import lime_tabular

class ExplainabilityTools:
    """可解釋性工具集"""

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def shap_explanation(self, X_test):
        """SHAP 解釋"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        return {
            "shap_values": shap_values,
            "expected_value": explainer.expected_value,
            "feature_importance": np.abs(shap_values).mean(axis=0)
        }

    def lime_explanation(self, instance, feature_names):
        """LIME 局部解釋"""
        explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=feature_names,
            mode="classification"
        )

        exp = explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=10
        )

        return {
            "local_explanation": exp.as_list(),
            "prediction": exp.predict_proba
        }

    def generate_explanation_text(self, instance, feature_names) -> str:
        """生成可讀的解釋文本"""
        lime_exp = self.lime_explanation(instance, feature_names)

        text = ["=== 模型決策解釋 ===\n"]
        text.append(f"預測結果：{lime_exp['prediction']}\n")
        text.append("\n影響因素（按重要性排序）：")

        for feature, weight in lime_exp["local_explanation"][:5]:
            direction = "正向" if weight > 0 else "負向"
            text.append(f"  • {feature}: {direction}影響 ({abs(weight):.3f})")

        return "\n".join(text)


# 使用範例
explainer = ExplainabilityTools(model, X_train)
explanation = explainer.generate_explanation_text(
    test_instance,
    feature_names=["年齡", "收入", "信用歷史", "負債比率"]
)
print(explanation)
```

### 3. 隱私保護 (Privacy)

```python
from typing import List, Dict
import hashlib
import numpy as np

class PrivacyProtection:
    """隱私保護工具"""

    @staticmethod
    def k_anonymity(data, quasi_identifiers: List[str], k: int) -> bool:
        """檢查是否滿足 k-匿名性"""
        groups = data.groupby(quasi_identifiers).size()
        return all(groups >= k)

    @staticmethod
    def differential_privacy_noise(
        data: np.ndarray,
        epsilon: float,
        sensitivity: float
    ) -> np.ndarray:
        """添加差分隱私噪音（拉普拉斯機制）"""
        noise = np.random.laplace(
            loc=0,
            scale=sensitivity / epsilon,
            size=data.shape
        )
        return data + noise

    @staticmethod
    def pseudonymize(data: str, salt: str) -> str:
        """假名化處理"""
        return hashlib.sha256((data + salt).encode()).hexdigest()[:16]

    @staticmethod
    def data_minimization_check(
        required_fields: List[str],
        collected_fields: List[str]
    ) -> Dict:
        """資料最小化檢查"""
        unnecessary = set(collected_fields) - set(required_fields)

        return {
            "required": required_fields,
            "collected": collected_fields,
            "unnecessary": list(unnecessary),
            "is_compliant": len(unnecessary) == 0
        }


class PIIDetector:
    """個人可識別資訊偵測器"""

    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone_tw": r"09\d{8}",
        "id_tw": r"[A-Z][12]\d{8}",
        "credit_card": r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}",
    }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """偵測文本中的 PII"""
        import re
        findings = {}

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches

        return findings

    def redact(self, text: str) -> str:
        """遮蔽 PII"""
        import re
        redacted = text

        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", redacted)

        return redacted
```

### 4. 問責機制 (Accountability)

```python
from datetime import datetime
from typing import Optional
import json

class AIAuditLog:
    """AI 審計日誌系統"""

    def __init__(self, storage_backend):
        self.storage = storage_backend

    def log_model_decision(
        self,
        model_id: str,
        input_data: dict,
        output: dict,
        explanation: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """記錄模型決策"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": model_id,
            "model_version": self._get_model_version(model_id),
            "input_hash": self._hash_data(input_data),
            "output": output,
            "explanation": explanation,
            "user_id": user_id,
            "traceable_id": self._generate_trace_id()
        }

        self.storage.append(log_entry)
        return log_entry["traceable_id"]

    def log_human_override(
        self,
        trace_id: str,
        override_by: str,
        reason: str,
        new_decision: dict
    ):
        """記錄人工覆寫"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "human_override",
            "original_trace_id": trace_id,
            "override_by": override_by,
            "reason": reason,
            "new_decision": new_decision
        }

        self.storage.append(log_entry)

    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """生成審計報告"""
        logs = self.storage.query(
            filter={
                "timestamp": {"$gte": start_date, "$lte": end_date}
            }
        )

        return {
            "period": f"{start_date} to {end_date}",
            "total_decisions": len(logs),
            "human_overrides": sum(1 for l in logs if l.get("type") == "human_override"),
            "model_breakdown": self._group_by_model(logs),
            "generated_at": datetime.utcnow().isoformat()
        }
```

## 🛡️ 合規實施框架

### 風險評估模板

```python
class AIRiskAssessment:
    """AI 風險評估框架"""

    RISK_CATEGORIES = {
        "bias": "偏見與歧視風險",
        "privacy": "隱私風險",
        "security": "安全風險",
        "transparency": "透明度風險",
        "reliability": "可靠性風險",
        "accountability": "問責風險"
    }

    RISK_LEVELS = {
        "low": {"score": 1, "color": "🟢"},
        "medium": {"score": 2, "color": "🟡"},
        "high": {"score": 3, "color": "🟠"},
        "critical": {"score": 4, "color": "🔴"}
    }

    def __init__(self, system_name: str, system_description: str):
        self.system_name = system_name
        self.description = system_description
        self.assessments = {}

    def assess_risk(
        self,
        category: str,
        level: str,
        description: str,
        mitigation: str
    ):
        """評估特定風險"""
        if category not in self.RISK_CATEGORIES:
            raise ValueError(f"未知風險類別: {category}")

        self.assessments[category] = {
            "level": level,
            "score": self.RISK_LEVELS[level]["score"],
            "description": description,
            "mitigation": mitigation
        }

    def calculate_overall_risk(self) -> str:
        """計算總體風險等級"""
        if not self.assessments:
            return "unknown"

        avg_score = sum(a["score"] for a in self.assessments.values()) / len(self.assessments)

        if avg_score <= 1.5:
            return "low"
        elif avg_score <= 2.5:
            return "medium"
        elif avg_score <= 3.5:
            return "high"
        else:
            return "critical"

    def generate_report(self) -> str:
        """生成風險評估報告"""
        report = [
            "=" * 50,
            f"AI 系統風險評估報告",
            "=" * 50,
            f"\n系統名稱: {self.system_name}",
            f"系統描述: {self.description}",
            f"評估日期: {datetime.now().strftime('%Y-%m-%d')}",
            f"\n總體風險等級: {self.RISK_LEVELS[self.calculate_overall_risk()]['color']} {self.calculate_overall_risk().upper()}",
            "\n" + "-" * 50,
            "詳細評估：",
            "-" * 50
        ]

        for category, assessment in self.assessments.items():
            level_info = self.RISK_LEVELS[assessment["level"]]
            report.extend([
                f"\n{level_info['color']} {self.RISK_CATEGORIES[category]}",
                f"   風險等級: {assessment['level'].upper()}",
                f"   風險描述: {assessment['description']}",
                f"   緩解措施: {assessment['mitigation']}"
            ])

        return "\n".join(report)


# 使用範例
assessment = AIRiskAssessment(
    system_name="信用評估 AI 系統",
    system_description="用於評估貸款申請人信用風險的機器學習系統"
)

assessment.assess_risk(
    category="bias",
    level="medium",
    description="模型可能對某些人口群體存在偏見",
    mitigation="實施定期公平性審計，使用去偏見技術"
)

assessment.assess_risk(
    category="privacy",
    level="high",
    description="處理大量敏感個人財務資料",
    mitigation="實施資料加密、存取控制和定期隱私影響評估"
)

assessment.assess_risk(
    category="transparency",
    level="medium",
    description="決策過程不夠透明",
    mitigation="整合 SHAP/LIME 解釋工具，提供決策理由"
)

print(assessment.generate_report())
```

## 📋 合規檢查清單

### 部署前檢查

```markdown
## AI 系統部署前合規檢查清單

### 1. 資料與隱私 ☐
- [ ] 已進行資料保護影響評估 (DPIA)
- [ ] 已取得必要的資料使用同意
- [ ] 已實施資料最小化原則
- [ ] 敏感資料已加密或匿名化
- [ ] 已建立資料保留和刪除政策

### 2. 公平性與偏見 ☐
- [ ] 已進行偏見審計
- [ ] 已測試不同人口群體的表現
- [ ] 已記錄模型限制和潛在偏見
- [ ] 已建立偏見監控機制

### 3. 透明度 ☐
- [ ] 已準備模型文檔（Model Card）
- [ ] 用戶知道正在與 AI 互動
- [ ] 已整合可解釋性工具
- [ ] 決策邏輯可被審計

### 4. 安全性 ☐
- [ ] 已進行對抗攻擊測試
- [ ] 已建立異常偵測機制
- [ ] 已實施存取控制
- [ ] 已建立事件響應計劃

### 5. 人工監督 ☐
- [ ] 高風險決策有人工審核
- [ ] 已建立覆寫機制
- [ ] 操作人員已受訓練
- [ ] 已定義升級流程

### 6. 問責機制 ☐
- [ ] 已建立審計日誌
- [ ] 已指定責任人
- [ ] 已建立投訴處理機制
- [ ] 已準備事故報告流程
```

## 📚 資源參考

### 法規文件
- [EU AI Act 官方文本](https://eur-lex.europa.eu)
- [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
- [中國生成式AI管理辦法](http://www.cac.gov.cn)

### 倫理指南
- IEEE Ethically Aligned Design
- Partnership on AI Guidelines
- OECD AI Principles

### 技術工具
- Fairlearn：公平性工具包
- AI Fairness 360：IBM 公平性工具
- SHAP/LIME：可解釋性工具
- Responsible AI Toolbox：微軟工具包

---

*本指南持續更新中，法規資訊請以官方來源為準。*
