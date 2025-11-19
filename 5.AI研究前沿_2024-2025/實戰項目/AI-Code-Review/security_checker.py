"""
安全檢查器
檢測代碼中的安全漏洞和敏感信息
"""

import os
import re
import logging
from typing import Dict, List, Any
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class SecurityChecker:
    """安全檢查器類"""

    def __init__(self, model_name: str = "gpt-4"):
        """初始化安全檢查器"""
        self.model_name = model_name
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.total_issues_found = 0

        # 敏感信息模式
        self.secret_patterns = {
            "api_key": r'(?i)(api[_-]?key|apikey)[\s]*[=:]["\']\s*([A-Za-z0-9_\-]{20,})',
            "aws_key": r'(?i)(AWS|aws)[_-]?(ACCESS|access)[_-]?(KEY|key)[_-]?(ID|id)?[\s]*[=:]["\']\s*([A-Z0-9]{20})',
            "password": r'(?i)(password|passwd|pwd)[\s]*[=:]["\']\s*([^"\'\s]{8,})',
            "token": r'(?i)(token|auth|bearer)[\s]*[=:]["\']\s*([A-Za-z0-9_\-\.]{20,})',
            "private_key": r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
            "jwt": r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
            "connection_string": r'(?i)(mongodb|mysql|postgres|mssql)://[^\s"\']+',
        }

        # OWASP Top 10 檢查規則
        self.owasp_rules = {
            "sql_injection": [
                r'execute\s*\(\s*["\']SELECT.*\+.*["\']',
                r'cursor\.execute\s*\(\s*f["\']',
                r'query\s*=\s*["\']SELECT.*\%s',
            ],
            "xss": [
                r'innerHTML\s*=',
                r'dangerouslySetInnerHTML',
                r'document\.write\s*\(',
            ],
            "command_injection": [
                r'os\.system\s*\(',
                r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\(',
            ],
            "path_traversal": [
                r'open\s*\(.*\+.*\)',
                r'\.\./',
            ],
        }

    def get_total_issues(self) -> int:
        """獲取發現的安全問題總數"""
        return self.total_issues_found

    async def check_vulnerabilities(
        self,
        code: str,
        language: str,
        check_types: List[str] = ["all"]
    ) -> Dict[str, Any]:
        """
        檢查安全漏洞

        Args:
            code: 代碼內容
            language: 編程語言
            check_types: 檢查類型列表

        Returns:
            漏洞檢查結果
        """
        issues = []

        # 1. 基於規則的檢查
        rule_based_issues = self._rule_based_check(code, check_types)
        issues.extend(rule_based_issues)

        # 2. 基於 LLM 的深度檢查
        llm_issues = await self._llm_based_check(code, language, check_types)
        issues.extend(llm_issues)

        # 統計嚴重性分布
        severity_distribution = self._calculate_severity_distribution(issues)

        # 計算安全評分
        security_score = self._calculate_security_score(issues)

        # 生成建議
        recommendations = self._generate_recommendations(issues)

        # 合規性檢查
        compliant_with = self._check_compliance(issues)

        self.total_issues_found += len(issues)

        return {
            "score": security_score,
            "issues": issues,
            "severity_distribution": severity_distribution,
            "recommendations": recommendations,
            "compliant_with": compliant_with
        }

    def _rule_based_check(
        self,
        code: str,
        check_types: List[str]
    ) -> List[Dict[str, Any]]:
        """基於規則的安全檢查"""
        issues = []

        # 檢查每個 OWASP 規則
        for vulnerability_type, patterns in self.owasp_rules.items():
            if "all" in check_types or vulnerability_type in check_types:
                for pattern in patterns:
                    matches = re.finditer(pattern, code, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        # 找到匹配的行號
                        line_number = code[:match.start()].count('\n') + 1

                        issues.append({
                            "type": vulnerability_type,
                            "severity": self._get_severity(vulnerability_type),
                            "description": f"Potential {vulnerability_type.replace('_', ' ')} vulnerability",
                            "line_number": line_number,
                            "code_snippet": match.group(0),
                            "cwe_id": self._get_cwe_id(vulnerability_type),
                            "owasp_category": self._get_owasp_category(vulnerability_type),
                            "recommendation": self._get_recommendation(vulnerability_type)
                        })

        return issues

    async def _llm_based_check(
        self,
        code: str,
        language: str,
        check_types: List[str]
    ) -> List[Dict[str, Any]]:
        """基於 LLM 的深度安全檢查"""
        check_types_str = ", ".join(check_types) if "all" not in check_types else "all security vulnerabilities"

        prompt = f"""
對以下 {language} 代碼進行深度安全分析。

代碼:
```{language}
{code}
```

檢查項目: {check_types_str}

重點關注：
1. OWASP Top 10 漏洞
2. 注入攻擊（SQL、命令、代碼）
3. 認證和授權缺陷
4. 敏感數據洩露
5. XML 外部實體 (XXE)
6. 安全配置錯誤
7. 跨站腳本攻擊 (XSS)
8. 不安全的反序列化
9. 使用含有已知漏洞的組件
10. 日誌和監控不足

以 JSON 格式返回發現的問題：
{{
    "vulnerabilities": [
        {{
            "type": "sql_injection",
            "severity": "high",
            "description": "描述",
            "line_number": 10,
            "recommendation": "修復建議"
        }}
    ]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位網絡安全專家，專注於代碼安全審查。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content

            # 解析 JSON 結果
            import json
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("vulnerabilities", [])
            else:
                return []

        except Exception as e:
            logger.error(f"Error in LLM-based security check: {str(e)}")
            return []

    def _get_severity(self, vulnerability_type: str) -> str:
        """獲取漏洞嚴重性"""
        severity_map = {
            "sql_injection": "critical",
            "command_injection": "critical",
            "xss": "high",
            "path_traversal": "high",
            "secrets": "critical",
        }
        return severity_map.get(vulnerability_type, "medium")

    def _get_cwe_id(self, vulnerability_type: str) -> str:
        """獲取 CWE ID"""
        cwe_map = {
            "sql_injection": "CWE-89",
            "xss": "CWE-79",
            "command_injection": "CWE-78",
            "path_traversal": "CWE-22",
        }
        return cwe_map.get(vulnerability_type, "CWE-000")

    def _get_owasp_category(self, vulnerability_type: str) -> str:
        """獲取 OWASP 類別"""
        owasp_map = {
            "sql_injection": "A03:2021 – Injection",
            "xss": "A03:2021 – Injection",
            "command_injection": "A03:2021 – Injection",
            "authentication": "A07:2021 – Identification and Authentication Failures",
        }
        return owasp_map.get(vulnerability_type, "Unknown")

    def _get_recommendation(self, vulnerability_type: str) -> str:
        """獲取修復建議"""
        recommendations = {
            "sql_injection": "使用參數化查詢或 ORM，永遠不要拼接 SQL 字符串",
            "xss": "對所有用戶輸入進行轉義，使用內容安全策略 (CSP)",
            "command_injection": "避免使用 shell=True，使用白名單驗證輸入",
            "path_traversal": "驗證和規範化文件路徑，使用白名單",
        }
        return recommendations.get(vulnerability_type, "遵循 OWASP 安全編碼指南")

    def _calculate_severity_distribution(
        self,
        issues: List[Dict]
    ) -> Dict[str, int]:
        """計算嚴重性分布"""
        distribution = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        for issue in issues:
            severity = issue.get("severity", "medium")
            distribution[severity] = distribution.get(severity, 0) + 1

        return distribution

    def _calculate_security_score(self, issues: List[Dict]) -> int:
        """計算安全評分"""
        # 從 100 分開始，根據問題嚴重性扣分
        score = 100

        severity_weights = {
            "critical": 25,
            "high": 15,
            "medium": 8,
            "low": 3
        }

        for issue in issues:
            severity = issue.get("severity", "medium")
            score -= severity_weights.get(severity, 5)

        return max(0, score)

    def _generate_recommendations(
        self,
        issues: List[Dict]
    ) -> List[str]:
        """生成總體建議"""
        recommendations = []

        # 根據發現的問題生成建議
        if any(i.get("type") == "sql_injection" for i in issues):
            recommendations.append("實施參數化查詢和 ORM 使用規範")

        if any(i.get("type") == "xss" for i in issues):
            recommendations.append("實施輸出編碼和內容安全策略 (CSP)")

        if any(i.get("severity") == "critical" for i in issues):
            recommendations.append("立即修復所有嚴重級別的安全問題")

        if len(issues) > 5:
            recommendations.append("進行全面的安全代碼審查")
            recommendations.append("實施自動化安全測試")

        return recommendations

    def _check_compliance(self, issues: List[Dict]) -> List[str]:
        """檢查合規性"""
        compliant_with = []

        # OWASP 合規性
        owasp_issues = [i for i in issues if i.get("owasp_category")]
        if len(owasp_issues) == 0:
            compliant_with.append("OWASP Top 10 2021")

        # CWE 合規性
        if len(issues) < 3:
            compliant_with.append("CWE Top 25")

        return compliant_with if compliant_with else ["None"]

    async def scan_secrets(
        self,
        code: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        掃描敏感信息

        Args:
            code: 代碼內容
            language: 編程語言

        Returns:
            發現的敏感信息列表
        """
        secrets_found = []

        # 使用正則表達式掃描
        for secret_type, pattern in self.secret_patterns.items():
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_number = code[:match.start()].count('\n') + 1

                # 提取匹配的值（隱藏部分內容）
                matched_value = match.group(0)
                if len(matched_value) > 20:
                    displayed_value = matched_value[:10] + "..." + matched_value[-5:]
                else:
                    displayed_value = matched_value[:5] + "..."

                secrets_found.append({
                    "type": secret_type,
                    "line_number": line_number,
                    "severity": "critical",
                    "value": displayed_value,
                    "description": f"Potential {secret_type.replace('_', ' ')} detected",
                    "recommendation": self._get_secret_recommendation(secret_type)
                })

        return secrets_found

    def _get_secret_recommendation(self, secret_type: str) -> str:
        """獲取敏感信息處理建議"""
        recommendations = {
            "api_key": "使用環境變量或密鑰管理服務（如 AWS Secrets Manager）",
            "aws_key": "使用 IAM 角色而不是硬編碼憑證",
            "password": "永遠不要硬編碼密碼，使用密碼管理器",
            "token": "使用安全的令牌存儲機制",
            "private_key": "私鑰應該安全存儲，永遠不要提交到版本控制",
            "jwt": "JWT 令牌不應硬編碼在代碼中",
            "connection_string": "使用配置文件或環境變量存儲連接字符串"
        }
        return recommendations.get(secret_type, "使用安全的憑證管理方案")
