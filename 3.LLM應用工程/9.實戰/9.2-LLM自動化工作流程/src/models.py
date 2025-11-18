"""數據模型"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Severity(str, Enum):
    """問題嚴重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IssueType(str, Enum):
    """問題類型"""
    SYNTAX = "syntax"
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    ERROR_HANDLING = "error_handling"
    NAMING = "naming"
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    BEST_PRACTICE = "best_practice"


class CodeIssue(BaseModel):
    """代碼問題"""
    type: IssueType
    severity: Severity
    message: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    fixed_code: Optional[str] = None
    confidence: float = Field(0.0, ge=0, le=1, description="置信度")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "performance",
                "severity": "medium",
                "message": "使用列表推導式可以提高性能",
                "line_start": 10,
                "line_end": 12,
                "suggestion": "使用 [x*2 for x in items] 代替循環",
                "confidence": 0.85
            }
        }


class ComplexityMetrics(BaseModel):
    """複雜度指標"""
    cyclomatic_complexity: int = Field(0, description="圈複雜度")
    cognitive_complexity: int = Field(0, description="認知複雜度")
    lines_of_code: int = Field(0, description="代碼行數")
    functions_count: int = Field(0, description="函數數量")
    classes_count: int = Field(0, description="類數量")
    max_nesting_depth: int = Field(0, description="最大嵌套深度")


class ReviewResult(BaseModel):
    """審查結果"""
    filename: str
    language: str
    summary: str = Field("", description="總結")
    score: float = Field(0.0, ge=0, le=10, description="代碼質量分數")
    issues: List[CodeIssue] = Field(default_factory=list)
    complexity_metrics: Optional[ComplexityMetrics] = None
    suggestions: List[str] = Field(default_factory=list, description="改進建議")
    optimized_code: Optional[str] = Field(None, description="優化後的代碼")
    review_time: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_issues_by_severity(self, severity: Severity) -> List[CodeIssue]:
        """按嚴重程度獲取問題"""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_critical_count(self) -> int:
        """獲取嚴重問題數量"""
        return len(self.get_issues_by_severity(Severity.CRITICAL))

    def has_critical_issues(self) -> bool:
        """是否有嚴重問題"""
        return self.get_critical_count() > 0


class TestCase(BaseModel):
    """測試用例"""
    name: str
    description: str
    test_code: str
    test_type: str = Field("unit", description="測試類型：unit, integration, e2e")
    target_function: Optional[str] = None


class GeneratedTests(BaseModel):
    """生成的測試"""
    filename: str
    language: str
    framework: str = Field("pytest", description="測試框架")
    test_cases: List[TestCase] = Field(default_factory=list)
    full_test_code: str = Field("", description="完整的測試代碼")
    coverage_estimate: float = Field(0.0, ge=0, le=100, description="預估覆蓋率")


class Documentation(BaseModel):
    """生成的文檔"""
    filename: str
    language: str
    docstring: str = Field("", description="文檔字符串")
    inline_comments: Dict[int, str] = Field(default_factory=dict, description="行號到註釋的映射")
    readme_section: Optional[str] = None
    api_docs: Optional[str] = None


class ReviewConfig(BaseModel):
    """審查配置"""
    severity_levels: List[Severity] = Field(
        default_factory=lambda: [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
    )
    enabled_checks: List[IssueType] = Field(
        default_factory=lambda: list(IssueType)
    )
    complexity_thresholds: Dict[str, int] = Field(
        default_factory=lambda: {
            "cyclomatic": 10,
            "cognitive": 15,
            "max_nesting": 4
        }
    )
    auto_fix: bool = Field(False, description="是否自動修復")
    include_suggestions: bool = Field(True, description="是否包含建議")
    max_issues: int = Field(50, description="最大問題數量")


class BatchReviewResult(BaseModel):
    """批量審查結果"""
    total_files: int = 0
    reviewed_files: int = 0
    failed_files: int = 0
    total_issues: int = 0
    critical_issues: int = 0
    results: List[ReviewResult] = Field(default_factory=list)
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    def add_result(self, result: ReviewResult):
        """添加審查結果"""
        self.results.append(result)
        self.reviewed_files += 1
        self.total_issues += len(result.issues)
        self.critical_issues += result.get_critical_count()

    def finalize(self):
        """完成統計"""
        self.end_time = datetime.now()
        if self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class PRComment(BaseModel):
    """PR 評論"""
    file_path: str
    line: int
    body: str
    side: str = Field("RIGHT", description="LEFT 或 RIGHT")


class GitHubIntegrationResult(BaseModel):
    """GitHub 集成結果"""
    pr_number: int
    comments_posted: int = 0
    labels_added: List[str] = Field(default_factory=list)
    review_submitted: bool = False
    success: bool = True
    error_message: Optional[str] = None
