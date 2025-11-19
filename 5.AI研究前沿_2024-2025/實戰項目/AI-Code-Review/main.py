"""
AI 代碼審查助手 - 主應用
自動化代碼審查、安全檢查、性能分析、最佳實踐建議
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime
import os

from code_analyzer import CodeAnalyzer
from security_checker import SecurityChecker
from performance_analyzer import PerformanceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 代碼審查助手",
    description="智能代碼審查、安全檢查和性能分析",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化組件
code_analyzer = CodeAnalyzer()
security_checker = SecurityChecker()
performance_analyzer = PerformanceAnalyzer()

# ==================== Pydantic 模型 ====================

class CodeReviewRequest(BaseModel):
    """代碼審查請求"""
    code: str
    language: str  # python, javascript, java, go, etc.
    review_type: str = "full"  # full, quick, security, performance
    context: Optional[Dict[str, Any]] = {}


class SecurityCheckRequest(BaseModel):
    """安全檢查請求"""
    code: str
    language: str
    check_types: List[str] = ["all"]  # sql_injection, xss, secrets, etc.


class PerformanceAnalysisRequest(BaseModel):
    """性能分析請求"""
    code: str
    language: str
    analysis_depth: str = "medium"  # quick, medium, deep


class CodeRefactorRequest(BaseModel):
    """代碼重構請求"""
    code: str
    language: str
    refactor_goals: List[str]  # readability, performance, maintainability


class BatchReviewRequest(BaseModel):
    """批量審查請求"""
    repository_url: Optional[str] = None
    file_paths: Optional[List[str]] = None
    review_type: str = "full"


# ==================== 健康檢查 ====================

@app.get("/api/health")
async def health_check():
    """健康檢查"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Code Review Assistant"
    }


@app.get("/api/stats")
async def get_stats():
    """獲取統計信息"""
    return {
        "total_reviews": code_analyzer.get_review_count(),
        "supported_languages": code_analyzer.get_supported_languages(),
        "avg_review_time": code_analyzer.get_avg_review_time(),
        "security_issues_found": security_checker.get_total_issues()
    }


# ==================== 代碼審查 ====================

@app.post("/api/review")
async def review_code(request: CodeReviewRequest):
    """
    審查代碼

    審查類型：
    - full: 完整審查（代碼質量、安全、性能、最佳實踐）
    - quick: 快速審查（主要問題）
    - security: 安全審查
    - performance: 性能審查
    """
    try:
        logger.info(f"Reviewing {request.language} code, type: {request.review_type}")

        result = await code_analyzer.review_code(
            code=request.code,
            language=request.language,
            review_type=request.review_type,
            context=request.context
        )

        return {
            "review_id": result["review_id"],
            "language": request.language,
            "review_type": request.review_type,
            "overall_score": result["overall_score"],
            "issues": result["issues"],
            "suggestions": result["suggestions"],
            "metrics": result["metrics"],
            "summary": result["summary"],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error reviewing code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/review/file")
async def review_file(
    file: UploadFile = File(...),
    review_type: str = "full"
):
    """上傳文件進行審查"""
    try:
        # 讀取文件
        content = await file.read()
        code = content.decode('utf-8')

        # 從文件名推斷語言
        file_ext = file.filename.split('.')[-1]
        language = code_analyzer.detect_language(file_ext)

        # 審查代碼
        result = await code_analyzer.review_code(
            code=code,
            language=language,
            review_type=review_type,
            context={"filename": file.filename}
        )

        return {
            "filename": file.filename,
            "language": language,
            "review_id": result["review_id"],
            "overall_score": result["overall_score"],
            "issues": result["issues"],
            "suggestions": result["suggestions"]
        }

    except Exception as e:
        logger.error(f"Error reviewing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 安全檢查 ====================

@app.post("/api/security/check")
async def security_check(request: SecurityCheckRequest):
    """
    安全漏洞檢查

    檢查類型：
    - sql_injection: SQL 注入
    - xss: 跨站腳本攻擊
    - secrets: 敏感信息洩露
    - authentication: 認證問題
    - authorization: 授權問題
    - input_validation: 輸入驗證
    - crypto: 加密問題
    """
    try:
        logger.info(f"Security check for {request.language} code")

        vulnerabilities = await security_checker.check_vulnerabilities(
            code=request.code,
            language=request.language,
            check_types=request.check_types
        )

        return {
            "security_score": vulnerabilities["score"],
            "vulnerabilities": vulnerabilities["issues"],
            "severity_distribution": vulnerabilities["severity_distribution"],
            "recommendations": vulnerabilities["recommendations"],
            "compliant_with": vulnerabilities["compliant_with"],  # OWASP, CWE
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in security check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/security/scan-secrets")
async def scan_secrets(request: SecurityCheckRequest):
    """掃描敏感信息（API 密鑰、密碼等）"""
    try:
        secrets_found = await security_checker.scan_secrets(
            code=request.code,
            language=request.language
        )

        return {
            "secrets_found": len(secrets_found),
            "secrets": secrets_found,
            "risk_level": "high" if secrets_found else "low",
            "recommendations": [
                "使用環境變量存儲敏感信息",
                "使用密鑰管理服務（如 AWS Secrets Manager）",
                "永遠不要將敏感信息提交到版本控制"
            ]
        }

    except Exception as e:
        logger.error(f"Error scanning secrets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 性能分析 ====================

@app.post("/api/performance/analyze")
async def analyze_performance(request: PerformanceAnalysisRequest):
    """
    性能分析

    分析項目：
    - 時間複雜度
    - 空間複雜度
    - 性能瓶頸識別
    - 優化建議
    """
    try:
        logger.info(f"Performance analysis for {request.language} code")

        analysis = await performance_analyzer.analyze(
            code=request.code,
            language=request.language,
            depth=request.analysis_depth
        )

        return {
            "performance_score": analysis["score"],
            "time_complexity": analysis["time_complexity"],
            "space_complexity": analysis["space_complexity"],
            "bottlenecks": analysis["bottlenecks"],
            "optimization_suggestions": analysis["optimizations"],
            "estimated_speedup": analysis["estimated_speedup"],
            "metrics": {
                "loops": analysis["loop_count"],
                "recursive_calls": analysis["recursion_depth"],
                "database_queries": analysis["db_queries"]
            }
        }

    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 代碼重構建議 ====================

@app.post("/api/refactor")
async def suggest_refactor(request: CodeRefactorRequest):
    """
    生成重構建議

    重構目標：
    - readability: 可讀性提升
    - performance: 性能優化
    - maintainability: 可維護性
    - testability: 可測試性
    - modularity: 模塊化
    """
    try:
        logger.info(f"Generating refactor suggestions for {request.language} code")

        refactor_plan = await code_analyzer.suggest_refactoring(
            code=request.code,
            language=request.language,
            goals=request.refactor_goals
        )

        return {
            "refactor_id": refactor_plan["id"],
            "original_code": request.code,
            "refactored_code": refactor_plan["refactored_code"],
            "changes": refactor_plan["changes"],
            "improvements": refactor_plan["improvements"],
            "risks": refactor_plan["risks"],
            "effort_estimate": refactor_plan["effort"],
            "priority": refactor_plan["priority"]
        }

    except Exception as e:
        logger.error(f"Error generating refactor suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 最佳實踐檢查 ====================

@app.post("/api/best-practices")
async def check_best_practices(request: CodeReviewRequest):
    """
    檢查最佳實踐

    檢查項目：
    - 命名規範
    - 代碼風格
    - 設計模式使用
    - 錯誤處理
    - 日誌記錄
    - 文檔註釋
    """
    try:
        best_practices = await code_analyzer.check_best_practices(
            code=request.code,
            language=request.language
        )

        return {
            "compliance_score": best_practices["score"],
            "violations": best_practices["violations"],
            "good_practices": best_practices["good_practices"],
            "recommendations": best_practices["recommendations"],
            "style_guide": best_practices["style_guide"],
            "patterns_used": best_practices["patterns"],
            "anti_patterns": best_practices["anti_patterns"]
        }

    except Exception as e:
        logger.error(f"Error checking best practices: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 代碼複雜度分析 ====================

@app.post("/api/complexity")
async def analyze_complexity(request: CodeReviewRequest):
    """
    分析代碼複雜度

    指標：
    - 圈複雜度（Cyclomatic Complexity）
    - 認知複雜度（Cognitive Complexity）
    - 維護性指數（Maintainability Index）
    - 代碼行數統計
    """
    try:
        complexity = await code_analyzer.analyze_complexity(
            code=request.code,
            language=request.language
        )

        return {
            "cyclomatic_complexity": complexity["cyclomatic"],
            "cognitive_complexity": complexity["cognitive"],
            "maintainability_index": complexity["maintainability_index"],
            "lines_of_code": complexity["loc"],
            "comment_ratio": complexity["comment_ratio"],
            "function_count": complexity["function_count"],
            "class_count": complexity["class_count"],
            "complexity_rating": complexity["rating"],  # A, B, C, D, F
            "suggestions": complexity["suggestions"]
        }

    except Exception as e:
        logger.error(f"Error analyzing complexity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 批量審查 ====================

@app.post("/api/review/batch")
async def batch_review(
    request: BatchReviewRequest,
    background_tasks: BackgroundTasks
):
    """
    批量審查代碼庫

    支持：
    - Git 倉庫 URL
    - 多個文件路徑
    """
    try:
        task_id = code_analyzer.create_batch_task()

        # 在後台執行批量審查
        background_tasks.add_task(
            code_analyzer.batch_review,
            task_id=task_id,
            repository_url=request.repository_url,
            file_paths=request.file_paths,
            review_type=request.review_type
        )

        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Batch review started in background"
        }

    except Exception as e:
        logger.error(f"Error starting batch review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/review/batch/{task_id}")
async def get_batch_review_status(task_id: str):
    """獲取批量審查狀態"""
    try:
        status = code_analyzer.get_batch_task_status(task_id)

        if not status:
            raise HTTPException(status_code=404, detail="Task not found")

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch review status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 代碼比較 ====================

@app.post("/api/compare")
async def compare_code(
    old_code: str,
    new_code: str,
    language: str
):
    """
    比較兩個版本的代碼

    分析：
    - 變更內容
    - 質量改進
    - 新引入的問題
    - 性能影響
    """
    try:
        comparison = await code_analyzer.compare_versions(
            old_code=old_code,
            new_code=new_code,
            language=language
        )

        return {
            "changes": comparison["changes"],
            "quality_delta": comparison["quality_delta"],
            "new_issues": comparison["new_issues"],
            "fixed_issues": comparison["fixed_issues"],
            "performance_impact": comparison["performance_impact"],
            "recommendation": comparison["recommendation"]
        }

    except Exception as e:
        logger.error(f"Error comparing code: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 代碼生成 ====================

@app.post("/api/generate/tests")
async def generate_tests(
    code: str,
    language: str,
    test_framework: Optional[str] = None
):
    """
    自動生成單元測試

    支持的框架：
    - Python: pytest, unittest
    - JavaScript: jest, mocha
    - Java: junit
    """
    try:
        tests = await code_analyzer.generate_unit_tests(
            code=code,
            language=language,
            framework=test_framework
        )

        return {
            "test_code": tests["code"],
            "test_cases": tests["cases"],
            "coverage_estimate": tests["coverage"],
            "framework": tests["framework"],
            "setup_instructions": tests["setup"]
        }

    except Exception as e:
        logger.error(f"Error generating tests: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/docs")
async def generate_documentation(
    code: str,
    language: str,
    doc_format: str = "markdown"
):
    """
    自動生成文檔

    格式：
    - markdown
    - restructuredtext
    - jsdoc
    - javadoc
    """
    try:
        docs = await code_analyzer.generate_documentation(
            code=code,
            language=language,
            format=doc_format
        )

        return {
            "documentation": docs["content"],
            "format": doc_format,
            "sections": docs["sections"],
            "api_reference": docs["api_reference"]
        }

    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 啟動應用 ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
