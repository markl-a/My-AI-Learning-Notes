# Makefile for My-AI-Learning-Notes
# 統一本地開發命令界面

.PHONY: help setup install install-dev install-full lint format test test-cov security docs clean deploy-dev deploy-staging deploy-prod

# 預設目標：顯示幫助
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     My-AI-Learning-Notes - 開發命令指南                      ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 安裝命令:"
	@echo "  make setup        - 完整環境設置（首次使用）"
	@echo "  make install      - 安裝核心依賴"
	@echo "  make install-dev  - 安裝開發依賴"
	@echo "  make install-full - 安裝所有依賴（ML+DL+LLM）"
	@echo ""
	@echo "🔍 代碼品質:"
	@echo "  make lint         - 運行代碼檢查（Ruff）"
	@echo "  make format       - 格式化代碼（Black + isort）"
	@echo "  make type-check   - 類型檢查（MyPy）"
	@echo "  make security     - 安全掃描（Bandit）"
	@echo "  make check-all    - 運行所有檢查"
	@echo ""
	@echo "🧪 測試:"
	@echo "  make test         - 運行單元測試"
	@echo "  make test-cov     - 運行測試 + 覆蓋率報告"
	@echo "  make test-fast    - 快速測試（跳過慢速測試）"
	@echo ""
	@echo "📚 文檔:"
	@echo "  make docs         - 生成 MkDocs 文檔"
	@echo "  make docs-serve   - 本地預覽文檔"
	@echo ""
	@echo "🚀 部署:"
	@echo "  make deploy-dev   - 部署到開發環境"
	@echo "  make deploy-prod  - 部署到生產環境"
	@echo ""
	@echo "🧹 清理:"
	@echo "  make clean        - 清理生成的文件"
	@echo "  make clean-all    - 深度清理（包括緩存）"

# ============================================================
# 安裝命令
# ============================================================

setup:
	@echo "🚀 開始完整環境設置..."
	@if [ -f scripts/setup.sh ]; then \
		bash scripts/setup.sh; \
	else \
		echo "⚠️  setup.sh 不存在，使用備用安裝流程"; \
		python -m venv .venv; \
		. .venv/bin/activate && pip install --upgrade pip; \
		. .venv/bin/activate && pip install -r requirements.txt; \
		. .venv/bin/activate && pip install -r requirements-dev.txt; \
		. .venv/bin/activate && pre-commit install; \
	fi
	@echo "✅ 環境設置完成！"

install:
	@echo "📦 安裝核心依賴..."
	pip install -r requirements.txt
	@echo "✅ 核心依賴安裝完成"

install-dev:
	@echo "📦 安裝開發依賴..."
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "✅ 開發依賴安裝完成"

install-full:
	@echo "📦 安裝完整依賴（可能需要較長時間）..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -r requirements-ml.txt
	pip install -r requirements-dl.txt
	pip install -r requirements-llm.txt
	@echo "✅ 完整依賴安裝完成"

# ============================================================
# 代碼品質
# ============================================================

lint:
	@echo "🔍 運行代碼檢查..."
	ruff check .
	@echo "✅ 代碼檢查完成"

format:
	@echo "✨ 格式化代碼..."
	black .
	isort .
	@echo "✅ 代碼格式化完成"

type-check:
	@echo "🔎 運行類型檢查..."
	mypy . --ignore-missing-imports || true
	@echo "✅ 類型檢查完成"

security:
	@echo "🔒 運行安全掃描..."
	bandit -r . -x ./tests,./venv,./.venv -ll || true
	@echo "✅ 安全掃描完成"

check-all: lint type-check security
	@echo "✅ 所有檢查完成"

# ============================================================
# 測試
# ============================================================

test:
	@echo "🧪 運行單元測試..."
	pytest tests/ -v
	@echo "✅ 測試完成"

test-cov:
	@echo "🧪 運行測試（含覆蓋率）..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing
	@echo "📊 覆蓋率報告已生成: htmlcov/index.html"

test-fast:
	@echo "⚡ 快速測試..."
	pytest tests/ -v -m "not slow"
	@echo "✅ 快速測試完成"

# ============================================================
# 文檔
# ============================================================

docs:
	@echo "📚 生成文檔..."
	@if command -v mkdocs &> /dev/null; then \
		mkdocs build; \
		echo "✅ 文檔生成完成: site/"; \
	else \
		echo "⚠️  請先安裝 mkdocs: pip install mkdocs mkdocs-material"; \
	fi

docs-serve:
	@echo "📚 啟動文檔預覽服務..."
	@if command -v mkdocs &> /dev/null; then \
		mkdocs serve; \
	else \
		echo "⚠️  請先安裝 mkdocs: pip install mkdocs mkdocs-material"; \
	fi

# ============================================================
# 部署
# ============================================================

deploy-dev:
	@echo "🚀 部署到開發環境..."
	@if [ -f scripts/deploy.sh ]; then \
		bash scripts/deploy.sh dev all; \
	else \
		echo "❌ deploy.sh 不存在"; \
	fi

deploy-staging:
	@echo "🚀 部署到測試環境..."
	@if [ -f scripts/deploy.sh ]; then \
		bash scripts/deploy.sh staging all; \
	else \
		echo "❌ deploy.sh 不存在"; \
	fi

deploy-prod:
	@echo "⚠️  即將部署到生產環境..."
	@read -p "確認部署到生產環境？(y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	@if [ -f scripts/deploy.sh ]; then \
		bash scripts/deploy.sh production all; \
	else \
		echo "❌ deploy.sh 不存在"; \
	fi

# ============================================================
# 清理
# ============================================================

clean:
	@echo "🧹 清理生成的文件..."
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf dist build *.egg-info
	rm -rf site
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ 清理完成"

clean-all: clean
	@echo "🧹 深度清理..."
	rm -rf .venv venv
	rm -rf node_modules
	rm -rf .jupyter
	@echo "✅ 深度清理完成"

# ============================================================
# 開發輔助
# ============================================================

pre-commit:
	@echo "🔧 運行 pre-commit 檢查..."
	pre-commit run --all-files

update-deps:
	@echo "📦 更新依賴..."
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt
	@echo "✅ 依賴更新完成"

notebook:
	@echo "📓 啟動 Jupyter Lab..."
	jupyter lab

# ============================================================
# 快捷命令
# ============================================================

# 開發前檢查
dev-check: format lint test-fast
	@echo "✅ 開發前檢查完成，可以提交代碼"

# 完整 CI 模擬
ci: check-all test-cov
	@echo "✅ CI 檢查完成"
