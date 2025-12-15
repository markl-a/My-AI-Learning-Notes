# ============================================================
# FastAPI LLM API - Dockerfile
# ============================================================
#
# 多階段構建，優化鏡像大小
# 支援生產環境部署
#
# 構建: docker build -t fastapi-llm-api .
# 運行: docker run -p 8000:8000 --env-file .env fastapi-llm-api
#
# ============================================================

# ==================== 基礎階段 ====================
FROM python:3.11-slim as base

# 設置環境變數
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 創建非 root 用戶
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# 安裝系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ==================== 依賴安裝階段 ====================
FROM base as dependencies

WORKDIR /app

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# ==================== 生產階段 ====================
FROM base as production

WORKDIR /app

# 從依賴階段複製已安裝的包
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# 複製應用代碼
COPY --chown=appuser:appgroup . .

# 切換到非 root 用戶
USER appuser

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 啟動命令
CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "8000"]
