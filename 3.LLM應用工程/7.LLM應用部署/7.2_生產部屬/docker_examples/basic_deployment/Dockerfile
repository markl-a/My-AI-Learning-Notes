# Multi-stage Dockerfile for LLM FastAPI Application
# 使用多階段構建減小最終鏡像大小

# ===== 階段 1: 構建階段 =====
FROM python:3.10-slim as builder

# 設置工作目錄
WORKDIR /app

# 安裝構建依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 安裝 Python 依賴到用戶目錄（便於複製到最終鏡像）
RUN pip install --user --no-cache-dir -r requirements.txt

# ===== 階段 2: 運行階段 =====
FROM python:3.10-slim

# 設置環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# 創建非 root 用戶（安全最佳實踐）
RUN useradd -m -u 1000 appuser

# 設置工作目錄
WORKDIR /app

# 從構建階段複製已安裝的依賴
COPY --from=builder /root/.local /root/.local

# 複製應用代碼
COPY --chown=appuser:appuser . .

# 切換到非 root 用戶
USER appuser

# 暴露端口
EXPOSE 8000

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# 啟動命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
