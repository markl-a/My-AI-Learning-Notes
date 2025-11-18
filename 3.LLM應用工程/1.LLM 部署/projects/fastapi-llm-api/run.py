#!/usr/bin/env python3
"""
FastAPI LLM API 启动脚本
"""

import uvicorn
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

if __name__ == "__main__":
    # 配置
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    reload = os.getenv("DEBUG", "true").lower() == "true"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              FastAPI LLM API Server                          ║
╠══════════════════════════════════════════════════════════════╣
║  🚀 Server starting...                                       ║
║  📍 Host: {host:<48}║
║  🔌 Port: {port:<48}║
║  👷 Workers: {workers:<45}║
║  🔄 Auto-reload: {str(reload):<42}║
║                                                              ║
║  📚 API Documentation:                                       ║
║     • Swagger UI: http://{host}:{port}/docs{' ' * (27 - len(host) - len(str(port)))}║
║     • ReDoc: http://{host}:{port}/redoc{' ' * (32 - len(host) - len(str(port)))}║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )
