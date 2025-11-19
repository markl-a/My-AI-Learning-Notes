#!/bin/bash

# ============================================================
# 項目初始化腳本 - 快速設置開發環境
# ============================================================

set -euo pipefail

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# ==================== 檢查 Python ====================

check_python() {
    log_info "檢查 Python 版本..."

    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 未安裝"
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.9"

    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python 版本過低，需要 >= 3.9"
    fi

    log_success "Python 版本檢查通過: $PYTHON_VERSION"
}

# ==================== 創建虛擬環境 ====================

setup_venv() {
    log_info "設置 Python 虛擬環境..."

    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "虛擬環境已創建"
    else
        log_warning "虛擬環境已存在"
    fi

    # 激活虛擬環境
    source venv/bin/activate || . venv/Scripts/activate

    # 升級 pip
    pip install --upgrade pip

    log_success "虛擬環境設置完成"
}

# ==================== 安裝依賴 ====================

install_dependencies() {
    log_info "安裝項目依賴..."

    # 安裝基礎依賴
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    # 安裝開發依賴
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi

    log_success "依賴安裝完成"
}

# ==================== 配置環境變量 ====================

setup_env() {
    log_info "配置環境變量..."

    # RAG ChatBot
    if [ -f "5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot/.env.example" ]; then
        if [ ! -f "5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot/.env" ]; then
            cp "5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot/.env.example" \
               "5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot/.env"
            log_success "已創建 RAG-ChatBot .env 文件"
        fi
    fi

    # Document Analyzer
    if [ -f "5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer/.env.example" ]; then
        if [ ! -f "5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer/.env" ]; then
            cp "5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer/.env.example" \
               "5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer/.env"
            log_success "已創建 AI-Document-Analyzer .env 文件"
        fi
    fi

    log_warning "請編輯 .env 文件並填入 API 密鑰"
}

# ==================== 創建必要目錄 ====================

create_directories() {
    log_info "創建必要目錄..."

    mkdir -p logs
    mkdir -p data
    mkdir -p backups
    mkdir -p monitoring/data

    log_success "目錄創建完成"
}

# ==================== 初始化 Git Hooks ====================

setup_git_hooks() {
    log_info "設置 Git Hooks..."

    if [ -d ".git" ]; then
        # 創建 pre-commit hook
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook - 代碼質量檢查

echo "Running pre-commit checks..."

# 運行 ruff 檢查
if command -v ruff &> /dev/null; then
    ruff check . || exit 1
fi

# 運行測試（可選）
# pytest tests/ || exit 1

echo "Pre-commit checks passed!"
EOF

        chmod +x .git/hooks/pre-commit
        log_success "Git Hooks 設置完成"
    else
        log_warning "不是 Git 倉庫，跳過 Git Hooks 設置"
    fi
}

# ==================== 下載模型（可選）====================

download_models() {
    log_info "下載 Sentence Transformers 模型（可選）..."

    read -p "是否預下載嵌入模型？這將節省首次運行時間 (yes/no): " download

    if [ "$download" = "yes" ]; then
        python3 << 'EOF'
from sentence_transformers import SentenceTransformer
print("下載 all-MiniLM-L6-v2...")
SentenceTransformer('all-MiniLM-L6-v2')
print("模型下載完成！")
EOF
        log_success "模型下載完成"
    else
        log_info "跳過模型下載"
    fi
}

# ==================== 運行測試 ====================

run_tests() {
    log_info "運行測試..."

    read -p "是否運行測試？(yes/no): " run_test

    if [ "$run_test" = "yes" ]; then
        pytest tests/ -v || log_warning "部分測試失敗"
        log_success "測試完成"
    else
        log_info "跳過測試"
    fi
}

# ==================== 顯示後續步驟 ====================

show_next_steps() {
    echo ""
    echo "============================================================"
    echo "🎉 設置完成！"
    echo "============================================================"
    echo ""
    echo "後續步驟："
    echo ""
    echo "1. 配置 API 密鑰："
    echo "   編輯 .env 文件並添加你的 OpenAI API 密鑰"
    echo ""
    echo "2. 激活虛擬環境："
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate     # Windows"
    echo ""
    echo "3. 運行項目："
    echo "   cd 5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot"
    echo "   python main.py"
    echo ""
    echo "4. 或使用 Docker："
    echo "   docker-compose up -d"
    echo ""
    echo "5. 訪問 API 文檔："
    echo "   http://localhost:8000/docs  # RAG ChatBot"
    echo "   http://localhost:8001/docs  # Document Analyzer"
    echo ""
    echo "============================================================"
}

# ==================== 主流程 ====================

main() {
    echo "============================================================"
    echo "🚀 AI 學習專案初始化腳本"
    echo "============================================================"
    echo ""

    check_python
    setup_venv
    install_dependencies
    setup_env
    create_directories
    setup_git_hooks
    download_models
    run_tests
    show_next_steps
}

main "$@"
