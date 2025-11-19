#!/bin/bash

# ============================================================
# 部署腳本 - 自動化部署 AI 學習專案
# ============================================================
#
# 用法：
#   ./scripts/deploy.sh [環境] [項目]
#
# 環境：dev, staging, production
# 項目：rag-chatbot, document-analyzer, all
#
# ============================================================

set -euo pipefail

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 工具函數 ====================

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

# 檢查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 未安裝，請先安裝後再運行"
    fi
}

# ==================== 環境檢查 ====================

check_requirements() {
    log_info "檢查部署環境..."

    check_command "docker"
    check_command "docker-compose"
    check_command "git"

    # 檢查 Docker 是否運行
    if ! docker info &> /dev/null; then
        log_error "Docker 未運行，請啟動 Docker"
    fi

    log_success "環境檢查通過"
}

# ==================== 備份功能 ====================

backup_current_deployment() {
    local project=$1
    local env=$2

    log_info "備份當前部署..."

    local backup_dir="${PROJECT_ROOT}/backups/${env}/${project}/${TIMESTAMP}"
    mkdir -p "$backup_dir"

    # 備份配置文件
    if [ -f "${PROJECT_ROOT}/.env.${env}" ]; then
        cp "${PROJECT_ROOT}/.env.${env}" "$backup_dir/"
    fi

    # 備份數據目錄（如果存在）
    if [ -d "${PROJECT_ROOT}/data/${project}" ]; then
        cp -r "${PROJECT_ROOT}/data/${project}" "$backup_dir/"
    fi

    log_success "備份完成: $backup_dir"
}

# ==================== 構建功能 ====================

build_project() {
    local project=$1
    local project_path=""

    case $project in
        "rag-chatbot")
            project_path="5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot"
            ;;
        "document-analyzer")
            project_path="5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer"
            ;;
        *)
            log_error "未知項目: $project"
            ;;
    esac

    log_info "構建項目: $project"

    cd "${PROJECT_ROOT}/${project_path}"

    # 構建 Docker 鏡像
    docker-compose build --no-cache

    log_success "構建完成: $project"
}

# ==================== 部署功能 ====================

deploy_project() {
    local project=$1
    local env=$2
    local project_path=""

    case $project in
        "rag-chatbot")
            project_path="5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot"
            ;;
        "document-analyzer")
            project_path="5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer"
            ;;
        *)
            log_error "未知項目: $project"
            ;;
    esac

    log_info "部署項目: $project (環境: $env)"

    cd "${PROJECT_ROOT}/${project_path}"

    # 檢查環境配置文件
    if [ ! -f ".env" ] && [ ! -f ".env.${env}" ]; then
        log_warning "未找到環境配置文件，使用 .env.example"
        cp .env.example .env
    elif [ -f ".env.${env}" ]; then
        cp ".env.${env}" .env
    fi

    # 停止現有容器
    log_info "停止現有容器..."
    docker-compose down || true

    # 啟動新容器
    log_info "啟動新容器..."
    docker-compose up -d

    log_success "部署完成: $project"
}

# ==================== 健康檢查 ====================

health_check() {
    local project=$1
    local port=""

    case $project in
        "rag-chatbot")
            port="8000"
            ;;
        "document-analyzer")
            port="8001"
            ;;
        *)
            log_error "未知項目: $project"
            ;;
    esac

    log_info "執行健康檢查..."

    local max_retries=30
    local retry=0

    while [ $retry -lt $max_retries ]; do
        if curl -f "http://localhost:${port}/api/health" &> /dev/null; then
            log_success "健康檢查通過"
            return 0
        fi

        retry=$((retry + 1))
        log_info "等待服務啟動... ($retry/$max_retries)"
        sleep 2
    done

    log_error "健康檢查失敗，服務未能正常啟動"
}

# ==================== 回滾功能 ====================

rollback() {
    local project=$1
    local env=$2
    local backup_dir="${PROJECT_ROOT}/backups/${env}/${project}"

    log_warning "執行回滾..."

    # 找到最新的備份
    local latest_backup=$(ls -t "$backup_dir" | head -1)

    if [ -z "$latest_backup" ]; then
        log_error "未找到備份，無法回滾"
    fi

    log_info "回滾到備份: $latest_backup"

    # 恢復配置
    if [ -f "${backup_dir}/${latest_backup}/.env.${env}" ]; then
        cp "${backup_dir}/${latest_backup}/.env.${env}" "${PROJECT_ROOT}/.env.${env}"
    fi

    # 重新部署
    deploy_project "$project" "$env"

    log_success "回滾完成"
}

# ==================== 清理功能 ====================

cleanup() {
    log_info "清理舊的 Docker 資源..."

    # 清理未使用的鏡像
    docker image prune -f

    # 清理未使用的容器
    docker container prune -f

    # 清理未使用的卷
    docker volume prune -f

    # 清理舊備份（保留最近 5 個）
    find "${PROJECT_ROOT}/backups" -type d -mindepth 3 | \
        sort -r | tail -n +6 | xargs rm -rf || true

    log_success "清理完成"
}

# ==================== 主流程 ====================

main() {
    local env=${1:-"dev"}
    local project=${2:-"all"}

    echo "============================================================"
    echo "🚀 AI 學習專案部署腳本"
    echo "============================================================"
    echo ""
    echo "環境: $env"
    echo "項目: $project"
    echo ""
    echo "============================================================"

    # 確認部署
    if [ "$env" = "production" ]; then
        read -p "⚠️  即將部署到生產環境，是否繼續？ (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_warning "取消部署"
            exit 0
        fi
    fi

    # 檢查環境
    check_requirements

    # 處理部署
    if [ "$project" = "all" ]; then
        projects=("rag-chatbot" "document-analyzer")
    else
        projects=("$project")
    fi

    for proj in "${projects[@]}"; do
        echo ""
        echo "------------------------------------------------------------"
        echo "處理項目: $proj"
        echo "------------------------------------------------------------"

        # 備份
        backup_current_deployment "$proj" "$env"

        # 構建
        build_project "$proj"

        # 部署
        deploy_project "$proj" "$env"

        # 健康檢查
        if ! health_check "$proj"; then
            log_error "健康檢查失敗"
            read -p "是否回滾到上一個版本？ (yes/no): " rollback_confirm
            if [ "$rollback_confirm" = "yes" ]; then
                rollback "$proj" "$env"
            fi
            exit 1
        fi
    done

    # 清理
    cleanup

    echo ""
    echo "============================================================"
    log_success "🎉 所有項目部署完成！"
    echo "============================================================"
}

# 運行主流程
main "$@"
