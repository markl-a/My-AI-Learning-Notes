#!/bin/bash
# 部署腳本

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
NAMESPACE="${NAMESPACE:-llm-production}"
DEPLOYMENT="${DEPLOYMENT:-llm-service}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HELM_RELEASE="${HELM_RELEASE:-llm-service}"

# 函數：打印帶顏色的消息
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函數：檢查必要的工具
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        log_warn "helm not found. Skipping Helm deployment."
    fi

    log_info "Prerequisites check passed."
}

# 函數：驗證 Kubernetes 連接
validate_k8s_connection() {
    log_info "Validating Kubernetes connection..."

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi

    log_info "Connected to Kubernetes cluster."
}

# 函數：創建命名空間（如果不存在）
ensure_namespace() {
    log_info "Ensuring namespace '${NAMESPACE}' exists..."

    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        kubectl create namespace "${NAMESPACE}"
        log_info "Namespace '${NAMESPACE}' created."
    else
        log_info "Namespace '${NAMESPACE}' already exists."
    fi
}

# 函數：備份當前部署
backup_deployment() {
    log_info "Backing up current deployment..."

    local backup_file="backup-$(date +%Y%m%d-%H%M%S).yaml"

    kubectl get deployment "${DEPLOYMENT}" -n "${NAMESPACE}" -o yaml > "${backup_file}" 2>/dev/null || {
        log_warn "No existing deployment to backup."
        return 0
    }

    log_info "Backup saved to ${backup_file}"
}

# 函數：部署應用
deploy_application() {
    log_info "Deploying application..."

    if command -v helm &> /dev/null; then
        # 使用 Helm 部署
        log_info "Deploying with Helm..."

        helm upgrade --install "${HELM_RELEASE}" ./helm/llm-service \
            --namespace "${NAMESPACE}" \
            --set image.tag="${IMAGE_TAG}" \
            --wait \
            --timeout 10m

    else
        # 使用 kubectl 部署
        log_info "Deploying with kubectl..."

        kubectl apply -f k8s/configmap.yaml
        kubectl apply -f k8s/secrets.yaml
        kubectl apply -f k8s/deployment.yaml
        kubectl apply -f k8s/service.yaml
        kubectl apply -f k8s/ingress.yaml

        # 更新鏡像
        kubectl set image deployment/"${DEPLOYMENT}" \
            llm-service="your-registry/llm-service:${IMAGE_TAG}" \
            -n "${NAMESPACE}"
    fi

    log_info "Deployment completed."
}

# 函數：等待部署就緒
wait_for_rollout() {
    log_info "Waiting for rollout to complete..."

    if kubectl rollout status deployment/"${DEPLOYMENT}" -n "${NAMESPACE}" --timeout=10m; then
        log_info "Rollout completed successfully."
    else
        log_error "Rollout failed or timed out."
        return 1
    fi
}

# 函數：驗證部署
verify_deployment() {
    log_info "Verifying deployment..."

    # 檢查 Pod 狀態
    local ready_pods=$(kubectl get pods -n "${NAMESPACE}" -l app=llm-service -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' | grep -o "True" | wc -l)
    local total_pods=$(kubectl get pods -n "${NAMESPACE}" -l app=llm-service --no-headers | wc -l)

    log_info "Ready pods: ${ready_pods}/${total_pods}"

    if [ "${ready_pods}" -lt 1 ]; then
        log_error "No pods are ready."
        return 1
    fi

    # 健康檢查
    log_info "Running health check..."

    local service_url=$(kubectl get ingress -n "${NAMESPACE}" -o jsonpath='{.items[0].spec.rules[0].host}')

    if [ -n "${service_url}" ]; then
        local health_status=$(curl -s -o /dev/null -w "%{http_code}" "https://${service_url}/health" || echo "000")

        if [ "${health_status}" == "200" ]; then
            log_info "Health check passed."
        else
            log_warn "Health check returned status: ${health_status}"
        fi
    else
        log_warn "Could not determine service URL for health check."
    fi

    log_info "Deployment verification completed."
}

# 函數：顯示部署狀態
show_status() {
    log_info "Deployment status:"

    kubectl get pods -n "${NAMESPACE}" -l app=llm-service
    echo ""
    kubectl get svc -n "${NAMESPACE}" -l app=llm-service
    echo ""
    kubectl get ingress -n "${NAMESPACE}"
}

# 函數：回滾部署
rollback_deployment() {
    log_warn "Rolling back deployment..."

    kubectl rollout undo deployment/"${DEPLOYMENT}" -n "${NAMESPACE}"

    log_warn "Rollback initiated. Waiting for completion..."

    wait_for_rollout

    log_warn "Rollback completed."
}

# 主函數
main() {
    log_info "Starting deployment to ${NAMESPACE}..."
    log_info "Image tag: ${IMAGE_TAG}"

    check_prerequisites
    validate_k8s_connection
    ensure_namespace
    backup_deployment

    if deploy_application && wait_for_rollout && verify_deployment; then
        show_status
        log_info "Deployment successful! 🎉"
    else
        log_error "Deployment failed."

        if [ "${AUTO_ROLLBACK}" == "true" ]; then
            rollback_deployment
        fi

        exit 1
    fi
}

# 運行主函數
main
