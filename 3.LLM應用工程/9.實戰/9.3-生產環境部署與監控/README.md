# 9.3 部署至生產環境並持續監控、調優與版本控制

## 項目概述

這是一個完整的生產級 LLM 應用部署方案，涵蓋從開發到生產的完整 MLOps 流程。包括：

- **容器化部署**：Docker + Kubernetes
- **基礎設施即程式碼**：Terraform + Ansible
- **CI/CD 流水線**：自動化構建、測試、部署
- **監控和告警**：Prometheus + Grafana + ELK Stack
- **性能優化**：快取、負載均衡、自動擴展
- **A/B 測試**：模型版本對比和漸進式發布
- **成本優化**：資源管理和成本追蹤

## 系統架構

```
                                ┌─────────────┐
                                │  用戶請求    │
                                └──────┬──────┘
                                       │
                                ┌──────▼──────┐
                                │  CloudFlare  │
                                │     CDN      │
                                └──────┬──────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │  Nginx Ingress Controller  │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │     API Gateway (Kong)     │
                         │  - Rate Limiting           │
                         │  - Authentication          │
                         │  - Request Routing         │
                         └─────────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
       ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
       │  LLM Service │         │  LLM Service │         │  LLM Service │
       │   Pod 1      │         │   Pod 2      │         │   Pod 3      │
       │  (Model A)   │         │  (Model A)   │         │  (Model B)   │
       └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │      Data Layer           │
                         │  - Redis Cache            │
                         │  - PostgreSQL             │
                         │  - ChromaDB (Vector)      │
                         └───────────────────────────┘

                         ┌───────────────────────────┐
                         │   Monitoring Stack        │
                         │  - Prometheus             │
                         │  - Grafana                │
                         │  - ELK Stack              │
                         │  - Jaeger (Tracing)       │
                         └───────────────────────────┘
```

## 核心功能

### 1. 容器化和編排

- **Docker 多階段構建**：優化鏡像大小
- **Kubernetes 部署**：自動擴展、滾動更新
- **Helm Charts**：可配置的部署模板
- **Service Mesh (Istio)**：流量管理和安全

### 2. CI/CD 流水線

```yaml
階段：
1. 程式碼提交 → Git Hook
2. 自動測試 → 單元測試、集成測試
3. 程式碼審查 → AI 程式碼審查
4. 構建鏡像 → Docker Build + Push
5. 安全掃描 → Container Scan
6. 部署到 Staging
7. 自動化測試 → E2E Testing
8. 部署到 Production (Canary/Blue-Green)
9. 健康檢查
10. 監控和告警
```

### 3. 監控系統

#### 指標監控 (Prometheus + Grafana)
- **系統指標**：CPU、內存、磁盤、網絡
- **應用指標**：
  - 請求率 (RPS)
  - 響應時間 (P50, P95, P99)
  - 錯誤率
  - LLM Token 使用量
  - 快取命中率
- **業務指標**：
  - 用戶活躍度
  - 查詢成功率
  - 平均查詢耗時

#### 日誌管理 (ELK Stack)
- **Elasticsearch**：日誌存儲和搜索
- **Logstash**：日誌聚合和處理
- **Kibana**：日誌可視化和分析

#### 分佈式追蹤 (Jaeger)
- 請求鏈路追蹤
- 性能瓶頸分析
- 依賴關係可視化

### 4. 性能優化

- **多級快取**：
  - L1: 本地內存快取 (LRU)
  - L2: Redis 分佈式快取
  - L3: CDN 快取

- **負載均衡**：
  - Nginx Ingress
  - Service Level Load Balancing
  - Client-side Load Balancing

- **自動擴展**：
  - HPA (Horizontal Pod Autoscaler)
  - VPA (Vertical Pod Autoscaler)
  - Cluster Autoscaler

### 5. 成本優化

- **資源配額管理**
- **Spot Instances 使用**
- **空閒資源回收**
- **成本分析和報告**

## 快速開始

### 前置條件

```bash
# 必需工具
- Docker >= 20.10
- Kubernetes >= 1.24
- kubectl
- helm >= 3.0
- terraform >= 1.0

# 可選工具
- k9s (Kubernetes CLI)
- kubectx (上下文切換)
```

### 1. 本地開發環境

```bash
# 使用 Docker Compose 啟動本地環境
cd local-dev
docker-compose up -d

# 驗證服務
curl http://localhost:8000/health
```

### 2. 部署到 Kubernetes

#### 方式 A：使用 Helm

```bash
# 添加 Helm 倉庫
helm repo add llm-app ./helm/llm-service
helm repo update

# 安裝應用
helm install llm-service llm-app/llm-service \
  --namespace production \
  --create-namespace \
  --values values-production.yaml

# 檢查部署狀態
kubectl get pods -n production
kubectl get svc -n production
```

#### 方式 B：使用 Kubectl

```bash
# 應用 Kubernetes 清單
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# 檢查部署
kubectl get all -n llm-production
```

### 3. 配置 CI/CD

#### GitHub Actions

```bash
# 設置 Secrets
gh secret set DOCKERHUB_USERNAME
gh secret set DOCKERHUB_TOKEN
gh secret set KUBECONFIG_DATA
gh secret set OPENAI_API_KEY

# 手動觸發部署
gh workflow run deploy-production.yml
```

#### GitLab CI

```bash
# 配置 CI/CD 變量
- KUBE_CONFIG
- DOCKER_REGISTRY_TOKEN
- OPENAI_API_KEY

# 推送程式碼觸發自動部署
git push origin main
```

### 4. 訪問監控面板

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# 瀏覽器訪問
# http://localhost:3000
# 預設登錄: admin / admin

# 導入預設儀表板
# dashboards/llm-service-dashboard.json
```

## 項目結構

```
9.3-生產環境部署與監控/
├── README.md                      # 本文件
├── docker/
│   ├── Dockerfile                # 生產環境 Dockerfile
│   ├── Dockerfile.dev            # 開發環境 Dockerfile
│   └── docker-compose.yml        # 本地開發環境
├── k8s/
│   ├── namespace.yaml            # 命名空間
│   ├── configmap.yaml            # 配置映射
│   ├── secrets.yaml              # 密鑰
│   ├── deployment.yaml           # 部署配置
│   ├── service.yaml              # 服務
│   ├── ingress.yaml              # 入口
│   ├── hpa.yaml                  # 自動擴展
│   └── service-monitor.yaml      # Prometheus 監控
├── helm/
│   └── llm-service/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-staging.yaml
│       ├── values-production.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           └── ...
├── terraform/
│   ├── main.tf                   # 主配置
│   ├── variables.tf              # 變量定義
│   ├── outputs.tf                # 輸出
│   ├── modules/
│   │   ├── eks/                  # AWS EKS 模塊
│   │   ├── gke/                  # Google GKE 模塊
│   │   └── aks/                  # Azure AKS 模塊
│   └── environments/
│       ├── dev/
│       ├── staging/
│       └── production/
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml        # Prometheus 配置
│   │   └── alerts.yml            # 告警規則
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── llm-service.json
│   │   │   ├── kubernetes.json
│   │   │   └── cost-analysis.json
│   │   └── provisioning/
│   ├── elasticsearch/
│   │   └── elasticsearch.yml
│   └── jaeger/
│       └── jaeger-all-in-one.yaml
├── scripts/
│   ├── deploy.sh                 # 部署腳本
│   ├── rollback.sh               # 回滾腳本
│   ├── backup.sh                 # 備份腳本
│   ├── health-check.sh           # 健康檢查
│   └── load-test.sh              # 壓力測試
├── .github/
│   └── workflows/
│       ├── ci.yml                # 持續集成
│       ├── cd-staging.yml        # 部署到 Staging
│       └── cd-production.yml     # 部署到 Production
└── docs/
    ├── architecture.md           # 架構文檔
    ├── deployment-guide.md       # 部署指南
    ├── monitoring-guide.md       # 監控指南
    ├── troubleshooting.md        # 故障排除
    └── runbook.md                # 運維手冊
```

## 部署策略

### 1. 藍綠部署 (Blue-Green Deployment)

```yaml
# 零停機部署
1. 部署新版本（綠色環境）
2. 運行健康檢查和煙霧測試
3. 切換流量到綠色環境
4. 監控性能和錯誤率
5. 如果正常，保留綠色；如果異常，切回藍色
6. 清理舊版本
```

**優點**：快速回滾、零停機
**缺點**：需要雙倍資源

### 2. 金絲雀發布 (Canary Deployment)

```yaml
# 漸進式發布
1. 部署新版本到小部分節點（5%）
2. 監控關鍵指標
3. 逐步增加流量（10% → 25% → 50% → 100%）
4. 每個階段都進行監控
5. 發現問題立即回滾
```

**優點**：風險小、影響範圍可控
**缺點**：發布時間較長

### 3. 滾動更新 (Rolling Update)

```yaml
# Kubernetes 預設策略
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0
```

**優點**：簡單、自動化
**缺點**：回滾較慢

## 監控和告警

### 關鍵指標

#### 黃金信號 (Golden Signals)

1. **延遲 (Latency)**
   ```
   - P50 < 500ms
   - P95 < 1s
   - P99 < 2s
   ```

2. **流量 (Traffic)**
   ```
   - 每秒請求數 (RPS)
   - 帶寬使用
   ```

3. **錯誤 (Errors)**
   ```
   - 錯誤率 < 0.1%
   - 4xx 錯誤
   - 5xx 錯誤
   ```

4. **飽和度 (Saturation)**
   ```
   - CPU 使用率 < 70%
   - 內存使用率 < 80%
   - 磁盤使用率 < 80%
   ```

### 告警規則示例

```yaml
# Prometheus 告警規則
groups:
  - name: llm-service-alerts
    rules:
      # 高錯誤率
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/sec"

      # 高延遲
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      # Pod 不健康
      - alert: PodNotHealthy
        expr: kube_pod_status_phase{phase!="Running"} > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is not running"
```

## 性能優化實踐

### 1. LLM 推論優化

```python
# 批處理
batch_size = 8
requests = queue.get_batch(batch_size)
responses = llm.batch_generate(requests)

# 流式響應
for chunk in llm.stream_generate(prompt):
    yield chunk

# 模型量化
model = load_model("gpt-4", quantization="int8")

# KV Cache 優化
cache = KVCache(max_size=1000)
```

### 2. 快取策略

```python
# 多級快取
class CacheManager:
    def __init__(self):
        self.l1_cache = LRUCache(maxsize=100)    # 本地快取
        self.l2_cache = RedisCache()              # 分佈式快取

    def get(self, key):
        # L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value

        return None
```

### 3. 資源限制

```yaml
# Kubernetes 資源配置
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 成本優化

### 1. 資源優化

```yaml
# 使用 Spot Instances
nodeSelector:
  node.kubernetes.io/instance-type: spot

# 自動關閉空閒資源
- name: scale-down-idle-pods
  schedule: "0 2 * * *"  # 每天凌晨2點
```

### 2. LLM API 成本控制

```python
# Token 限制
max_tokens = 500

# 使用更便宜的模型
model = "gpt-3.5-turbo"  # 而不是 gpt-4

# 快取常見查詢
if query in cache:
    return cache[query]
```

### 3. 成本監控

```bash
# 使用 Kubecost
helm install kubecost kubecost/cost-analyzer \
  --namespace kubecost --create-namespace

# 訪問儀表板
kubectl port-forward -n kubecost svc/kubecost-cost-analyzer 9090:9090
```

## A/B 測試

### 配置示例

```yaml
# Istio VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llm-service-ab-test
spec:
  hosts:
    - llm-service
  http:
    - match:
        - headers:
            x-version:
              exact: "v2"
      route:
        - destination:
            host: llm-service
            subset: v2
    - route:
        - destination:
            host: llm-service
            subset: v1
          weight: 90
        - destination:
            host: llm-service
            subset: v2
          weight: 10
```

## 安全最佳實踐

1. **網絡隔離**
   - Network Policies
   - Service Mesh (mTLS)

2. **密鑰管理**
   - Kubernetes Secrets
   - HashiCorp Vault
   - AWS Secrets Manager

3. **鏡像安全**
   - 定期掃描漏洞
   - 使用最小化基礎鏡像
   - 簽名驗證

4. **訪問控制**
   - RBAC
   - API Key 管理
   - Rate Limiting

## 災難恢復

### 備份策略

```bash
# 資料庫備份
./scripts/backup.sh --type database --retention 30d

# 配置備份
./scripts/backup.sh --type config --retention 90d

# 完整快照
./scripts/backup.sh --type full --retention 7d
```

### 恢復流程

```bash
# 1. 評估影響範圍
# 2. 通知相關人員
# 3. 執行回滾或恢復
./scripts/rollback.sh --version v1.2.3

# 4. 驗證服務
./scripts/health-check.sh

# 5. 更新文檔
```

## 故障排查

### 常見問題

#### 1. Pod 無法啟動

```bash
# 查看 Pod 狀態
kubectl describe pod <pod-name> -n <namespace>

# 查看日誌
kubectl logs <pod-name> -n <namespace>

# 進入容器
kubectl exec -it <pod-name> -n <namespace> -- /bin/bash
```

#### 2. 服務無響應

```bash
# 檢查服務端點
kubectl get endpoints -n <namespace>

# 測試服務連通性
kubectl run test-pod --image=curlimages/curl -it --rm -- \
  curl http://llm-service:8000/health
```

#### 3. 高延遲

```bash
# 查看資源使用
kubectl top pods -n <namespace>

# 查看網絡延遲
kubectl exec -it <pod-name> -- traceroute <target>
```

## 最佳實踐總結

1. **自動化一切**：從構建到部署全部自動化
2. **監控和告警**：建立完善的監控體系
3. **逐步發布**：使用金絲雀或藍綠部署
4. **快速回滾**：確保能夠快速回退
5. **定期演練**：災難恢復演練
6. **文檔完善**：維護詳細的運維文檔
7. **成本意識**：持續優化資源使用

## 參考資源

- [Kubernetes 官方文檔](https://kubernetes.io/docs/)
- [Prometheus 文檔](https://prometheus.io/docs/)
- [Grafana 文檔](https://grafana.com/docs/)
- [Terraform 文檔](https://www.terraform.io/docs/)
- [CNCF 雲原生最佳實踐](https://www.cncf.io/)

## 授權

MIT License
