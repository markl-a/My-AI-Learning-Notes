# 部署指南

本文檔介紹如何部署 AI 學習專案中的實戰項目到不同環境。

## 📋 目錄

- [環境要求](#環境要求)
- [快速開始](#快速開始)
- [本地開發部署](#本地開發部署)
- [Docker 部署](#docker-部署)
- [Kubernetes 部署](#kubernetes-部署)
- [雲平台部署](#雲平台部署)
- [監控和日誌](#監控和日誌)
- [故障排除](#故障排除)

## 🔧 環境要求

### 基礎環境

- **Python**: >= 3.9
- **Docker**: >= 20.10
- **Docker Compose**: >= 2.0
- **Git**: >= 2.30

### 可選環境

- **Kubernetes**: >= 1.20（用於 K8s 部署）
- **kubectl**: 最新版本
- **Helm**: >= 3.0

### API 密鑰

- **OpenAI API Key**: 用於 LLM 功能
- **其他 API**: 根據具體項目需求

## 🚀 快速開始

### 1. 使用自動化腳本

```bash
# 克隆項目
git clone https://github.com/your-repo/My-AI-Learning-Notes.git
cd My-AI-Learning-Notes

# 運行設置腳本
chmod +x scripts/setup.sh
./scripts/setup.sh

# 配置 API 密鑰
# 編輯 .env 文件並添加你的 OpenAI API 密鑰

# 運行部署腳本
chmod +x scripts/deploy.sh
./scripts/deploy.sh dev all
```

### 2. 手動設置

```bash
# 創建虛擬環境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安裝依賴
pip install -r requirements.txt

# 配置環境變量
cp .env.example .env
# 編輯 .env 文件

# 運行項目
cd 5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot
python main.py
```

## 💻 本地開發部署

### RAG ChatBot

```bash
cd 5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot

# 配置環境
cp .env.example .env
# 編輯 .env 並添加 API 密鑰

# 安裝依賴
pip install -r requirements.txt

# 運行開發服務器
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 訪問 API 文檔
# http://localhost:8000/docs
```

### AI Document Analyzer

```bash
cd 5.AI研究前沿_2024-2025/實戰項目/AI-Document-Analyzer

# 配置環境
cp .env.example .env

# 運行服務
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# 訪問 API 文檔
# http://localhost:8001/docs
```

## 🐳 Docker 部署

### 使用 Docker Compose（推薦）

```bash
# RAG ChatBot
cd 5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot

# 配置環境變量
cp .env.example .env
# 編輯 .env

# 啟動所有服務
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down

# 完全清理（包括卷）
docker-compose down -v
```

### 使用獨立 Docker 容器

```bash
# 構建鏡像
docker build -t rag-chatbot:latest .

# 運行容器
docker run -d \
  --name rag-chatbot \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  rag-chatbot:latest

# 查看日誌
docker logs -f rag-chatbot

# 停止和刪除
docker stop rag-chatbot
docker rm rag-chatbot
```

### 多項目部署

```bash
# 使用部署腳本
./scripts/deploy.sh dev all

# 或手動部署
cd 5.AI研究前沿_2024-2025/實戰項目/RAG-ChatBot
docker-compose up -d

cd ../AI-Document-Analyzer
docker-compose up -d
```

## ☸️ Kubernetes 部署

### 準備配置文件

創建 `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-chatbot
  template:
    metadata:
      labels:
        app: rag-chatbot
    spec:
      containers:
      - name: rag-chatbot
        image: your-registry/rag-chatbot:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-chatbot-service
spec:
  selector:
    app: rag-chatbot
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 部署到 Kubernetes

```bash
# 創建 Secret
kubectl create secret generic openai-secret \
  --from-literal=api-key=your_openai_api_key

# 應用配置
kubectl apply -f k8s/deployment.yaml

# 查看狀態
kubectl get pods
kubectl get services

# 查看日誌
kubectl logs -f deployment/rag-chatbot

# 擴展副本
kubectl scale deployment/rag-chatbot --replicas=5

# 更新鏡像
kubectl set image deployment/rag-chatbot \
  rag-chatbot=your-registry/rag-chatbot:v2.0

# 回滾
kubectl rollout undo deployment/rag-chatbot
```

## ☁️ 雲平台部署

### AWS 部署（ECS + Fargate）

```bash
# 1. 推送鏡像到 ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  your-account.dkr.ecr.us-east-1.amazonaws.com

docker tag rag-chatbot:latest \
  your-account.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest

docker push your-account.dkr.ecr.us-east-1.amazonaws.com/rag-chatbot:latest

# 2. 創建 ECS 任務定義和服務
# 使用 AWS 控制台或 Terraform
```

### Google Cloud Platform (Cloud Run)

```bash
# 1. 推送到 Container Registry
gcloud builds submit --tag gcr.io/your-project/rag-chatbot

# 2. 部署到 Cloud Run
gcloud run deploy rag-chatbot \
  --image gcr.io/your-project/rag-chatbot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key
```

### Azure (Container Instances)

```bash
# 1. 推送到 Azure Container Registry
az acr build --registry your-registry \
  --image rag-chatbot:latest .

# 2. 部署到 Container Instances
az container create \
  --resource-group your-resource-group \
  --name rag-chatbot \
  --image your-registry.azurecr.io/rag-chatbot:latest \
  --dns-name-label rag-chatbot \
  --ports 8000 \
  --environment-variables OPENAI_API_KEY=your_key
```

## 📊 監控和日誌

### Prometheus + Grafana

```bash
# 使用 Docker Compose 啟動監控棧
cd monitoring
docker-compose up -d

# 訪問服務
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### 配置告警

編輯 `monitoring/alerting-rules.yml` 並重新加載：

```bash
# 重新加載 Prometheus 配置
curl -X POST http://localhost:9090/-/reload
```

### 查看日誌

```bash
# Docker 日誌
docker-compose logs -f rag-api

# Kubernetes 日誌
kubectl logs -f deployment/rag-chatbot

# 使用 Loki（如果配置）
# 在 Grafana 中查看
```

## 🔧 故障排除

### 常見問題

#### 1. 容器無法啟動

```bash
# 檢查日誌
docker-compose logs rag-api

# 檢查配置
docker-compose config

# 驗證環境變量
docker-compose exec rag-api env | grep OPENAI
```

#### 2. 無法連接到 API

```bash
# 檢查端口是否正確映射
docker-compose ps

# 檢查防火牆設置
sudo ufw status

# 測試連接
curl http://localhost:8000/api/health
```

#### 3. 內存不足

```bash
# 增加 Docker 內存限制
# 編輯 docker-compose.yml
services:
  rag-api:
    mem_limit: 4g

# 或在 K8s 中調整資源限制
```

#### 4. OpenAI API 錯誤

```bash
# 驗證 API 密鑰
echo $OPENAI_API_KEY

# 測試 API 連接
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### 健康檢查失敗

```bash
# 手動測試健康檢查端點
curl -v http://localhost:8000/api/health

# 查看詳細錯誤
docker-compose logs --tail=100 rag-api
```

### 性能問題

```bash
# 檢查資源使用
docker stats

# Kubernetes 資源使用
kubectl top pods

# 查看慢查詢
# 分析 Prometheus 指標
```

## 🔄 回滾策略

### Docker Compose

```bash
# 標記當前版本
docker tag rag-chatbot:latest rag-chatbot:backup

# 回滾到之前的版本
docker-compose down
docker-compose up -d rag-chatbot:backup
```

### Kubernetes

```bash
# 查看部署歷史
kubectl rollout history deployment/rag-chatbot

# 回滾到上一個版本
kubectl rollout undo deployment/rag-chatbot

# 回滾到特定版本
kubectl rollout undo deployment/rag-chatbot --to-revision=2
```

### 使用部署腳本回滾

```bash
# 部署腳本會自動創建備份
# 如果健康檢查失敗，會提示是否回滾

./scripts/deploy.sh production rag-chatbot
# 如果失敗，選擇 yes 進行回滾
```

## 📈 擴展和優化

### 水平擴展

```bash
# Docker Compose（使用多個副本）
docker-compose up -d --scale rag-api=3

# Kubernetes
kubectl scale deployment/rag-chatbot --replicas=5

# 自動擴展（K8s HPA）
kubectl autoscale deployment rag-chatbot \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

### 性能優化

1. **啟用緩存**
   ```env
   ENABLE_CACHE=true
   CACHE_TTL=3600
   ```

2. **使用 CDN**（用於靜態資源）

3. **數據庫優化**
   - 使用持久化卷
   - 定期備份
   - 優化查詢

4. **負載均衡**
   - 使用 Nginx 或雲負載均衡器
   - 配置健康檢查

## 🔒 安全最佳實踐

1. **永遠不要在代碼中硬編碼密鑰**
2. **使用 Secret 管理服務**（AWS Secrets Manager、Vault等）
3. **啟用 HTTPS**
4. **限制 CORS**
5. **實施速率限制**
6. **定期更新依賴**
7. **掃描容器鏡像漏洞**

```bash
# 使用 Trivy 掃描鏡像
trivy image rag-chatbot:latest
```

## 📚 參考資源

- [Docker 文檔](https://docs.docker.com/)
- [Kubernetes 文檔](https://kubernetes.io/docs/)
- [FastAPI 部署](https://fastapi.tiangolo.com/deployment/)
- [Prometheus 文檔](https://prometheus.io/docs/)
