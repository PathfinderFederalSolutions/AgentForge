# AgentForge Deployment Guide

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Development Environment Setup](#development-environment-setup)
4. [Production Kubernetes Deployment](#production-kubernetes-deployment)
5. [Infrastructure Setup](#infrastructure-setup)
6. [Service Configuration](#service-configuration)
7. [Security Configuration](#security-configuration)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Scaling and Performance](#scaling-and-performance)
10. [Troubleshooting](#troubleshooting)

## Deployment Overview

AgentForge supports multiple deployment patterns:

- **Development**: Local Docker Compose for development and testing
- **Staging**: Kubernetes cluster with reduced resource allocation
- **Production**: Full Kubernetes deployment with high availability
- **Edge**: Lightweight deployment for edge computing scenarios

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer / CDN                      │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Kubernetes Ingress                        │
│               (NGINX/Traefik/Istio)                        │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     Service Mesh                            │
│  API Gateway │ Core Services │ AI Services │ Support       │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Data & Message Layer                      │
│  PostgreSQL │ Redis │ NATS │ Pinecone │ Monitoring        │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements (Development):**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

**Recommended Requirements (Production):**
- CPU: 16+ cores per node
- RAM: 64+ GB per node
- Storage: 500+ GB NVMe SSD
- Network: 10+ Gbps

### Software Dependencies

**Required:**
- Docker 24.0+
- Kubernetes 1.28+
- Helm 3.12+
- kubectl 1.28+
- Python 3.13+

**Optional:**
- Terraform (for infrastructure as code)
- ArgoCD (for GitOps deployment)
- Istio (for advanced service mesh)

### External Services

**Required:**
- NATS JetStream cluster
- PostgreSQL 15+ database
- Redis 7+ cluster

**Optional:**
- Pinecone vector database
- External LLM providers (OpenAI, Anthropic, etc.)
- Monitoring stack (Prometheus, Grafana, Jaeger)

## Development Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/agentforge/agentforge.git
cd agentforge
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
vim .env
```

**Key Environment Variables:**
```bash
# Core Configuration
AF_ENVIRONMENT=development
AF_AGENTS=2
AF_FORCE_MOCK=0
AF_SKIP_HEALTHCHECK=1

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/agentforge
REDIS_URL=redis://localhost:6379/0

# External Services
NATS_URL=nats://localhost:4222
PINECONE_API_KEY=your-pinecone-key
OPENAI_API_KEY=your-openai-key

# Security
JWT_SECRET_KEY=your-jwt-secret
ENCRYPTION_KEY=your-encryption-key
```

### 3. Docker Compose Setup

```bash
# Start infrastructure services
docker-compose -f docker-compose.infrastructure.yml up -d

# Start AgentForge services
docker-compose up -d

# Verify deployment
docker-compose ps
curl http://localhost:8000/health
```

**Docker Compose Configuration:**
```yaml
# docker-compose.yml
version: '3.8'
services:
  agentforge-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AF_ENVIRONMENT=development
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agentforge
      - REDIS_URL=redis://redis:6379/0
      - NATS_URL=nats://nats:4222
    depends_on:
      - postgres
      - redis
      - nats

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: agentforge
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  nats:
    image: nats:2.10-alpine
    command: ["-js", "-m", "8222"]
    ports:
      - "4222:4222"
      - "8222:8222"

volumes:
  postgres_data:
  redis_data:
```

### 4. Local Development

```bash
# Activate virtual environment
source source/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements.dev.txt

# Run database migrations
python scripts/migrate.py

# Start development server
python main.py

# Run tests
pytest -v
```

## Production Kubernetes Deployment

### 1. Cluster Preparation

**Create Kubernetes Cluster:**
```bash
# Using kind (for testing)
kind create cluster --name agentforge --config deployment/k8s/kind-config.yaml

# Using cloud providers
# AWS EKS
eksctl create cluster --name agentforge --region us-west-2 --nodes 3 --node-type m5.2xlarge

# Google GKE
gcloud container clusters create agentforge --num-nodes=3 --machine-type=n1-standard-4

# Azure AKS
az aks create --resource-group agentforge --name agentforge --node-count 3 --node-vm-size Standard_D4s_v3
```

**Install Required Components:**
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for TLS
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Install KEDA for auto-scaling
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.1/keda-2.12.1.yaml
```

### 2. Namespace and RBAC Setup

```bash
# Create namespace
kubectl create namespace agentforge

# Apply RBAC configurations
kubectl apply -f deployment/k8s/rbac/
```

**Service Account Configuration:**
```yaml
# deployment/k8s/rbac/service-account.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agentforge
  namespace: agentforge
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: agentforge
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: agentforge
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: agentforge
subjects:
- kind: ServiceAccount
  name: agentforge
  namespace: agentforge
```

### 3. Configuration Management

**ConfigMap for Application Configuration:**
```yaml
# deployment/k8s/config/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agentforge-config
  namespace: agentforge
data:
  AF_ENVIRONMENT: "production"
  AF_AGENTS: "10"
  AF_SKIP_HEALTHCHECK: "0"
  LOG_LEVEL: "INFO"
  METRICS_ENABLED: "true"
  TRACING_ENABLED: "true"
```

**Secret for Sensitive Configuration:**
```bash
# Create secrets
kubectl create secret generic agentforge-secrets \
  --from-literal=database-url="postgresql://user:pass@postgres:5432/agentforge" \
  --from-literal=redis-url="redis://redis:6379/0" \
  --from-literal=openai-api-key="your-openai-key" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --namespace=agentforge
```

### 4. Infrastructure Services Deployment

**PostgreSQL Deployment:**
```yaml
# deployment/k8s/infrastructure/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: agentforge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: agentforge
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agentforge-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-data
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: agentforge
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

**NATS JetStream Deployment:**
```yaml
# deployment/k8s/infrastructure/nats.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nats
  namespace: agentforge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nats
  template:
    metadata:
      labels:
        app: nats
    spec:
      containers:
      - name: nats
        image: nats:2.10-alpine
        args:
        - "--config"
        - "/etc/nats/nats.conf"
        - "--js"
        - "--cluster_name"
        - "agentforge"
        ports:
        - containerPort: 4222
          name: client
        - containerPort: 6222
          name: cluster
        - containerPort: 8222
          name: monitor
        volumeMounts:
        - name: nats-config
          mountPath: /etc/nats
        - name: nats-data
          mountPath: /data
      volumes:
      - name: nats-config
        configMap:
          name: nats-config
      - name: nats-data
        persistentVolumeClaim:
          claimName: nats-pvc
```

### 5. Core Services Deployment

**API Gateway Deployment:**
```yaml
# deployment/k8s/services/api-gateway.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentforge-api
  namespace: agentforge
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentforge-api
  template:
    metadata:
      labels:
        app: agentforge-api
    spec:
      serviceAccountName: agentforge
      containers:
      - name: api
        image: agentforge/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: AF_ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: agentforge-config
              key: AF_ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agentforge-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

**Swarm Service Deployment:**
```yaml
# deployment/k8s/services/swarm-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swarm-service
  namespace: agentforge
spec:
  replicas: 5
  selector:
    matchLabels:
      app: swarm-service
  template:
    metadata:
      labels:
        app: swarm-service
    spec:
      containers:
      - name: swarm
        image: agentforge/swarm:latest
        env:
        - name: NATS_URL
          value: "nats://nats:4222"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: agentforge-secrets
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 6. Deploy All Services

```bash
# Deploy infrastructure
kubectl apply -f deployment/k8s/infrastructure/

# Deploy core services
kubectl apply -f deployment/k8s/services/

# Deploy ingress and networking
kubectl apply -f deployment/k8s/networking/

# Deploy monitoring (optional)
kubectl apply -f deployment/k8s/monitoring/
```

## Infrastructure Setup

### 1. Using Terraform (Infrastructure as Code)

**Main Terraform Configuration:**
```hcl
# deployment/terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  node_groups = {
    main = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3
      
      instance_types = ["m5.2xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        Application = "agentforge"
      }
    }
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier = "${var.cluster_name}-postgres"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r5.xlarge"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "agentforge"
  username = "postgres"
  password = random_password.db_password.result
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  skip_final_snapshot = var.environment != "production"
  
  tags = {
    Name        = "${var.cluster_name}-postgres"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "Redis cluster for AgentForge"
  
  port               = 6379
  parameter_group_name = "default.redis7"
  node_type          = "cache.r6g.xlarge"
  num_cache_clusters = 3
  
  subnet_group_name = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "${var.cluster_name}-redis"
    Environment = var.environment
  }
}
```

**Deploy Infrastructure:**
```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="environments/production.tfvars"

# Apply infrastructure
terraform apply -var-file="environments/production.tfvars"

# Get kubeconfig
aws eks update-kubeconfig --region us-west-2 --name agentforge-prod
```

### 2. Using Helm Charts

**Helm Chart Structure:**
```
deployment/helm/agentforge/
├── Chart.yaml
├── values.yaml
├── values-production.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   └── secrets.yaml
└── charts/
    ├── postgresql/
    ├── redis/
    └── nats/
```

**Deploy with Helm:**
```bash
# Add required helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add nats https://nats-io.github.io/k8s/helm/charts/
helm repo update

# Install dependencies
helm dependency update deployment/helm/agentforge/

# Deploy to production
helm install agentforge deployment/helm/agentforge/ \
  --namespace agentforge \
  --create-namespace \
  --values deployment/helm/agentforge/values-production.yaml

# Verify deployment
helm status agentforge -n agentforge
```

## Service Configuration

### 1. Environment-Specific Configuration

**Development Configuration:**
```yaml
# config/environments/development.yaml
environment: development
debug: true
log_level: DEBUG

database:
  url: postgresql://postgres:password@localhost:5432/agentforge_dev
  pool_size: 5
  max_overflow: 10

redis:
  url: redis://localhost:6379/0
  max_connections: 10

nats:
  url: nats://localhost:4222
  max_reconnects: 5

agents:
  default_count: 2
  max_count: 10
  timeout: 300

security:
  jwt_expiry: 3600
  encryption_enabled: false
  audit_logging: false
```

**Production Configuration:**
```yaml
# config/environments/production.yaml
environment: production
debug: false
log_level: INFO

database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 50
  ssl_require: true

redis:
  url: ${REDIS_URL}
  max_connections: 100
  ssl_cert_reqs: required

nats:
  url: ${NATS_URL}
  max_reconnects: -1
  tls_required: true

agents:
  default_count: 10
  max_count: 1000
  timeout: 600

security:
  jwt_expiry: 1800
  encryption_enabled: true
  audit_logging: true
  hsm_enabled: true

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  profiling_enabled: false
```

### 2. Service-Specific Configuration

**Swarm Service Configuration:**
```yaml
# services/swarm/config/production.yaml
swarm:
  coordination:
    algorithm: quantum_inspired
    max_agents_per_node: 10000
    heartbeat_interval: 30
    
  memory:
    neural_mesh_enabled: true
    l3_l4_memory_enabled: true
    vector_db_url: ${PINECONE_API_KEY}
    
  scaling:
    auto_scaling_enabled: true
    min_replicas: 3
    max_replicas: 100
    target_cpu_utilization: 70
    
  security:
    encryption_at_rest: true
    encryption_in_transit: true
    audit_all_operations: true
```

**Universal I/O Configuration:**
```yaml
# services/universal-io/config/production.yaml
universal_io:
  processing:
    max_concurrent_streams: 10000
    stream_buffer_size: 1000000
    processing_timeout: 300
    
  input_adapters:
    document:
      max_file_size: 100MB
      supported_formats: [pdf, docx, txt, md]
    media:
      max_file_size: 1GB
      supported_formats: [jpg, png, mp4, wav]
    sensor:
      max_data_rate: 1000000  # events per second
      
  output_generators:
    application:
      frameworks: [react, vue, angular, flutter]
    visualization:
      libraries: [d3, plotly, chartjs, matplotlib]
      
  security:
    input_sanitization: strict
    output_validation: strict
    virus_scanning: enabled
```

## Security Configuration

### 1. TLS/SSL Configuration

**Certificate Management with cert-manager:**
```yaml
# deployment/k8s/security/certificate.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@agentforge.ai
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: agentforge-tls
  namespace: agentforge
spec:
  secretName: agentforge-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.agentforge.ai
  - admin.agentforge.ai
```

### 2. Network Policies

**Network Security Configuration:**
```yaml
# deployment/k8s/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agentforge-network-policy
  namespace: agentforge
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  - from:
    - podSelector:
        matchLabels:
          app: agentforge-api
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: nats
    ports:
    - protocol: TCP
      port: 4222
```

### 3. Security Scanning and Compliance

**Security Scanning Pipeline:**
```bash
# scripts/security-scan.sh
#!/bin/bash

# Container image scanning with Trivy
trivy image agentforge/api:latest --severity HIGH,CRITICAL

# Kubernetes configuration scanning
trivy config deployment/k8s/ --severity HIGH,CRITICAL

# Infrastructure scanning
trivy fs . --severity HIGH,CRITICAL

# OWASP ZAP security testing
docker run -t owasp/zap2docker-stable zap-baseline.py -t https://api.agentforge.ai
```

## Monitoring and Observability

### 1. Prometheus and Grafana Setup

**Prometheus Configuration:**
```yaml
# deployment/k8s/monitoring/prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - /etc/prometheus/rules/*.yml
    
    scrape_configs:
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
    
    - job_name: 'agentforge-api'
      static_configs:
      - targets: ['agentforge-api:8000']
      metrics_path: /metrics
      scrape_interval: 5s
    
    - job_name: 'swarm-service'
      static_configs:
      - targets: ['swarm-service:8001']
      metrics_path: /metrics
      scrape_interval: 10s
```

**Grafana Dashboard Configuration:**
```json
{
  "dashboard": {
    "title": "AgentForge System Overview",
    "panels": [
      {
        "title": "Agent Deployment Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agents_deployed_total[5m])",
            "legendFormat": "Agents/sec"
          }
        ]
      },
      {
        "title": "Neural Mesh Operations",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(neural_mesh_operations_total[5m])",
            "legendFormat": "Operations/sec"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### 2. Distributed Tracing with Jaeger

**Jaeger Deployment:**
```yaml
# deployment/k8s/monitoring/jaeger.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.50
        env:
        - name: COLLECTOR_OTLP_ENABLED
          value: "true"
        ports:
        - containerPort: 16686
          name: ui
        - containerPort: 14250
          name: grpc
        - containerPort: 4317
          name: otlp-grpc
        - containerPort: 4318
          name: otlp-http
```

### 3. Logging with ELK Stack

**Elasticsearch and Kibana:**
```bash
# Deploy ELK stack
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/crds.yaml
kubectl apply -f https://download.elastic.co/downloads/eck/2.9.0/operator.yaml

# Deploy Elasticsearch cluster
kubectl apply -f deployment/k8s/monitoring/elasticsearch.yaml

# Deploy Kibana
kubectl apply -f deployment/k8s/monitoring/kibana.yaml

# Deploy Filebeat for log collection
kubectl apply -f deployment/k8s/monitoring/filebeat.yaml
```

## Scaling and Performance

### 1. Horizontal Pod Autoscaling (HPA)

**HPA Configuration:**
```yaml
# deployment/k8s/scaling/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentforge-api-hpa
  namespace: agentforge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentforge-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

### 2. KEDA Scaling for Event-Driven Workloads

**KEDA Scaler for NATS JetStream:**
```yaml
# deployment/k8s/scaling/keda-nats-scaler.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: swarm-worker-scaler
  namespace: agentforge
spec:
  scaleTargetRef:
    name: swarm-worker
  minReplicaCount: 2
  maxReplicaCount: 100
  triggers:
  - type: nats-jetstream
    metadata:
      natsServerMonitoringEndpoint: "nats:8222"
      stream: "TOOLS"
      consumer: "tool-executor"
      lagThreshold: "50"
```

### 3. Vertical Pod Autoscaling (VPA)

**VPA Configuration:**
```yaml
# deployment/k8s/scaling/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: agentforge-api-vpa
  namespace: agentforge
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentforge-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

## Troubleshooting

### 1. Common Issues and Solutions

**Pod Startup Issues:**
```bash
# Check pod status
kubectl get pods -n agentforge

# Describe problematic pod
kubectl describe pod <pod-name> -n agentforge

# Check pod logs
kubectl logs <pod-name> -n agentforge --previous

# Check resource constraints
kubectl top pods -n agentforge
```

**Database Connection Issues:**
```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres -U postgres -d agentforge

# Check database pod logs
kubectl logs -l app=postgres -n agentforge

# Verify database secrets
kubectl get secret agentforge-secrets -n agentforge -o yaml
```

**NATS JetStream Issues:**
```bash
# Check NATS cluster status
kubectl exec -it nats-0 -n agentforge -- nats server check jetstream

# Monitor stream status
kubectl exec -it nats-0 -n agentforge -- nats stream ls

# Check consumer lag
kubectl exec -it nats-0 -n agentforge -- nats consumer info TOOLS tool-executor
```

### 2. Performance Debugging

**Resource Usage Analysis:**
```bash
# Check cluster resource usage
kubectl top nodes

# Check namespace resource usage
kubectl top pods -n agentforge --sort-by=cpu
kubectl top pods -n agentforge --sort-by=memory

# Analyze resource requests vs limits
kubectl describe nodes | grep -A 5 "Allocated resources"
```

**Network Debugging:**
```bash
# Test service connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- \
  curl http://agentforge-api:8000/health

# Check DNS resolution
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- \
  nslookup agentforge-api.agentforge.svc.cluster.local

# Analyze network policies
kubectl get networkpolicies -n agentforge
```

### 3. Disaster Recovery

**Backup Procedures:**
```bash
# Database backup
kubectl exec postgres-0 -n agentforge -- pg_dump -U postgres agentforge > backup.sql

# Redis backup
kubectl exec redis-0 -n agentforge -- redis-cli BGSAVE
kubectl cp redis-0:/data/dump.rdb ./redis-backup.rdb -n agentforge

# Configuration backup
kubectl get all,configmaps,secrets -n agentforge -o yaml > agentforge-backup.yaml
```

**Restore Procedures:**
```bash
# Database restore
kubectl exec -i postgres-0 -n agentforge -- psql -U postgres agentforge < backup.sql

# Redis restore
kubectl cp ./redis-backup.rdb redis-0:/data/dump.rdb -n agentforge
kubectl exec redis-0 -n agentforge -- redis-cli DEBUG RESTART

# Configuration restore
kubectl apply -f agentforge-backup.yaml
```

This comprehensive deployment guide provides step-by-step instructions for deploying AgentForge in various environments, from development to production. The guide includes infrastructure setup, security configuration, monitoring, scaling, and troubleshooting procedures to ensure successful deployment and operation of the AgentForge platform.
