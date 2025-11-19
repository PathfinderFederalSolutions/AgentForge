# AgentForge Cloud Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying AgentForge to cloud environments with enterprise-grade security, authentication, and compliance.

## Prerequisites

### Required API Keys
The following API keys must be configured for full functionality:

```bash
# LLM Providers (at least one required)
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export COHERE_API_KEY="your_cohere_api_key"
export MISTRAL_API_KEY="your_mistral_api_key"
export XAI_API_KEY="your_xai_api_key"

# Authentication & Security
export JWT_SECRET_KEY="your_jwt_secret_key_change_in_production"
export AF_ENCRYPTION_KEY="your_encryption_key_for_data"

# Infrastructure (Production)
export DATABASE_URL="postgresql://user:password@db:5432/agentforge"
export REDIS_URL="redis://redis:6379/0"
export NATS_URL="nats://nats:4222"

# Optional: Vector Database
export PINECONE_API_KEY="your_pinecone_api_key"
export PINECONE_INDEX="agentforge-production"
```

### Environment Configuration

```bash
# Deployment Settings
export AF_ENVIRONMENT="production"  # development, staging, production
export AF_EDITION="commercial"      # commercial, fedciv, dod, private

# Security Settings
export AF_REQUIRE_AUTH="true"
export AF_REQUIRE_HTTPS="true"
export AF_FIPS_MODE="false"         # Set to "true" for FedCiv/DoD
export AF_RATE_LIMITING_ENABLED="true"
export AF_RATE_LIMIT_RPM="100"

# Performance Settings
export AF_MAX_AGENTS="1000"
export AF_AGENT_TIMEOUT="600"
export AF_PARALLEL_EXECUTION="true"

# Feature Flags
export AF_FEATURE_NEURAL_MESH_ENABLED="true"
export AF_FEATURE_QUANTUM_SCHEDULER_ENABLED="true"
export AF_FEATURE_UNIVERSAL_IO_ENABLED="true"
export AF_FEATURE_ADVANCED_FUSION_ENABLED="true"
export AF_FEATURE_SELF_BOOTSTRAP_ENABLED="true"
export AF_FEATURE_SECURITY_ORCHESTRATOR_ENABLED="true"
```

## Deployment Editions

### Commercial Edition
- **Target:** Commercial enterprises
- **Features:** Full AGI capabilities, admin dashboard, individual frontend
- **Security:** Standard OAuth2/OIDC, rate limiting
- **Compliance:** SOC 2, GDPR ready

### FedCiv Edition
- **Target:** Federal civilian agencies
- **Features:** Full AGI capabilities with FIPS compliance
- **Security:** Enhanced authentication, stricter rate limits
- **Compliance:** FedRAMP ready, FIPS 140-2

### DoD Edition
- **Target:** Department of Defense
- **Features:** Admin-only interface, enhanced security
- **Security:** FIPS mode, no WebSocket streaming, strict audit
- **Compliance:** CMMC Level 2, IL4/IL5 ready

### Private Edition
- **Target:** Private cloud deployments
- **Features:** Full capabilities with custom security
- **Security:** Configurable authentication, custom domains
- **Compliance:** Custom compliance frameworks

## Authentication Setup

### OAuth2/OIDC Configuration

1. **Create OAuth2 Application** in your identity provider
2. **Configure Redirect URIs:**
   - Commercial: `https://app.agentforge.com/auth/callback`
   - FedCiv: `https://app.agentforge.gov/auth/callback`
   - DoD: `https://app.agentforge.mil/auth/callback`
   - Private: `https://app.agentforge.local/auth/callback`

3. **Set Environment Variables:**
```bash
export OAUTH2_CLIENT_ID="your_client_id"
export OAUTH2_CLIENT_SECRET="your_client_secret"
export OAUTH2_ISSUER_URL="https://your-identity-provider.com"
export OAUTH2_AUDIENCE="agentforge-api"
```

### RBAC Configuration

Default roles and permissions are configured in `core/auth_system.py`:

- **Admin:** Full system access
- **Developer:** Agent deployment, swarm management, fusion access
- **Analyst:** Read access to agents, jobs, metrics, fusion capabilities
- **Operator:** Agent management, job management, metrics access
- **User:** Basic read access to agents, jobs, metrics
- **Viewer:** Read-only access

### Multi-Tenant Setup

1. **Create Default Tenant:**
```python
from core.auth_system import rbac_manager

tenant = rbac_manager.create_tenant("Your Organization", "enterprise")
```

2. **Create Admin User:**
```python
admin_user = rbac_manager.create_user(
    username="admin",
    email="admin@yourorg.com",
    roles=[Role.ADMIN],
    tenant_id=tenant.tenant_id
)
```

## Kubernetes Deployment

### Health Checks Configuration

```yaml
livenessProbe:
  httpGet:
    path: /live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Prometheus Monitoring

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"
  prometheus.io/path: "/metrics"
```

### Environment-Specific Configurations

#### Development
```yaml
env:
  - name: AF_ENVIRONMENT
    value: "development"
  - name: AF_EDITION
    value: "commercial"
  - name: AF_REQUIRE_AUTH
    value: "false"
```

#### Production
```yaml
env:
  - name: AF_ENVIRONMENT
    value: "production"
  - name: AF_EDITION
    value: "commercial"
  - name: AF_REQUIRE_AUTH
    value: "true"
  - name: AF_REQUIRE_HTTPS
    value: "true"
  - name: AF_FIPS_MODE
    value: "false"
```

#### FedCiv/DoD
```yaml
env:
  - name: AF_ENVIRONMENT
    value: "production"
  - name: AF_EDITION
    value: "fedciv"  # or "dod"
  - name: AF_REQUIRE_AUTH
    value: "true"
  - name: AF_REQUIRE_HTTPS
    value: "true"
  - name: AF_FIPS_MODE
    value: "true"
  - name: AF_RATE_LIMIT_RPM
    value: "30"
```

## Security Hardening

### 1. HTTPS/TLS Configuration
- Use TLS 1.2+ for all communications
- Configure proper certificate management
- Enable HSTS headers

### 2. Network Security
- Configure proper firewall rules
- Use private networking for internal communication
- Enable VPC/subnet isolation

### 3. Data Protection
- Enable encryption at rest for databases
- Use encrypted Redis instances
- Configure backup encryption

### 4. Audit Logging
- All API calls are logged with correlation IDs
- Immutable audit logs for compliance
- Structured logging for SIEM integration

## Monitoring and Observability

### Prometheus Metrics
- `agentforge_services_available_total` - Available services count
- `agentforge_llm_providers_total` - LLM providers count
- `agentforge_websocket_connections_total` - WebSocket connections
- `agentforge_agent_executions_total` - Agent executions
- `agentforge_users_total` - Total users
- `agentforge_tenants_total` - Total tenants

### Health Endpoints
- `GET /live` - Kubernetes liveness probe
- `GET /ready` - Kubernetes readiness probe with dependency checks
- `GET /health` - Detailed health information

### Real-time Monitoring
- `GET /v1/chat/stream` - SSE stream for chat updates
- `GET /v1/events/stream` - SSE stream for system events
- `WebSocket /v1/realtime/ws` - WebSocket for admin dashboard

## API Endpoints Summary

### Core APIs (30+ endpoints)
- Chat processing with agent deployment
- Service management and monitoring
- Fusion capabilities (Bayesian, EO/IR, ROC/DET)
- Analytics and performance tracking

### Control Plane APIs (16 new endpoints)
- `/v1/usage` - Usage breakdown by tenant/project/agent
- `/v1/quotas` - Quota management
- `/v1/audit-logs` - Audit log search
- `/v1/webhooks` - Webhook management
- `/v1/auth/token` - OAuth2/OIDC token exchange
- `/v1/auth/me` - User introspection
- `/v1/tenants` - Tenant management
- `/v1/projects` - Project management
- `/v1/models/providers` - Model provider management
- `/v1/models/routes` - Routing policy management

### Kubernetes Probes
- `/live` - Liveness probe
- `/ready` - Readiness probe
- `/metrics` - Prometheus metrics

## Next Steps for Production

### 1. API Key Management
Set up secure API key rotation:
```bash
# Use secrets management (e.g., Kubernetes secrets, AWS Secrets Manager)
kubectl create secret generic agentforge-secrets \
  --from-literal=openai-api-key="your_key" \
  --from-literal=anthropic-api-key="your_key" \
  --from-literal=jwt-secret="your_secret"
```

### 2. Database Migration
Set up production database:
```bash
# PostgreSQL for production
export DATABASE_URL="postgresql://agentforge:secure_password@db.example.com:5432/agentforge_prod"

# Run migrations
python -c "from core.database_manager import get_db_manager; get_db_manager()"
```

### 3. Identity Provider Integration
Configure with your OAuth2/OIDC provider:
- Auth0, Okta, Azure AD, Google Workspace, etc.
- Set up proper scopes and claims mapping
- Configure role-based access control

### 4. Compliance Configuration
For FedCiv/DoD deployments:
- Enable FIPS mode
- Configure data residency
- Set up compliance logging
- Implement data classification

### 5. Monitoring Setup
- Configure Prometheus scraping
- Set up Grafana dashboards
- Configure alerting rules
- Set up log aggregation (ELK, Splunk, etc.)

## Troubleshooting

### Common Issues
1. **Authentication not working:** Check JWT_SECRET_KEY and OAuth2 configuration
2. **Services not available:** Verify all required API keys are set
3. **CORS errors:** Check AF_ENVIRONMENT and AF_EDITION settings
4. **Rate limiting issues:** Adjust AF_RATE_LIMIT_RPM
5. **Database errors:** Verify DATABASE_URL and connection

### Debug Mode
Enable debug logging:
```bash
export AF_LOG_LEVEL="DEBUG"
export AF_DEBUG_MODE="true"
```

### Health Check Validation
```bash
# Test liveness
curl http://localhost:8000/live

# Test readiness
curl http://localhost:8000/ready

# Test metrics
curl http://localhost:8000/metrics
```
