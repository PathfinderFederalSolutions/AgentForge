# AgentForge Authentication Setup Guide

## OAuth2/OIDC Integration Next Steps

### Required API Keys and Secrets

#### 1. JWT Configuration
```bash
# Generate secure JWT secret (256-bit)
export JWT_SECRET_KEY=$(openssl rand -hex 32)

# Or use a specific key for consistency across deployments
export JWT_SECRET_KEY="your_256_bit_secret_key_here"
```

#### 2. OAuth2 Provider Setup

##### Option A: Auth0
```bash
export OAUTH2_PROVIDER="auth0"
export OAUTH2_DOMAIN="your-domain.auth0.com"
export OAUTH2_CLIENT_ID="your_auth0_client_id"
export OAUTH2_CLIENT_SECRET="your_auth0_client_secret"
export OAUTH2_AUDIENCE="agentforge-api"
```

##### Option B: Okta
```bash
export OAUTH2_PROVIDER="okta"
export OAUTH2_DOMAIN="your-domain.okta.com"
export OAUTH2_CLIENT_ID="your_okta_client_id"
export OAUTH2_CLIENT_SECRET="your_okta_client_secret"
export OAUTH2_AUDIENCE="agentforge-api"
```

##### Option C: Azure AD
```bash
export OAUTH2_PROVIDER="azure"
export OAUTH2_TENANT_ID="your_azure_tenant_id"
export OAUTH2_CLIENT_ID="your_azure_client_id"
export OAUTH2_CLIENT_SECRET="your_azure_client_secret"
```

##### Option D: Google Workspace
```bash
export OAUTH2_PROVIDER="google"
export OAUTH2_CLIENT_ID="your_google_client_id"
export OAUTH2_CLIENT_SECRET="your_google_client_secret"
```

#### 3. Database Encryption
```bash
# Database encryption key
export DB_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Redis encryption (if using Redis AUTH)
export REDIS_PASSWORD="your_secure_redis_password"
```

#### 4. API Security
```bash
# API key for internal service communication
export AF_INTERNAL_API_KEY=$(openssl rand -hex 16)

# Webhook signing secret
export WEBHOOK_SIGNING_SECRET=$(openssl rand -hex 32)
```

## RBAC Configuration

### Default Roles Setup

The system comes with predefined roles. Customize as needed:

```python
# In core/auth_system.py, modify role_permissions mapping
CUSTOM_ROLE_PERMISSIONS = {
    Role.ENTERPRISE_ADMIN: {
        # All permissions + custom enterprise permissions
        Permission.ADMIN_SYSTEM,
        Permission.MANAGE_TENANTS,
        Permission.MANAGE_BILLING,
        Permission.ACCESS_AUDIT_LOGS
    },
    Role.SECURITY_OFFICER: {
        Permission.READ_AUDIT,
        Permission.WRITE_AUDIT,
        Permission.MANAGE_SECURITY,
        Permission.READ_AGENTS,
        Permission.READ_JOBS
    },
    Role.DATA_SCIENTIST: {
        Permission.ACCESS_FUSION,
        Permission.READ_AGENTS,
        Permission.WRITE_JOBS,
        Permission.READ_METRICS,
        Permission.DEPLOY_AGENTS
    }
}
```

### Custom Permissions

Add custom permissions for your organization:

```python
class CustomPermission(Enum):
    """Custom permissions for your organization"""
    ACCESS_SENSITIVE_DATA = "access:sensitive_data"
    MANAGE_COMPLIANCE = "manage:compliance"
    EXPORT_MODELS = "export:models"
    MANAGE_INTEGRATIONS = "manage:integrations"
```

## Identity Provider Integration

### 1. Auth0 Integration

```python
# Create auth0_integration.py
import requests
from jose import jwt

class Auth0Integration:
    def __init__(self, domain, client_id, client_secret):
        self.domain = domain
        self.client_id = client_id
        self.client_secret = client_secret
        self.algorithms = ["RS256"]
    
    async def verify_token(self, token: str):
        # Get Auth0 public key
        jwks_url = f"https://{self.domain}/.well-known/jwks.json"
        jwks = requests.get(jwks_url).json()
        
        # Verify token
        payload = jwt.decode(
            token,
            jwks,
            algorithms=self.algorithms,
            audience=self.client_id,
            issuer=f"https://{self.domain}/"
        )
        
        return payload
```

### 2. Okta Integration

```python
# Create okta_integration.py
from okta.client import Client as OktaClient

class OktaIntegration:
    def __init__(self, domain, token):
        self.client = OktaClient({
            'orgUrl': f'https://{domain}',
            'token': token
        })
    
    async def get_user_info(self, user_id: str):
        user = await self.client.get_user(user_id)
        return {
            'user_id': user.id,
            'username': user.profile.login,
            'email': user.profile.email,
            'groups': [group.profile.name for group in await user.get_groups()]
        }
```

### 3. Azure AD Integration

```python
# Create azure_integration.py
from msal import ConfidentialClientApplication

class AzureADIntegration:
    def __init__(self, tenant_id, client_id, client_secret):
        self.app = ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=f"https://login.microsoftonline.com/{tenant_id}"
        )
    
    async def verify_token(self, token: str):
        # Verify Azure AD token
        # Implementation depends on your Azure AD configuration
        pass
```

## Database Setup

### PostgreSQL for Production

```sql
-- Create database and user
CREATE DATABASE agentforge_prod;
CREATE USER agentforge WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE agentforge_prod TO agentforge;

-- Enable required extensions
\c agentforge_prod;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For embeddings if using pgvector
```

### Redis Configuration

```bash
# Redis configuration for production
redis-server --requirepass your_secure_password \
             --maxmemory 2gb \
             --maxmemory-policy allkeys-lru \
             --save 900 1 \
             --save 300 10 \
             --save 60 10000
```

## Secrets Management

### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: agentforge-secrets
type: Opaque
data:
  jwt-secret: <base64-encoded-jwt-secret>
  openai-api-key: <base64-encoded-openai-key>
  anthropic-api-key: <base64-encoded-anthropic-key>
  database-url: <base64-encoded-database-url>
  oauth2-client-secret: <base64-encoded-oauth2-secret>
```

### AWS Secrets Manager

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
    --name "agentforge/prod/api-keys" \
    --description "AgentForge API keys for production" \
    --secret-string '{
        "jwt_secret": "your_jwt_secret",
        "openai_api_key": "your_openai_key",
        "anthropic_api_key": "your_anthropic_key",
        "oauth2_client_secret": "your_oauth2_secret"
    }'
```

## Compliance Configuration

### FIPS Mode Setup

```bash
# Enable FIPS mode for FedCiv/DoD
export AF_FIPS_MODE="true"
export AF_CRYPTO_PROVIDER="fips"

# Use FIPS-compliant LLM providers only
export AF_ALLOWED_PROVIDERS="anthropic,openai"
```

### Audit Logging

```bash
# Enhanced audit logging
export AF_AUDIT_LOGGING_REQUIRED="true"
export AF_AUDIT_LOG_LEVEL="INFO"
export AF_AUDIT_LOG_FORMAT="json"
export AF_AUDIT_LOG_RETENTION_DAYS="2555"  # 7 years for compliance
```

### Data Residency

```bash
# Configure data residency
export AF_DATA_RESIDENCY_REGION="us-east-1"  # or us-gov-east-1 for FedCiv/DoD
export AF_DATA_CLASSIFICATION="sensitive"
export AF_BACKUP_REGION="us-west-2"
```

## Testing Authentication

### 1. Test OAuth2 Flow

```bash
# Get access token
curl -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "grant_type": "password",
    "username": "admin",
    "password": "admin_password"
  }'
```

### 2. Test Protected Endpoints

```bash
# Use access token
curl -X GET http://localhost:8000/v1/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### 3. Test RBAC

```bash
# Test different role permissions
curl -X GET http://localhost:8000/v1/tenants \
  -H "Authorization: Bearer ADMIN_TOKEN"

curl -X GET http://localhost:8000/v1/services/status \
  -H "Authorization: Bearer USER_TOKEN"
```

## Production Checklist

### Security
- [ ] JWT secret key configured (256-bit)
- [ ] OAuth2/OIDC provider configured
- [ ] HTTPS/TLS enabled
- [ ] CORS origins restricted to production domains
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] FIPS mode enabled (if required)

### Authentication
- [ ] Default admin user created
- [ ] Organization tenant created
- [ ] User roles and permissions configured
- [ ] OAuth2 redirect URIs configured
- [ ] Token expiration policies set

### Infrastructure
- [ ] Production database configured
- [ ] Redis instance configured
- [ ] NATS cluster configured (optional)
- [ ] Vector database configured (optional)
- [ ] Backup strategy implemented

### Monitoring
- [ ] Prometheus metrics endpoint working
- [ ] Health probes configured
- [ ] Audit logging enabled
- [ ] Log aggregation configured
- [ ] Alerting rules configured

### Compliance (if required)
- [ ] FIPS mode validated
- [ ] Audit log retention configured
- [ ] Data residency verified
- [ ] Compliance frameworks activated
- [ ] Security controls documented

## Support Contacts

For production deployment support:
- **Security:** Configure OAuth2/OIDC with your identity provider
- **Compliance:** Enable FIPS mode and audit logging for government deployments
- **Performance:** Tune agent limits and timeouts for your workload
- **Integration:** Connect with your existing systems via webhooks and APIs
