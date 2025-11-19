# AgentForge Deployment Editions Guide

## Overview

AgentForge supports four deployment editions, each tailored for specific customer segments with appropriate security, compliance, and feature configurations.

## Edition Comparison

| Feature | Commercial | FedCiv | DoD | Private |
|---------|------------|--------|-----|---------|
| **Target Audience** | Enterprises | Federal Civilian | Department of Defense | Private Cloud |
| **Security Level** | Standard | High | Maximum | Configurable |
| **FIPS Compliance** | Optional | Required | Required | Optional |
| **Authentication** | OAuth2/OIDC | CAC/PIV + OAuth2 | CAC/PIV Required | Custom |
| **Data Residency** | Global | US Only | US Gov Cloud | Customer Choice |
| **Audit Logging** | Standard | Enhanced | Maximum | Configurable |
| **Rate Limiting** | 100 req/min | 50 req/min | 30 req/min | Configurable |
| **WebSocket Streaming** | Enabled | Enabled | Disabled | Configurable |
| **Individual Frontend** | Enabled | Enabled | Admin Only | Configurable |

---

## Commercial Edition Deployment

### Target Customers
- Commercial enterprises
- SaaS platforms
- Technology companies
- Consulting firms

### Required Infrastructure

#### **Database (REQUIRED)**
```bash
# PostgreSQL 13+ for production
DATABASE_URL=postgresql://agentforge:secure_password@db.company.com:5432/agentforge_prod

# Alternative: Managed database services
# AWS RDS: postgresql://user:pass@agentforge.xxxxx.us-east-1.rds.amazonaws.com:5432/agentforge
# Google Cloud SQL: postgresql://user:pass@google-cloud-sql-ip:5432/agentforge
# Azure Database: postgresql://user:pass@agentforge.postgres.database.azure.com:5432/agentforge
```

#### **Redis (REQUIRED)**
```bash
# Redis 6+ for session management and caching
REDIS_URL=redis://redis.company.com:6379/0
REDIS_PASSWORD=secure_redis_password

# Alternative: Managed Redis services
# AWS ElastiCache: redis://agentforge.xxxxx.cache.amazonaws.com:6379/0
# Google Memorystore: redis://memorystore-ip:6379/0
# Azure Cache: redis://agentforge.redis.cache.windows.net:6380/0
```

### Environment Configuration
```bash
# Commercial Edition Settings
AF_ENVIRONMENT=production
AF_EDITION=commercial
AF_REQUIRE_AUTH=true
AF_REQUIRE_HTTPS=true
AF_FIPS_MODE=false
AF_RATE_LIMIT_RPM=100
AF_DATA_RESIDENCY_REGION=us-east-1
AF_COMPLIANCE_FRAMEWORK=soc2
```

### Deployment Steps
1. **Infrastructure Setup**
   - Deploy PostgreSQL database
   - Deploy Redis instance
   - Configure load balancer with SSL termination
   - Set up monitoring (Prometheus/Grafana)

2. **Security Configuration**
   - Configure OAuth2 provider (Auth0, Okta, etc.)
   - Generate JWT secrets
   - Set up HTTPS certificates
   - Configure CORS origins

3. **Application Deployment**
   - Deploy backend API (port 8000)
   - Deploy admin dashboard (port 3001)
   - Deploy individual frontend (port 3002)
   - Configure health checks

4. **Monitoring Setup**
   - Configure Prometheus scraping
   - Set up Grafana dashboards
   - Configure log aggregation
   - Set up alerting

---

## FedCiv Edition Deployment

### Target Customers
- Federal civilian agencies
- State and local government
- Government contractors
- Regulated industries

### Required Infrastructure

#### **Database (REQUIRED - US Gov Cloud)**
```bash
# PostgreSQL in US Government regions
DATABASE_URL=postgresql://agentforge:secure_password@db.us-gov-east-1.amazonaws.com:5432/agentforge_fedciv

# Must be in government-approved regions:
# - us-gov-east-1 (AWS GovCloud)
# - us-gov-west-1 (AWS GovCloud)
# - Government-approved Azure regions
```

#### **Redis (REQUIRED - US Gov Cloud)**
```bash
# Redis in US Government regions
REDIS_URL=redis://redis.us-gov-east-1.amazonaws.com:6379/0
REDIS_PASSWORD=fips_compliant_password

# Must use FIPS-compliant encryption
```

### Environment Configuration
```bash
# FedCiv Edition Settings
AF_ENVIRONMENT=production
AF_EDITION=fedciv
AF_REQUIRE_AUTH=true
AF_REQUIRE_HTTPS=true
AF_FIPS_MODE=true
AF_RATE_LIMIT_RPM=50
AF_DATA_RESIDENCY_REGION=us-gov-east-1
AF_COMPLIANCE_FRAMEWORK=fedramp
AF_AUDIT_LOG_RETENTION_DAYS=2555  # 7 years
```

### Compliance Requirements
1. **FedRAMP Authorization**
   - Complete FedRAMP assessment
   - Implement required security controls
   - Continuous monitoring

2. **FIPS 140-2 Compliance**
   - Enable FIPS mode for cryptographic operations
   - Use FIPS-validated encryption modules
   - FIPS-compliant LLM providers only

3. **Data Residency**
   - All data must remain in US government regions
   - No cross-border data transfer
   - Government-approved cloud providers only

### Deployment Steps
1. **Pre-deployment**
   - Obtain FedRAMP authorization
   - Complete security assessment
   - Set up government cloud accounts

2. **Infrastructure**
   - Deploy in US Gov Cloud regions
   - Configure FIPS-compliant encryption
   - Set up government-approved monitoring

3. **Security**
   - Integrate with government identity providers
   - Configure CAC/PIV authentication
   - Enable enhanced audit logging

4. **Compliance**
   - Implement continuous monitoring
   - Set up compliance reporting
   - Configure data loss prevention

---

## DoD Edition Deployment

### Target Customers
- Department of Defense
- Defense contractors
- Military branches
- Intelligence agencies

### Required Infrastructure

#### **Database (REQUIRED - IL4/IL5)**
```bash
# PostgreSQL in DoD-approved environments
DATABASE_URL=postgresql://agentforge:cmmc_compliant_password@db.disa.mil:5432/agentforge_dod

# Must meet IL4/IL5 requirements:
# - DISA-approved cloud environments
# - Enhanced encryption at rest and in transit
# - Continuous monitoring
```

#### **Redis (REQUIRED - IL4/IL5)**
```bash
# Redis with DoD-level security
REDIS_URL=redis://redis.disa.mil:6379/0
REDIS_PASSWORD=dod_secure_password

# Must use DoD-approved encryption standards
```

### Environment Configuration
```bash
# DoD Edition Settings
AF_ENVIRONMENT=production
AF_EDITION=dod
AF_REQUIRE_AUTH=true
AF_REQUIRE_HTTPS=true
AF_FIPS_MODE=true
AF_RATE_LIMIT_RPM=30
AF_DATA_RESIDENCY_REGION=us-gov-east-1
AF_COMPLIANCE_FRAMEWORK=cmmc
AF_FEATURE_WEBSOCKET_SUPPORT_ENABLED=false  # Disabled for security
AF_FEATURE_INDIVIDUAL_FRONTEND_ENABLED=false  # Admin only
```

### Compliance Requirements
1. **CMMC Level 2+ Certification**
   - Implement all CMMC controls
   - Continuous compliance monitoring
   - Third-party assessment

2. **IL4/IL5 Data Handling**
   - Controlled Unclassified Information (CUI) protection
   - Enhanced access controls
   - Audit trail for all data access

3. **DoD-Specific Security**
   - CAC/PIV authentication required
   - Network isolation and segmentation
   - Incident response procedures

### Deployment Steps
1. **Pre-deployment**
   - Obtain CMMC certification
   - Complete DoD security assessment
   - Set up DISA-approved infrastructure

2. **Infrastructure**
   - Deploy in DoD-approved cloud (AWS GovCloud, Azure Government)
   - Configure IL4/IL5 security controls
   - Implement network segmentation

3. **Security**
   - Integrate with DoD identity systems
   - Configure CAC/PIV authentication
   - Enable maximum audit logging

4. **Operations**
   - Implement DoD change management
   - Set up incident response
   - Configure backup and recovery

---

## Private Edition Deployment

### Target Customers
- Large enterprises with private clouds
- Financial institutions
- Healthcare organizations
- Custom deployments

### Required Infrastructure

#### **Database (REQUIRED - Customer Choice)**
```bash
# Customer's preferred database
DATABASE_URL=postgresql://agentforge:password@customer-db:5432/agentforge

# Supported databases:
# - PostgreSQL 13+
# - MySQL 8+
# - SQL Server 2019+
# - Oracle 19c+
```

#### **Redis (REQUIRED - Customer Choice)**
```bash
# Customer's Redis instance
REDIS_URL=redis://customer-redis:6379/0

# Alternative: In-memory caching
# CACHE_BACKEND=memory  # For single-instance deployments
```

### Environment Configuration
```bash
# Private Edition Settings
AF_ENVIRONMENT=production
AF_EDITION=private
AF_REQUIRE_AUTH=true
AF_REQUIRE_HTTPS=true
AF_FIPS_MODE=false  # Customer configurable
AF_RATE_LIMIT_RPM=200  # Higher for private deployments
AF_CUSTOM_DOMAIN=agentforge.customer.com
AF_DATA_RESIDENCY_REGION=customer_choice
AF_COMPLIANCE_FRAMEWORK=custom
```

### Deployment Steps
1. **Customer Requirements Analysis**
   - Assess security requirements
   - Determine compliance needs
   - Plan integration requirements

2. **Custom Configuration**
   - Configure custom domains
   - Set up customer identity integration
   - Customize security policies

3. **Infrastructure Deployment**
   - Deploy on customer infrastructure
   - Configure customer databases
   - Set up customer monitoring

4. **Integration**
   - Integrate with customer systems
   - Configure custom authentication
   - Set up customer-specific features

---

## Production Infrastructure Requirements

### **YES - Database URL Required**

**Why Database is Required:**
- **Agent execution tracking** - Performance analytics and monitoring
- **Swarm coordination analytics** - Real-time coordination metrics
- **System metrics storage** - Time-series performance data
- **Request processing analytics** - API performance optimization
- **User session management** - Authentication and authorization
- **Audit logging** - Compliance and security tracking
- **Configuration persistence** - Dynamic configuration storage

**Supported Databases:**
1. **PostgreSQL (Recommended)** - Full feature support, best performance
2. **MySQL** - Good compatibility, standard features
3. **SQLite** - Development only, not for production
4. **SQL Server** - Enterprise customers, Windows environments
5. **Oracle** - Large enterprise customers

### **YES - Redis URL Required**

**Why Redis is Required:**
- **Session storage** - JWT token blacklisting and session management
- **Caching layer** - LLM response caching for performance
- **Rate limiting** - Distributed rate limiting across instances
- **WebSocket state** - Real-time connection management
- **Queue management** - Background task processing
- **Neural mesh coordination** - Distributed agent state
- **Configuration caching** - Hot configuration reloading

**Redis Alternatives:**
1. **Redis Cluster** - High availability and scaling
2. **AWS ElastiCache** - Managed Redis service
3. **Azure Cache for Redis** - Azure managed service
4. **Google Memorystore** - Google Cloud managed Redis

## Deployment Process by Edition

### Commercial Edition Process
```bash
# 1. Set up infrastructure
terraform apply -var="edition=commercial"

# 2. Configure environment
cp config/env_production_template.txt .env
# Edit .env with your values

# 3. Deploy application
docker-compose -f docker-compose.production.yml up -d

# 4. Configure authentication
curl -X POST https://app.agentforge.com/v1/auth/setup \
  -d '{"admin_email": "admin@company.com"}'

# 5. Verify deployment
curl https://app.agentforge.com/ready
```

### FedCiv Edition Process
```bash
# 1. Obtain FedRAMP authorization
# 2. Set up GovCloud infrastructure
# 3. Configure FIPS-compliant environment
AF_EDITION=fedciv AF_FIPS_MODE=true docker-compose up -d

# 4. Integrate with government identity providers
# 5. Complete compliance validation
```

### DoD Edition Process
```bash
# 1. Obtain CMMC certification
# 2. Set up DISA-approved infrastructure
# 3. Configure maximum security
AF_EDITION=dod AF_FIPS_MODE=true docker-compose up -d

# 4. Integrate with DoD identity systems
# 5. Complete security validation
```

### Private Edition Process
```bash
# 1. Customer requirements analysis
# 2. Custom infrastructure setup
# 3. Configure customer-specific settings
AF_EDITION=private AF_CUSTOM_DOMAIN=customer.com docker-compose up -d

# 4. Customer integration and testing
# 5. Go-live support
```

## Infrastructure Sizing Recommendations

### Small Deployment (< 1000 users)
- **Backend:** 2 CPU, 4GB RAM
- **Database:** 2 CPU, 8GB RAM, 100GB storage
- **Redis:** 1 CPU, 2GB RAM
- **Frontend:** 1 CPU, 1GB RAM each

### Medium Deployment (1000-10000 users)
- **Backend:** 4 CPU, 8GB RAM (2+ instances)
- **Database:** 4 CPU, 16GB RAM, 500GB storage
- **Redis:** 2 CPU, 4GB RAM (cluster)
- **Frontend:** 2 CPU, 2GB RAM each

### Large Deployment (10000+ users)
- **Backend:** 8 CPU, 16GB RAM (3+ instances)
- **Database:** 8 CPU, 32GB RAM, 1TB+ storage
- **Redis:** 4 CPU, 8GB RAM (cluster)
- **Frontend:** 4 CPU, 4GB RAM each
- **Load Balancer:** High-availability setup

## Security Requirements by Edition

### Commercial
- **TLS 1.2+** for all communications
- **OAuth2/OIDC** authentication
- **Standard encryption** at rest and in transit
- **SOC 2 Type II** compliance ready

### FedCiv
- **TLS 1.3** required
- **FIPS 140-2** validated encryption
- **FedRAMP Moderate/High** authorization
- **US Government cloud** regions only
- **Enhanced audit logging**

### DoD
- **TLS 1.3** with perfect forward secrecy
- **FIPS 140-2 Level 3** encryption
- **CMMC Level 2+** certification
- **IL4/IL5** data classification
- **CAC/PIV** authentication required
- **Maximum audit logging**

### Private
- **Customer-defined** security requirements
- **Flexible compliance** framework support
- **Custom authentication** integration
- **On-premises** or private cloud deployment

## Next Steps Summary

### 1. **Choose Your Edition**
   - **Commercial:** Standard enterprise deployment
   - **FedCiv:** Government civilian agencies
   - **DoD:** Military and defense contractors
   - **Private:** Custom enterprise deployment

### 2. **Set Up Infrastructure**
   - **Database:** PostgreSQL/MySQL in appropriate region
   - **Redis:** For caching and session management
   - **Load Balancer:** For high availability
   - **Monitoring:** Prometheus + Grafana

### 3. **Configure Authentication**
   - **OAuth2 Provider:** Auth0, Okta, Azure AD, or Google
   - **JWT Secrets:** Generate secure 256-bit keys
   - **RBAC Setup:** Configure roles and permissions
   - **Multi-tenant:** Set up organizations and projects

### 4. **Security Hardening**
   - **HTTPS/TLS:** Configure certificates
   - **CORS:** Set production origins
   - **Rate Limiting:** Configure appropriate limits
   - **Audit Logging:** Enable compliance logging

### 5. **Compliance Configuration**
   - **FIPS Mode:** Enable for government deployments
   - **Data Residency:** Configure appropriate regions
   - **Audit Retention:** Set compliance-appropriate retention
   - **Backup Encryption:** Enable encrypted backups

### 6. **Go-Live Checklist**
   - [ ] All API keys configured
   - [ ] Database and Redis operational
   - [ ] Authentication provider integrated
   - [ ] Health checks passing
   - [ ] Monitoring configured
   - [ ] Security scanning completed
   - [ ] Compliance validation done
   - [ ] Backup and recovery tested

## Support and Compliance

### Commercial Support
- **Standard SLA:** 99.9% uptime
- **Support Hours:** Business hours
- **Response Time:** 4-hour initial response

### Government Support (FedCiv/DoD)
- **Enhanced SLA:** 99.95% uptime
- **Support Hours:** 24/7 for critical issues
- **Response Time:** 1-hour initial response
- **Security Clearance:** Support staff with appropriate clearances

### Private Support
- **Custom SLA:** Negotiated with customer
- **Dedicated Support:** Optional dedicated support team
- **On-site Support:** Available for large deployments

**The AgentForge platform is now ready for enterprise deployment across all editions with comprehensive security, compliance, and infrastructure support!** 🚀
