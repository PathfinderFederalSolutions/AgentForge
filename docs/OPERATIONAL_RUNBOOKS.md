# AgentForge Operational Runbooks

## Table of Contents

1. [Runbook Overview](#runbook-overview)
2. [System Health Monitoring](#system-health-monitoring)
3. [Agent Swarm Management](#agent-swarm-management)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Security Incident Response](#security-incident-response)
6. [Database Operations](#database-operations)
7. [Scaling Operations](#scaling-operations)
8. [Backup and Recovery](#backup-and-recovery)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Common Issues and Solutions](#common-issues-and-solutions)

## Runbook Overview

This document provides operational procedures for managing and troubleshooting the AgentForge platform. These runbooks are designed for:

- **Platform Engineers**: Infrastructure and deployment management
- **DevOps Teams**: Monitoring, scaling, and incident response
- **Support Engineers**: Troubleshooting and user support
- **Security Teams**: Security incident response and compliance

### Emergency Contacts

- **Primary On-Call**: +1-555-AGENT-01
- **Secondary On-Call**: +1-555-AGENT-02
- **Security Team**: security@agentforge.ai
- **Platform Team**: platform@agentforge.ai

### Severity Levels

- **P0 (Critical)**: System down, major security breach
- **P1 (High)**: Significant performance degradation, partial outage
- **P2 (Medium)**: Minor performance issues, non-critical errors
- **P3 (Low)**: Cosmetic issues, feature requests

## System Health Monitoring

### 1. Health Check Procedures

#### Basic Health Check
```bash
#!/bin/bash
# scripts/health-check.sh

echo "=== AgentForge Health Check ==="
echo "Timestamp: $(date)"

# API Health
echo "Checking API health..."
curl -f http://localhost:8000/health || echo "❌ API unhealthy"

# Database connectivity
echo "Checking database..."
kubectl exec -n agentforge postgres-0 -- pg_isready -U postgres || echo "❌ Database unhealthy"

# Redis connectivity
echo "Checking Redis..."
kubectl exec -n agentforge redis-0 -- redis-cli ping || echo "❌ Redis unhealthy"

# NATS connectivity
echo "Checking NATS..."
kubectl exec -n agentforge nats-0 -- nats server check || echo "❌ NATS unhealthy"

# Service status
echo "Checking service status..."
kubectl get pods -n agentforge | grep -v Running && echo "❌ Some pods not running"

echo "Health check completed"
```

#### Comprehensive System Status
```bash
#!/bin/bash
# scripts/system-status.sh

echo "=== Comprehensive System Status ==="

# Get system metrics
curl -s http://localhost:8000/v1/system/status | jq '.'

# Check resource usage
echo "Resource Usage:"
kubectl top nodes
kubectl top pods -n agentforge --sort-by=cpu | head -10

# Check agent deployments
echo "Active Agent Deployments:"
curl -s http://localhost:8000/v1/agents/deployments | jq '.active_deployments'

# Check neural mesh status
echo "Neural Mesh Status:"
curl -s http://localhost:8000/v1/neural-mesh/status | jq '.'
```

### 2. Key Metrics to Monitor

#### System Metrics
```yaml
# monitoring/key-metrics.yaml
metrics:
  system:
    - name: "system_health"
      query: "up{job='agentforge-api'}"
      threshold: "< 1"
      severity: "P0"
    
    - name: "response_time"
      query: "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
      threshold: "> 5"
      severity: "P1"
    
    - name: "error_rate"
      query: "rate(http_requests_total{status=~'5..'}[5m])"
      threshold: "> 0.05"
      severity: "P1"

  agents:
    - name: "agent_deployment_rate"
      query: "rate(agents_deployed_total[5m])"
      threshold: "< 0.1"
      severity: "P2"
    
    - name: "agent_failure_rate"
      query: "rate(agent_failures_total[5m])"
      threshold: "> 0.1"
      severity: "P1"

  infrastructure:
    - name: "cpu_usage"
      query: "100 - (avg(irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)"
      threshold: "> 80"
      severity: "P2"
    
    - name: "memory_usage"
      query: "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
      threshold: "> 85"
      severity: "P1"
    
    - name: "disk_usage"
      query: "100 - ((node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100)"
      threshold: "> 90"
      severity: "P1"
```

### 3. Automated Health Monitoring

#### Health Check Service
```python
# scripts/health_monitor.py
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List

class HealthMonitor:
    def __init__(self):
        self.checks = [
            {"name": "API", "url": "http://localhost:8000/health"},
            {"name": "System Status", "url": "http://localhost:8000/v1/system/status"},
            {"name": "Neural Mesh", "url": "http://localhost:8000/v1/neural-mesh/status"}
        ]
    
    async def run_health_check(self) -> Dict:
        """Run comprehensive health check"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": []
        }
        
        async with aiohttp.ClientSession() as session:
            for check in self.checks:
                try:
                    async with session.get(check["url"], timeout=10) as response:
                        if response.status == 200:
                            status = "healthy"
                        else:
                            status = "unhealthy"
                            results["overall_status"] = "unhealthy"
                except Exception as e:
                    status = "error"
                    results["overall_status"] = "unhealthy"
                    logging.error(f"Health check failed for {check['name']}: {e}")
                
                results["checks"].append({
                    "name": check["name"],
                    "status": status,
                    "url": check["url"]
                })
        
        return results
    
    async def continuous_monitoring(self, interval: int = 60):
        """Run continuous health monitoring"""
        while True:
            results = await self.run_health_check()
            
            if results["overall_status"] != "healthy":
                await self.send_alert(results)
            
            logging.info(f"Health check completed: {results['overall_status']}")
            await asyncio.sleep(interval)
    
    async def send_alert(self, results: Dict):
        """Send alert for unhealthy system"""
        # Implement alerting logic (Slack, email, PagerDuty, etc.)
        pass

if __name__ == "__main__":
    monitor = HealthMonitor()
    asyncio.run(monitor.continuous_monitoring())
```

## Agent Swarm Management

### 1. Agent Deployment Management

#### Deploy Agent Swarm
```bash
#!/bin/bash
# scripts/deploy-agent-swarm.sh

OBJECTIVE="$1"
SCALE="$2"
SPECIALIZATIONS="$3"

if [ -z "$OBJECTIVE" ]; then
    echo "Usage: $0 <objective> <scale> <specializations>"
    echo "Example: $0 'Security analysis' 'large_swarm' 'security_analysis,vulnerability_scanning'"
    exit 1
fi

echo "Deploying agent swarm..."
echo "Objective: $OBJECTIVE"
echo "Scale: $SCALE"
echo "Specializations: $SPECIALIZATIONS"

# Deploy swarm via API
curl -X POST http://localhost:8000/v1/agents/deploy \
  -H "Content-Type: application/json" \
  -d "{
    \"objective\": \"$OBJECTIVE\",
    \"scale\": \"$SCALE\",
    \"specializations\": [\"$(echo $SPECIALIZATIONS | sed 's/,/","/g')\"],
    \"configuration\": {
      \"timeout\": 600,
      \"priority\": \"high\"
    }
  }" | jq '.'
```

#### Monitor Agent Swarm Status
```bash
#!/bin/bash
# scripts/monitor-swarm.sh

SWARM_ID="$1"

if [ -z "$SWARM_ID" ]; then
    echo "Usage: $0 <swarm_id>"
    exit 1
fi

echo "Monitoring swarm: $SWARM_ID"

while true; do
    echo "$(date): Checking swarm status..."
    
    STATUS=$(curl -s http://localhost:8000/v1/agents/status/$SWARM_ID | jq -r '.status')
    ACTIVE=$(curl -s http://localhost:8000/v1/agents/status/$SWARM_ID | jq -r '.agentsActive')
    COMPLETED=$(curl -s http://localhost:8000/v1/agents/status/$SWARM_ID | jq -r '.agentsCompleted')
    
    echo "Status: $STATUS, Active: $ACTIVE, Completed: $COMPLETED"
    
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        echo "Swarm finished with status: $STATUS"
        break
    fi
    
    sleep 30
done
```

### 2. Agent Performance Optimization

#### Agent Resource Monitoring
```python
# scripts/agent_monitor.py
import requests
import time
import json
from datetime import datetime

class AgentMonitor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_agent_metrics(self):
        """Get current agent metrics"""
        try:
            response = requests.get(f"{self.base_url}/v1/agents/metrics")
            return response.json()
        except Exception as e:
            print(f"Error fetching agent metrics: {e}")
            return None
    
    def analyze_performance(self, metrics):
        """Analyze agent performance and identify issues"""
        issues = []
        
        # Check for high failure rates
        if metrics.get('failure_rate', 0) > 0.1:
            issues.append("High agent failure rate detected")
        
        # Check for slow response times
        if metrics.get('avg_response_time', 0) > 10:
            issues.append("Slow agent response times detected")
        
        # Check for resource constraints
        if metrics.get('resource_utilization', 0) > 0.9:
            issues.append("High resource utilization detected")
        
        return issues
    
    def recommend_optimizations(self, issues):
        """Recommend optimizations based on identified issues"""
        recommendations = []
        
        for issue in issues:
            if "failure rate" in issue:
                recommendations.append("Consider reducing agent concurrency")
                recommendations.append("Check for resource constraints")
            
            if "response times" in issue:
                recommendations.append("Scale up agent resources")
                recommendations.append("Optimize agent algorithms")
            
            if "resource utilization" in issue:
                recommendations.append("Scale out agent deployment")
                recommendations.append("Implement resource limits")
        
        return recommendations
    
    def monitor_continuously(self, interval=60):
        """Monitor agent performance continuously"""
        while True:
            print(f"\n{datetime.now()}: Checking agent performance...")
            
            metrics = self.get_agent_metrics()
            if metrics:
                issues = self.analyze_performance(metrics)
                
                if issues:
                    print("Issues detected:")
                    for issue in issues:
                        print(f"  - {issue}")
                    
                    recommendations = self.recommend_optimizations(issues)
                    print("Recommendations:")
                    for rec in recommendations:
                        print(f"  - {rec}")
                else:
                    print("No performance issues detected")
            
            time.sleep(interval)

if __name__ == "__main__":
    monitor = AgentMonitor()
    monitor.monitor_continuously()
```

### 3. Agent Troubleshooting

#### Common Agent Issues
```bash
#!/bin/bash
# scripts/troubleshoot-agents.sh

echo "=== Agent Troubleshooting ==="

# Check for failed agents
echo "Checking for failed agents..."
kubectl get pods -n agentforge | grep -E "(Error|CrashLoopBackOff|Failed)"

# Check agent logs for errors
echo "Checking recent agent errors..."
kubectl logs -n agentforge -l app=swarm-service --tail=100 | grep -i error

# Check resource constraints
echo "Checking resource constraints..."
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check NATS message queue backlog
echo "Checking message queue backlog..."
kubectl exec -n agentforge nats-0 -- nats stream info TOOLS

# Check neural mesh connectivity
echo "Checking neural mesh connectivity..."
curl -s http://localhost:8000/v1/neural-mesh/status | jq '.connectivity'
```

## Performance Troubleshooting

### 1. Response Time Issues

#### Diagnose Slow Response Times
```bash
#!/bin/bash
# scripts/diagnose-performance.sh

echo "=== Performance Diagnosis ==="

# Check API response times
echo "API Response Times (last 5 minutes):"
curl -s http://localhost:9090/api/v1/query?query='histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))' | jq '.data.result[0].value[1]'

# Check database performance
echo "Database Performance:"
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;"

# Check Redis performance
echo "Redis Performance:"
kubectl exec -n agentforge redis-0 -- redis-cli --latency-history -i 1 | head -10

# Check agent processing times
echo "Agent Processing Times:"
curl -s http://localhost:8000/v1/agents/metrics | jq '.processing_times'
```

#### Performance Optimization Procedures
```python
# scripts/performance_optimizer.py
import requests
import json
import time
from typing import Dict, List

class PerformanceOptimizer:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.prometheus_url = "http://localhost:9090"
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        metrics = {}
        
        # API response times
        query = "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
        response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                               params={"query": query})
        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                metrics['api_response_time'] = float(data['data']['result'][0]['value'][1])
        
        # System resource usage
        cpu_query = "100 - (avg(irate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)"
        response = requests.get(f"{self.prometheus_url}/api/v1/query",
                               params={"query": cpu_query})
        if response.status_code == 200:
            data = response.json()
            if data['data']['result']:
                metrics['cpu_usage'] = float(data['data']['result'][0]['value'][1])
        
        return metrics
    
    def identify_bottlenecks(self, metrics: Dict) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if metrics.get('api_response_time', 0) > 5:
            bottlenecks.append("High API response times")
        
        if metrics.get('cpu_usage', 0) > 80:
            bottlenecks.append("High CPU usage")
        
        return bottlenecks
    
    def apply_optimizations(self, bottlenecks: List[str]):
        """Apply performance optimizations"""
        for bottleneck in bottlenecks:
            if "API response times" in bottleneck:
                self.scale_api_pods()
            
            if "CPU usage" in bottleneck:
                self.optimize_resource_allocation()
    
    def scale_api_pods(self):
        """Scale API pods to improve response times"""
        import subprocess
        
        print("Scaling API pods...")
        result = subprocess.run([
            "kubectl", "scale", "deployment", "agentforge-api", 
            "--replicas=5", "-n", "agentforge"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("API pods scaled successfully")
        else:
            print(f"Failed to scale API pods: {result.stderr}")
    
    def optimize_resource_allocation(self):
        """Optimize resource allocation"""
        print("Optimizing resource allocation...")
        # Implement resource optimization logic
        pass

if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    
    metrics = optimizer.get_performance_metrics()
    print(f"Current metrics: {json.dumps(metrics, indent=2)}")
    
    bottlenecks = optimizer.identify_bottlenecks(metrics)
    if bottlenecks:
        print(f"Bottlenecks identified: {bottlenecks}")
        optimizer.apply_optimizations(bottlenecks)
    else:
        print("No performance bottlenecks detected")
```

### 2. Memory and Resource Issues

#### Memory Usage Investigation
```bash
#!/bin/bash
# scripts/memory-investigation.sh

echo "=== Memory Usage Investigation ==="

# Check pod memory usage
echo "Pod Memory Usage:"
kubectl top pods -n agentforge --sort-by=memory | head -20

# Check node memory usage
echo "Node Memory Usage:"
kubectl top nodes

# Check for memory leaks in specific pods
echo "Checking for memory leaks..."
kubectl exec -n agentforge agentforge-api-0 -- ps aux | sort -nrk 4 | head

# Check system memory pressure
echo "Memory pressure indicators:"
kubectl describe nodes | grep -A 10 "Conditions:" | grep -E "(MemoryPressure|DiskPressure)"

# Check OOM kills
echo "Recent OOM kills:"
dmesg | grep -i "killed process" | tail -10
```

## Security Incident Response

### 1. Security Alert Response

#### Immediate Response Procedures
```bash
#!/bin/bash
# scripts/security-incident-response.sh

INCIDENT_TYPE="$1"
SEVERITY="$2"

echo "=== Security Incident Response ==="
echo "Incident Type: $INCIDENT_TYPE"
echo "Severity: $SEVERITY"
echo "Timestamp: $(date)"

# Step 1: Isolate affected systems
if [ "$SEVERITY" = "critical" ]; then
    echo "CRITICAL: Isolating affected systems..."
    
    # Block suspicious traffic
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-isolation
  namespace: agentforge
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress: []
  egress: []
EOF
fi

# Step 2: Collect evidence
echo "Collecting security evidence..."
mkdir -p /tmp/incident-$(date +%Y%m%d-%H%M%S)
EVIDENCE_DIR="/tmp/incident-$(date +%Y%m%d-%H%M%S)"

# Collect logs
kubectl logs -n agentforge --all-containers=true --since=1h > "$EVIDENCE_DIR/system-logs.txt"

# Collect security audit logs
curl -s http://localhost:8000/v1/security/audit-logs > "$EVIDENCE_DIR/audit-logs.json"

# Collect system metrics
curl -s http://localhost:9090/api/v1/query_range?query=up&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60 > "$EVIDENCE_DIR/metrics.json"

# Step 3: Notify security team
echo "Notifying security team..."
# Implement notification logic (Slack, email, PagerDuty)

echo "Evidence collected in: $EVIDENCE_DIR"
echo "Incident response initiated"
```

#### Security Monitoring
```python
# scripts/security_monitor.py
import requests
import json
import time
import logging
from datetime import datetime, timedelta

class SecurityMonitor:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.alert_thresholds = {
            'failed_logins': 10,
            'suspicious_requests': 50,
            'anomalous_behavior': 5
        }
    
    def get_security_metrics(self):
        """Get current security metrics"""
        try:
            response = requests.get(f"{self.base_url}/v1/security/metrics")
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching security metrics: {e}")
            return None
    
    def analyze_threats(self, metrics):
        """Analyze metrics for security threats"""
        threats = []
        
        # Check for brute force attacks
        if metrics.get('failed_logins_per_hour', 0) > self.alert_thresholds['failed_logins']:
            threats.append({
                'type': 'brute_force',
                'severity': 'high',
                'description': f"High number of failed logins: {metrics['failed_logins_per_hour']}"
            })
        
        # Check for suspicious request patterns
        if metrics.get('suspicious_requests_per_hour', 0) > self.alert_thresholds['suspicious_requests']:
            threats.append({
                'type': 'suspicious_activity',
                'severity': 'medium',
                'description': f"Suspicious request patterns detected: {metrics['suspicious_requests_per_hour']}"
            })
        
        return threats
    
    def respond_to_threat(self, threat):
        """Respond to identified security threat"""
        if threat['severity'] == 'high':
            # Immediate response for high severity threats
            self.trigger_incident_response(threat)
        
        # Log the threat
        logging.warning(f"Security threat detected: {threat}")
        
        # Send alert
        self.send_security_alert(threat)
    
    def trigger_incident_response(self, threat):
        """Trigger automated incident response"""
        print(f"SECURITY ALERT: {threat['type']} - {threat['description']}")
        
        # Implement automated response actions
        if threat['type'] == 'brute_force':
            self.implement_rate_limiting()
        
    def implement_rate_limiting(self):
        """Implement emergency rate limiting"""
        # This would integrate with your API gateway or load balancer
        print("Implementing emergency rate limiting...")
    
    def send_security_alert(self, threat):
        """Send security alert to team"""
        # Implement alerting (Slack, email, etc.)
        pass
    
    def monitor_continuously(self, interval=300):
        """Monitor security continuously"""
        while True:
            print(f"\n{datetime.now()}: Running security check...")
            
            metrics = self.get_security_metrics()
            if metrics:
                threats = self.analyze_threats(metrics)
                
                if threats:
                    for threat in threats:
                        self.respond_to_threat(threat)
                else:
                    print("No security threats detected")
            
            time.sleep(interval)

if __name__ == "__main__":
    monitor = SecurityMonitor()
    monitor.monitor_continuously()
```

### 2. Access Control Management

#### User Access Audit
```bash
#!/bin/bash
# scripts/access-audit.sh

echo "=== Access Control Audit ==="

# Check active user sessions
echo "Active user sessions:"
curl -s http://localhost:8000/v1/auth/sessions | jq '.active_sessions'

# Check recent login attempts
echo "Recent login attempts:"
curl -s http://localhost:8000/v1/security/audit-logs?event_type=login&since=24h | jq '.events[] | {timestamp, user_id, result, ip_address}'

# Check privileged access usage
echo "Privileged access usage:"
curl -s http://localhost:8000/v1/security/audit-logs?event_type=privileged_access&since=24h | jq '.events[] | {timestamp, user_id, action, resource}'

# Check for dormant accounts
echo "Dormant accounts (no activity in 30 days):"
curl -s http://localhost:8000/v1/auth/users/dormant?days=30 | jq '.dormant_users'
```

## Database Operations

### 1. Database Health and Maintenance

#### Database Health Check
```bash
#!/bin/bash
# scripts/database-health.sh

echo "=== Database Health Check ==="

# Check database connectivity
echo "Database connectivity:"
kubectl exec -n agentforge postgres-0 -- pg_isready -U postgres

# Check database size
echo "Database size:"
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT pg_database.datname,
       pg_size_pretty(pg_database_size(pg_database.datname)) AS size
FROM pg_database;"

# Check active connections
echo "Active connections:"
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';"

# Check slow queries
echo "Slow queries (>5 seconds):"
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 5000
ORDER BY mean_exec_time DESC
LIMIT 10;"

# Check database locks
echo "Current locks:"
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT blocked_locks.pid AS blocked_pid,
       blocked_activity.usename AS blocked_user,
       blocking_locks.pid AS blocking_pid,
       blocking_activity.usename AS blocking_user,
       blocked_activity.query AS blocked_statement,
       blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.DATABASE IS NOT DISTINCT FROM blocked_locks.DATABASE
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
    AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
    AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
    AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
    AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
    AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
    AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
    AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
    AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.GRANTED;"
```

#### Database Maintenance
```bash
#!/bin/bash
# scripts/database-maintenance.sh

echo "=== Database Maintenance ==="

# Vacuum and analyze tables
echo "Running VACUUM ANALYZE..."
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "VACUUM ANALYZE;"

# Update statistics
echo "Updating table statistics..."
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "ANALYZE;"

# Check and rebuild indexes if needed
echo "Checking index usage..."
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC;"

# Clean up old data (if applicable)
echo "Cleaning up old audit logs (>90 days)..."
kubectl exec -n agentforge postgres-0 -- psql -U postgres -d agentforge -c "
DELETE FROM audit_logs 
WHERE created_at < NOW() - INTERVAL '90 days';"

echo "Database maintenance completed"
```

### 2. Database Backup and Recovery

#### Automated Backup
```bash
#!/bin/bash
# scripts/database-backup.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="agentforge_backup_${TIMESTAMP}.sql"

echo "=== Database Backup ==="
echo "Creating backup: $BACKUP_FILE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database dump
kubectl exec -n agentforge postgres-0 -- pg_dump -U postgres agentforge > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

# Upload to S3 (if configured)
if [ -n "$AWS_S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp "$BACKUP_DIR/${BACKUP_FILE}.gz" "s3://$AWS_S3_BUCKET/database-backups/"
fi

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "agentforge_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

## Scaling Operations

### 1. Horizontal Pod Autoscaling

#### Configure HPA
```bash
#!/bin/bash
# scripts/configure-hpa.sh

SERVICE="$1"
MIN_REPLICAS="${2:-3}"
MAX_REPLICAS="${3:-50}"
CPU_TARGET="${4:-70}"

if [ -z "$SERVICE" ]; then
    echo "Usage: $0 <service> [min_replicas] [max_replicas] [cpu_target]"
    exit 1
fi

echo "Configuring HPA for service: $SERVICE"
echo "Min replicas: $MIN_REPLICAS"
echo "Max replicas: $MAX_REPLICAS"
echo "CPU target: $CPU_TARGET%"

kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ${SERVICE}-hpa
  namespace: agentforge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: $SERVICE
  minReplicas: $MIN_REPLICAS
  maxReplicas: $MAX_REPLICAS
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: $CPU_TARGET
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

echo "HPA configured for $SERVICE"
```

#### Monitor Scaling Events
```bash
#!/bin/bash
# scripts/monitor-scaling.sh

echo "=== Scaling Monitoring ==="

# Check HPA status
echo "HPA Status:"
kubectl get hpa -n agentforge

# Check recent scaling events
echo "Recent scaling events:"
kubectl get events -n agentforge | grep -E "(Scaled|SuccessfulCreate|SuccessfulDelete)" | tail -20

# Check pod resource usage
echo "Pod resource usage:"
kubectl top pods -n agentforge --sort-by=cpu | head -10

# Check node resource usage
echo "Node resource usage:"
kubectl top nodes
```

### 2. Manual Scaling Procedures

#### Emergency Scaling
```bash
#!/bin/bash
# scripts/emergency-scale.sh

SERVICE="$1"
REPLICAS="$2"

if [ -z "$SERVICE" ] || [ -z "$REPLICAS" ]; then
    echo "Usage: $0 <service> <replicas>"
    echo "Example: $0 agentforge-api 10"
    exit 1
fi

echo "Emergency scaling: $SERVICE to $REPLICAS replicas"

# Scale the deployment
kubectl scale deployment $SERVICE --replicas=$REPLICAS -n agentforge

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=$SERVICE -n agentforge --timeout=300s

# Check status
echo "Scaling completed. Current status:"
kubectl get deployment $SERVICE -n agentforge
kubectl get pods -l app=$SERVICE -n agentforge
```

## Backup and Recovery

### 1. Complete System Backup

#### Full System Backup
```bash
#!/bin/bash
# scripts/full-system-backup.sh

BACKUP_DIR="/backups/agentforge"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="agentforge_full_backup_${TIMESTAMP}"

echo "=== Full System Backup ==="
echo "Backup name: $BACKUP_NAME"

mkdir -p "$BACKUP_DIR/$BACKUP_NAME"

# Backup Kubernetes configurations
echo "Backing up Kubernetes configurations..."
kubectl get all,configmaps,secrets,pv,pvc -n agentforge -o yaml > "$BACKUP_DIR/$BACKUP_NAME/k8s-resources.yaml"

# Backup database
echo "Backing up database..."
kubectl exec -n agentforge postgres-0 -- pg_dump -U postgres agentforge > "$BACKUP_DIR/$BACKUP_NAME/database.sql"

# Backup Redis data
echo "Backing up Redis data..."
kubectl exec -n agentforge redis-0 -- redis-cli BGSAVE
kubectl cp agentforge/redis-0:/data/dump.rdb "$BACKUP_DIR/$BACKUP_NAME/redis-dump.rdb"

# Backup persistent volumes
echo "Backing up persistent volumes..."
kubectl get pv -o yaml > "$BACKUP_DIR/$BACKUP_NAME/persistent-volumes.yaml"

# Backup configuration files
echo "Backing up configuration files..."
cp -r deployment/k8s/ "$BACKUP_DIR/$BACKUP_NAME/k8s-manifests/"
cp -r config/ "$BACKUP_DIR/$BACKUP_NAME/config/"

# Create archive
echo "Creating backup archive..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"

# Upload to remote storage
if [ -n "$AWS_S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" "s3://$AWS_S3_BUCKET/system-backups/"
fi

# Clean up temporary files
rm -rf "$BACKUP_DIR/$BACKUP_NAME"

echo "Full system backup completed: ${BACKUP_NAME}.tar.gz"
```

### 2. Disaster Recovery

#### Disaster Recovery Procedure
```bash
#!/bin/bash
# scripts/disaster-recovery.sh

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 agentforge_full_backup_20240101_120000.tar.gz"
    exit 1
fi

echo "=== Disaster Recovery ==="
echo "Restoring from backup: $BACKUP_FILE"

# Extract backup
RESTORE_DIR="/tmp/restore_$(date +%s)"
mkdir -p "$RESTORE_DIR"
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR"

BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)

# Restore Kubernetes resources
echo "Restoring Kubernetes resources..."
kubectl apply -f "$RESTORE_DIR/$BACKUP_NAME/k8s-resources.yaml"

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n agentforge --timeout=300s

# Restore database
echo "Restoring database..."
kubectl exec -i postgres-0 -n agentforge -- psql -U postgres agentforge < "$RESTORE_DIR/$BACKUP_NAME/database.sql"

# Restore Redis data
echo "Restoring Redis data..."
kubectl cp "$RESTORE_DIR/$BACKUP_NAME/redis-dump.rdb" agentforge/redis-0:/data/dump.rdb
kubectl exec -n agentforge redis-0 -- redis-cli DEBUG RESTART

# Verify restoration
echo "Verifying restoration..."
kubectl get pods -n agentforge
curl -f http://localhost:8000/health

echo "Disaster recovery completed"

# Clean up
rm -rf "$RESTORE_DIR"
```

## Monitoring and Alerting

### 1. Alert Configuration

#### Prometheus Alert Rules
```yaml
# monitoring/alert-rules.yaml
groups:
- name: agentforge.rules
  rules:
  - alert: APIHighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High API error rate detected"
      description: "API error rate is {{ $value }} errors per second"

  - alert: DatabaseConnectionsHigh
    expr: pg_stat_database_numbackends > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database connection count"
      description: "Database has {{ $value }} active connections"

  - alert: AgentDeploymentFailed
    expr: increase(agent_deployment_failures_total[5m]) > 5
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Multiple agent deployment failures"
      description: "{{ $value }} agent deployments failed in the last 5 minutes"

  - alert: NeuralMeshDisconnected
    expr: neural_mesh_connectivity < 0.8
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Neural mesh connectivity degraded"
      description: "Neural mesh connectivity is at {{ $value }}"
```

### 2. Custom Monitoring Scripts

#### System Health Dashboard
```python
# scripts/health_dashboard.py
import requests
import json
import time
from datetime import datetime
import subprocess

class HealthDashboard:
    def __init__(self):
        self.services = [
            {"name": "API", "url": "http://localhost:8000/health"},
            {"name": "Database", "check": self.check_database},
            {"name": "Redis", "check": self.check_redis},
            {"name": "NATS", "check": self.check_nats}
        ]
    
    def check_database(self):
        """Check database connectivity"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", "agentforge", "postgres-0", "--",
                "pg_isready", "-U", "postgres"
            ], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def check_redis(self):
        """Check Redis connectivity"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", "agentforge", "redis-0", "--",
                "redis-cli", "ping"
            ], capture_output=True, text=True)
            return "PONG" in result.stdout
        except:
            return False
    
    def check_nats(self):
        """Check NATS connectivity"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", "agentforge", "nats-0", "--",
                "nats", "server", "check"
            ], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def get_system_metrics(self):
        """Get system metrics"""
        try:
            response = requests.get("http://localhost:8000/v1/system/status")
            return response.json()
        except:
            return None
    
    def display_dashboard(self):
        """Display health dashboard"""
        print("\n" + "="*60)
        print(f"AgentForge Health Dashboard - {datetime.now()}")
        print("="*60)
        
        # Service health
        print("\nService Health:")
        for service in self.services:
            if 'url' in service:
                try:
                    response = requests.get(service['url'], timeout=5)
                    status = "✅ Healthy" if response.status_code == 200 else "❌ Unhealthy"
                except:
                    status = "❌ Unhealthy"
            else:
                status = "✅ Healthy" if service['check']() else "❌ Unhealthy"
            
            print(f"  {service['name']:20} {status}")
        
        # System metrics
        metrics = self.get_system_metrics()
        if metrics:
            print(f"\nSystem Metrics:")
            print(f"  Active Agents:     {metrics.get('metrics', {}).get('active_agents', 'N/A')}")
            print(f"  Total Deployments: {metrics.get('metrics', {}).get('total_deployments', 'N/A')}")
            print(f"  Avg Response Time: {metrics.get('metrics', {}).get('avg_response_time', 'N/A')}s")
            print(f"  System Load:       {metrics.get('metrics', {}).get('system_load', 'N/A')}")
        
        print("="*60)
    
    def run_continuous(self, interval=30):
        """Run dashboard continuously"""
        while True:
            self.display_dashboard()
            time.sleep(interval)

if __name__ == "__main__":
    dashboard = HealthDashboard()
    dashboard.run_continuous()
```

## Common Issues and Solutions

### 1. Pod Startup Issues

#### Issue: Pods stuck in Pending state
**Symptoms:**
- Pods remain in Pending state
- Events show scheduling failures

**Diagnosis:**
```bash
# Check pod events
kubectl describe pod <pod-name> -n agentforge

# Check node resources
kubectl describe nodes

# Check resource requests
kubectl get pods -n agentforge -o custom-columns=NAME:.metadata.name,CPU_REQUEST:.spec.containers[*].resources.requests.cpu,MEMORY_REQUEST:.spec.containers[*].resources.requests.memory
```

**Solutions:**
1. **Insufficient Resources:**
   ```bash
   # Scale down non-essential services
   kubectl scale deployment <deployment> --replicas=1 -n agentforge
   
   # Add more nodes to cluster
   # (cloud-provider specific commands)
   ```

2. **Resource Quotas:**
   ```bash
   # Check resource quotas
   kubectl get resourcequota -n agentforge
   
   # Increase quota if needed
   kubectl patch resourcequota <quota-name> -p '{"spec":{"hard":{"requests.cpu":"20","requests.memory":"40Gi"}}}'
   ```

### 2. Database Connection Issues

#### Issue: Database connection timeouts
**Symptoms:**
- Applications can't connect to database
- Connection timeout errors in logs

**Diagnosis:**
```bash
# Check database pod status
kubectl get pods -n agentforge | grep postgres

# Check database logs
kubectl logs -n agentforge postgres-0

# Test database connectivity
kubectl exec -n agentforge postgres-0 -- pg_isready -U postgres
```

**Solutions:**
1. **Database Pod Issues:**
   ```bash
   # Restart database pod
   kubectl delete pod postgres-0 -n agentforge
   
   # Check persistent volume
   kubectl get pv,pvc -n agentforge
   ```

2. **Connection Pool Exhaustion:**
   ```bash
   # Check active connections
   kubectl exec -n agentforge postgres-0 -- psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Increase connection limits in postgresql.conf
   # Or restart applications to reset connection pools
   kubectl rollout restart deployment agentforge-api -n agentforge
   ```

### 3. High CPU/Memory Usage

#### Issue: High resource utilization
**Symptoms:**
- Slow response times
- Pod restarts due to resource limits
- Node resource pressure

**Diagnosis:**
```bash
# Check resource usage
kubectl top pods -n agentforge --sort-by=cpu
kubectl top nodes

# Check resource limits
kubectl describe pods -n agentforge | grep -A 5 "Limits:"

# Check for memory leaks
kubectl exec -n agentforge <pod-name> -- ps aux | sort -nrk 4
```

**Solutions:**
1. **Scale Resources:**
   ```bash
   # Horizontal scaling
   kubectl scale deployment agentforge-api --replicas=5 -n agentforge
   
   # Vertical scaling (increase limits)
   kubectl patch deployment agentforge-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"cpu":"2","memory":"4Gi"}}}]}}}}' -n agentforge
   ```

2. **Optimize Applications:**
   ```bash
   # Enable resource monitoring
   kubectl apply -f monitoring/resource-monitoring.yaml
   
   # Analyze performance bottlenecks
   kubectl exec -n agentforge <pod-name> -- top -b -n 1
   ```

### 4. Network Connectivity Issues

#### Issue: Service-to-service communication failures
**Symptoms:**
- Services can't reach each other
- DNS resolution failures
- Timeout errors

**Diagnosis:**
```bash
# Test service connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- curl http://agentforge-api:8000/health

# Check DNS resolution
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- nslookup agentforge-api.agentforge.svc.cluster.local

# Check network policies
kubectl get networkpolicies -n agentforge
```

**Solutions:**
1. **DNS Issues:**
   ```bash
   # Restart CoreDNS
   kubectl rollout restart deployment coredns -n kube-system
   
   # Check DNS configuration
   kubectl get configmap coredns -n kube-system -o yaml
   ```

2. **Network Policy Issues:**
   ```bash
   # Temporarily disable network policies
   kubectl delete networkpolicy --all -n agentforge
   
   # Check service endpoints
   kubectl get endpoints -n agentforge
   ```

This comprehensive runbook provides operational procedures for managing the AgentForge platform effectively. Regular use of these procedures will help maintain system health, prevent issues, and respond quickly to incidents when they occur.
