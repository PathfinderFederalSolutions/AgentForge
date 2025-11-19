# AgentForge API Documentation

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Service-Specific APIs](#service-specific-apis)
5. [WebSocket APIs](#websocket-apis)
6. [OpenAPI Specifications](#openapi-specifications)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [SDK and Client Libraries](#sdk-and-client-libraries)

## API Overview

AgentForge provides multiple API interfaces to support different use cases:

- **Enhanced Chat API** (`apis/enhanced_chat_api.py`) - Primary conversational interface
- **Production AGI API** (`apis/production_agi_api.py`) - Production-grade AGI endpoints
- **Comprehensive AGI API** (`apis/comprehensive_agi_api.py`) - Complete system API
- **Main API** (`apis/main_api_fixed.py`) - Core system functionality
- **Service APIs** - Individual service endpoints

### Base URLs

- **Development**: `http://localhost:8000`
- **Staging**: `https://staging-api.agentforge.ai`
- **Production**: `https://api.agentforge.ai`

### API Versioning

All APIs use versioned endpoints with the pattern `/v1/`, `/v2/`, etc.

```
/v1/chat/message          # Version 1 chat endpoint
/v1/agents/deploy         # Version 1 agent deployment
/v1/system/status         # Version 1 system status
```

## Authentication

### API Key Authentication

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.agentforge.ai/v1/system/status
```

### JWT Token Authentication

```bash
# Login to get JWT token
curl -X POST https://api.agentforge.ai/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "user", "password": "pass"}'

# Use JWT token
curl -H "Authorization: Bearer JWT_TOKEN" \
     -H "Content-Type: application/json" \
     https://api.agentforge.ai/v1/chat/message
```

### OAuth 2.0 (Enterprise)

```bash
# Get access token
curl -X POST https://api.agentforge.ai/oauth/token \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "grant_type=client_credentials&client_id=CLIENT_ID&client_secret=CLIENT_SECRET"

# Use access token
curl -H "Authorization: Bearer ACCESS_TOKEN" \
     https://api.agentforge.ai/v1/agents/deploy
```

## Core APIs

### 1. Chat API

#### POST /v1/chat/message

Process a conversational message with AGI capabilities.

**Request:**
```json
{
  "message": "Analyze this dataset and create a dashboard",
  "context": {
    "userId": "user123",
    "sessionId": "session456",
    "conversationHistory": [
      {
        "role": "user",
        "content": "Hello",
        "timestamp": "2024-01-01T12:00:00Z"
      }
    ],
    "dataSources": [
      {
        "type": "file",
        "name": "sales_data.csv",
        "url": "https://example.com/data.csv"
      }
    ],
    "userPreferences": {
      "responseFormat": "detailed",
      "agentScale": "medium_swarm"
    }
  },
  "capabilities": ["data_analysis", "visualization", "dashboard_creation"]
}
```

**Response:**
```json
{
  "response": "I'm deploying 25 specialized agents to analyze your sales data...",
  "agentsDeployed": 25,
  "processingTime": 2.5,
  "confidence": 0.95,
  "capabilities": ["data_analysis", "visualization", "dashboard_creation"],
  "artifacts": [
    {
      "type": "dashboard",
      "url": "https://dashboards.agentforge.ai/dashboard123",
      "title": "Sales Analytics Dashboard"
    }
  ],
  "metadata": {
    "sessionId": "session456",
    "messageId": "msg789",
    "timestamp": "2024-01-01T12:00:05Z"
  }
}
```

#### GET /v1/chat/history/{sessionId}

Retrieve conversation history for a session.

**Response:**
```json
{
  "sessionId": "session456",
  "messages": [
    {
      "id": "msg001",
      "role": "user",
      "content": "Hello",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "id": "msg002", 
      "role": "assistant",
      "content": "Hello! I'm AgentForge AI...",
      "timestamp": "2024-01-01T12:00:01Z",
      "agentsDeployed": 1
    }
  ],
  "totalMessages": 2,
  "createdAt": "2024-01-01T12:00:00Z",
  "updatedAt": "2024-01-01T12:00:01Z"
}
```

### 2. Agent Management API

#### POST /v1/agents/deploy

Deploy agent swarm for specific tasks.

**Request:**
```json
{
  "objective": "Analyze codebase for security vulnerabilities",
  "scale": "large_swarm",
  "specializations": [
    "security_analysis",
    "code_review", 
    "vulnerability_scanning"
  ],
  "configuration": {
    "maxAgents": 100,
    "timeout": 600,
    "priority": "high"
  },
  "dataSources": [
    {
      "type": "repository",
      "url": "https://github.com/company/repo",
      "branch": "main"
    }
  ]
}
```

**Response:**
```json
{
  "deploymentId": "deploy123",
  "swarmId": "swarm456",
  "agentsDeployed": 87,
  "specializations": {
    "security_analysis": 30,
    "code_review": 35,
    "vulnerability_scanning": 22
  },
  "status": "active",
  "estimatedCompletion": "2024-01-01T12:15:00Z",
  "progress": {
    "filesAnalyzed": 245,
    "totalFiles": 311,
    "completion": 0.79
  }
}
```

#### GET /v1/agents/status/{swarmId}

Get status of deployed agent swarm.

**Response:**
```json
{
  "swarmId": "swarm456",
  "status": "completed",
  "agentsActive": 0,
  "agentsCompleted": 87,
  "agentsFailed": 0,
  "results": {
    "vulnerabilitiesFound": 12,
    "criticalIssues": 3,
    "warningsIssued": 25,
    "filesProcessed": 311
  },
  "artifacts": [
    {
      "type": "security_report",
      "url": "https://reports.agentforge.ai/security123.pdf",
      "title": "Security Analysis Report"
    }
  ],
  "completedAt": "2024-01-01T12:14:30Z"
}
```

### 3. System Status API

#### GET /v1/system/status

Get comprehensive system status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704110400.0,
  "version": "2.0.0",
  "services": {
    "core_services": {
      "enhanced_logging": true,
      "database_manager": true,
      "retry_handler": true,
      "request_pipeline": true
    },
    "ai_services": {
      "neural_mesh_coordinator": true,
      "enhanced_neural_mesh": true,
      "agent_swarm": true,
      "self_coding": true,
      "agi_evolution": true
    },
    "advanced_services": {
      "quantum_scheduler": true,
      "universal_io": true,
      "mega_swarm": true,
      "self_bootstrap": true,
      "security_orchestrator": true
    }
  },
  "metrics": {
    "active_agents": 1247,
    "total_deployments": 5432,
    "avg_response_time": 0.125,
    "system_load": 0.67
  }
}
```

#### GET /v1/system/health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704110400.0,
  "uptime": 86400,
  "checks": {
    "database": "healthy",
    "redis": "healthy", 
    "nats": "healthy",
    "neural_mesh": "healthy"
  }
}
```

### 4. File Upload API

#### POST /v1/upload

Upload files for processing by agents.

**Request (multipart/form-data):**
```bash
curl -X POST https://api.agentforge.ai/v1/upload \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@document.pdf" \
     -F "type=document" \
     -F "processing=extract_insights"
```

**Response:**
```json
{
  "fileId": "file123",
  "filename": "document.pdf",
  "size": 2048576,
  "type": "application/pdf",
  "status": "uploaded",
  "processingJobId": "job456",
  "uploadedAt": "2024-01-01T12:00:00Z",
  "url": "https://files.agentforge.ai/file123"
}
```

#### GET /v1/upload/status/{fileId}

Check file processing status.

**Response:**
```json
{
  "fileId": "file123",
  "status": "completed",
  "processing": {
    "jobId": "job456",
    "agentsDeployed": 5,
    "insights": [
      "Document contains financial projections for Q1-Q4 2024",
      "Revenue growth projected at 15% year-over-year",
      "Key risks identified in market analysis section"
    ]
  },
  "artifacts": [
    {
      "type": "summary",
      "url": "https://artifacts.agentforge.ai/summary123.pdf"
    }
  ]
}
```

## Service-Specific APIs

### 1. Universal I/O Service API

#### POST /v1/universal-io/process

Process any input to any output format.

**Request:**
```json
{
  "input_data": {
    "type": "sensor_data",
    "data": [
      {"timestamp": "2024-01-01T12:00:00Z", "temperature": 23.5, "humidity": 45.2},
      {"timestamp": "2024-01-01T12:01:00Z", "temperature": 23.7, "humidity": 45.1}
    ]
  },
  "output_format": "tactical_dashboard",
  "vertical_domain": "environmental_monitoring",
  "processing_scale": "small_swarm",
  "configuration": {
    "real_time": true,
    "update_frequency": "1min"
  }
}
```

**Response:**
```json
{
  "processId": "proc123",
  "status": "processing",
  "agentsDeployed": 12,
  "estimatedCompletion": "2024-01-01T12:02:00Z",
  "outputPreview": {
    "type": "dashboard",
    "url": "https://dashboards.agentforge.ai/env-monitor-123"
  },
  "processingDetails": {
    "inputParsed": true,
    "agentsSpecialized": ["sensor_analysis", "visualization", "dashboard_generation"],
    "dataPointsProcessed": 2,
    "qualityScore": 0.98
  }
}
```

#### GET /v1/universal-io/capabilities

Get supported input/output capabilities.

**Response:**
```json
{
  "input_types": [
    {
      "type": "document",
      "formats": ["pdf", "docx", "txt", "md", "html"],
      "max_size": "100MB"
    },
    {
      "type": "media", 
      "formats": ["jpg", "png", "mp4", "wav", "mp3"],
      "max_size": "1GB"
    },
    {
      "type": "sensor_data",
      "formats": ["json", "csv", "mqtt", "websocket"],
      "max_rate": "1M events/sec"
    }
  ],
  "output_formats": [
    {
      "type": "application",
      "frameworks": ["react", "vue", "flutter", "native"]
    },
    {
      "type": "visualization",
      "libraries": ["d3", "plotly", "chartjs", "matplotlib"]
    },
    {
      "type": "document",
      "formats": ["pdf", "docx", "html", "markdown"]
    }
  ],
  "vertical_domains": [
    "defense_intelligence",
    "healthcare",
    "finance",
    "environmental_monitoring",
    "business_intelligence"
  ]
}
```

### 2. Neural Mesh API

#### POST /v1/neural-mesh/knowledge

Store knowledge in the neural mesh.

**Request:**
```json
{
  "agentId": "agent123",
  "knowledgeType": "pattern_recognition",
  "data": {
    "patterns": [
      {"type": "anomaly", "confidence": 0.95, "description": "Unusual network traffic pattern"},
      {"type": "trend", "confidence": 0.87, "description": "Increasing CPU usage trend"}
    ],
    "context": "Security monitoring analysis",
    "timestamp": "2024-01-01T12:00:00Z"
  },
  "memoryTier": "L3",
  "retention": "medium_term"
}
```

**Response:**
```json
{
  "knowledgeId": "knowledge456",
  "status": "stored",
  "memoryTier": "L3",
  "agentId": "agent123",
  "correlations": [
    {
      "agentId": "agent789",
      "similarity": 0.82,
      "sharedPatterns": 3
    }
  ],
  "storedAt": "2024-01-01T12:00:01Z"
}
```

#### GET /v1/neural-mesh/context

Retrieve contextual knowledge for an agent.

**Request Parameters:**
- `agentId`: Agent requesting context
- `query`: Context query string
- `memoryTiers`: Comma-separated list of memory tiers (L1,L2,L3,L4)

**Response:**
```json
{
  "agentId": "agent123",
  "context": [
    {
      "knowledgeId": "knowledge456",
      "relevance": 0.94,
      "memoryTier": "L3",
      "data": {
        "patterns": [...],
        "insights": "Previous analysis shows similar patterns indicate potential security threat"
      }
    }
  ],
  "correlatedAgents": ["agent789", "agent101"],
  "confidenceScore": 0.91,
  "retrievedAt": "2024-01-01T12:00:02Z"
}
```

### 3. Quantum Orchestrator API

#### POST /v1/orchestrator/schedule

Schedule quantum-inspired task coordination.

**Request:**
```json
{
  "taskDescription": "Optimize resource allocation across 1000 agents",
  "priority": "high",
  "requiredAgents": 1000,
  "requiredCapabilities": ["optimization", "resource_management", "coordination"],
  "constraints": {
    "maxExecutionTime": 300,
    "resourceLimits": {
      "cpu": "100 cores",
      "memory": "200GB"
    }
  },
  "quantumParameters": {
    "coherenceLevel": "high",
    "entanglementDepth": 3
  }
}
```

**Response:**
```json
{
  "taskId": "task789",
  "schedulingId": "schedule123",
  "status": "scheduled",
  "quantumState": {
    "coherenceLevel": 0.98,
    "entangledAgents": 1000,
    "superpositionStates": 1024
  },
  "estimatedCompletion": "2024-01-01T12:05:00Z",
  "resourceAllocation": {
    "nodesAllocated": 10,
    "cpuCores": 100,
    "memoryGB": 200
  }
}
```

## WebSocket APIs

### Real-Time Dashboard Updates

#### WebSocket: /ws/dashboard

Connect to real-time dashboard updates.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.agentforge.ai/ws/dashboard');

// Subscribe to specific dashboard
ws.send(JSON.stringify({
  type: 'subscribe',
  dashboardId: 'tactical_cop_001',
  updateFrequency: 'real_time'
}));

// Receive updates
ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  if (update.type === 'dashboard_update') {
    updateDashboard(update.widgets);
  }
};
```

**Message Types:**
```json
// Subscribe to dashboard
{
  "type": "subscribe",
  "dashboardId": "tactical_cop_001",
  "updateFrequency": "real_time"
}

// Dashboard update
{
  "type": "dashboard_update",
  "dashboardId": "tactical_cop_001",
  "timestamp": "2024-01-01T12:00:00Z",
  "widgets": [
    {
      "id": "widget1",
      "type": "gauge",
      "value": 87.5,
      "threshold": 90,
      "status": "warning"
    }
  ]
}

// Agent deployment notification
{
  "type": "agent_deployment",
  "swarmId": "swarm123",
  "agentsDeployed": 25,
  "specialization": "data_analysis",
  "status": "active"
}
```

### Real-Time Agent Coordination

#### WebSocket: /ws/agents

Monitor agent activities in real-time.

**Message Types:**
```json
// Agent status update
{
  "type": "agent_status",
  "agentId": "agent123",
  "status": "processing",
  "task": "analyzing_document_section_3",
  "progress": 0.65,
  "timestamp": "2024-01-01T12:00:00Z"
}

// Swarm coordination event
{
  "type": "swarm_coordination",
  "swarmId": "swarm456",
  "event": "knowledge_sharing",
  "participants": ["agent123", "agent456", "agent789"],
  "sharedKnowledge": "pattern_recognition_insights"
}
```

## OpenAPI Specifications

### Enhanced Chat API Specification

```yaml
openapi: 3.0.3
info:
  title: AgentForge Enhanced Chat API
  description: Primary conversational interface for AgentForge AGI platform
  version: 2.0.0
  contact:
    name: AgentForge Support
    email: support@agentforge.ai
    url: https://docs.agentforge.ai
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.agentforge.ai/v1
    description: Production server
  - url: https://staging-api.agentforge.ai/v1
    description: Staging server
  - url: http://localhost:8000/v1
    description: Development server

security:
  - BearerAuth: []
  - ApiKeyAuth: []

paths:
  /chat/message:
    post:
      summary: Process conversational message
      description: Process a message with AGI capabilities and agent coordination
      operationId: processChatMessage
      tags:
        - Chat
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatMessageRequest'
            examples:
              simple_query:
                summary: Simple question
                value:
                  message: "What is artificial intelligence?"
                  context:
                    userId: "user123"
                    sessionId: "session456"
              data_analysis:
                summary: Data analysis request
                value:
                  message: "Analyze this sales data and create insights"
                  context:
                    userId: "user123"
                    sessionId: "session456"
                    dataSources:
                      - type: "file"
                        name: "sales_data.csv"
                        url: "https://example.com/data.csv"
                  capabilities: ["data_analysis", "visualization"]
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessageResponse'
        '400':
          description: Bad request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '429':
          description: Rate limit exceeded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /system/status:
    get:
      summary: Get system status
      description: Retrieve comprehensive system status and health information
      operationId: getSystemStatus
      tags:
        - System
      responses:
        '200':
          description: System status information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemStatus'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    ChatMessageRequest:
      type: object
      required:
        - message
        - context
      properties:
        message:
          type: string
          description: User message content
          example: "Analyze this dataset and create a dashboard"
        context:
          $ref: '#/components/schemas/ChatContext'
        capabilities:
          type: array
          items:
            type: string
          description: Requested capabilities
          example: ["data_analysis", "visualization"]

    ChatContext:
      type: object
      required:
        - userId
        - sessionId
      properties:
        userId:
          type: string
          description: Unique user identifier
        sessionId:
          type: string
          description: Unique session identifier
        conversationHistory:
          type: array
          items:
            $ref: '#/components/schemas/Message'
        dataSources:
          type: array
          items:
            $ref: '#/components/schemas/DataSource'
        userPreferences:
          type: object
          additionalProperties: true

    Message:
      type: object
      properties:
        role:
          type: string
          enum: [user, assistant]
        content:
          type: string
        timestamp:
          type: string
          format: date-time

    DataSource:
      type: object
      properties:
        type:
          type: string
          enum: [file, url, database, stream]
        name:
          type: string
        url:
          type: string
          format: uri

    ChatMessageResponse:
      type: object
      properties:
        response:
          type: string
          description: AI response content
        agentsDeployed:
          type: integer
          description: Number of agents deployed for this request
        processingTime:
          type: number
          description: Processing time in seconds
        confidence:
          type: number
          minimum: 0
          maximum: 1
          description: Response confidence score
        capabilities:
          type: array
          items:
            type: string
          description: Capabilities used in response
        artifacts:
          type: array
          items:
            $ref: '#/components/schemas/Artifact'
        metadata:
          type: object
          properties:
            sessionId:
              type: string
            messageId:
              type: string
            timestamp:
              type: string
              format: date-time

    Artifact:
      type: object
      properties:
        type:
          type: string
          enum: [dashboard, report, application, visualization]
        url:
          type: string
          format: uri
        title:
          type: string
        description:
          type: string

    SystemStatus:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        timestamp:
          type: number
          format: unix-timestamp
        version:
          type: string
        services:
          type: object
          properties:
            core_services:
              type: object
              additionalProperties:
                type: boolean
            ai_services:
              type: object
              additionalProperties:
                type: boolean
            advanced_services:
              type: object
              additionalProperties:
                type: boolean
        metrics:
          type: object
          properties:
            active_agents:
              type: integer
            total_deployments:
              type: integer
            avg_response_time:
              type: number
            system_load:
              type: number

    Error:
      type: object
      properties:
        error:
          type: string
          description: Error message
        code:
          type: string
          description: Error code
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
```

## Error Handling

### Standard Error Format

All APIs return errors in a consistent format:

```json
{
  "error": "Validation failed",
  "code": "VALIDATION_ERROR",
  "details": {
    "field": "message",
    "issue": "Message cannot be empty"
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "requestId": "req123"
}
```

### HTTP Status Codes

- **200 OK** - Request successful
- **201 Created** - Resource created successfully
- **400 Bad Request** - Invalid request parameters
- **401 Unauthorized** - Authentication required
- **403 Forbidden** - Insufficient permissions
- **404 Not Found** - Resource not found
- **409 Conflict** - Resource conflict
- **429 Too Many Requests** - Rate limit exceeded
- **500 Internal Server Error** - Server error
- **502 Bad Gateway** - Upstream service error
- **503 Service Unavailable** - Service temporarily unavailable

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_ERROR` | Authentication failed |
| `AUTHORIZATION_ERROR` | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded |
| `AGENT_DEPLOYMENT_FAILED` | Agent deployment failed |
| `PROCESSING_TIMEOUT` | Request processing timeout |
| `INSUFFICIENT_RESOURCES` | Insufficient system resources |
| `SERVICE_UNAVAILABLE` | Required service unavailable |

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1704110460
X-RateLimit-Window: 3600
```

### Rate Limits by Tier

| Tier | Requests/Hour | Agent Deployments/Hour | Concurrent Agents |
|------|---------------|------------------------|-------------------|
| **Free** | 100 | 10 | 5 |
| **Pro** | 10,000 | 1,000 | 100 |
| **Enterprise** | 100,000 | 10,000 | 10,000 |
| **Custom** | Unlimited | Unlimited | Unlimited |

### Rate Limit Exceeded Response

```json
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "limit": 1000,
    "window": 3600,
    "retryAfter": 1800
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## SDK and Client Libraries

### Python SDK

```python
from agentforge import AgentForgeClient

# Initialize client
client = AgentForgeClient(
    api_key="your-api-key",
    base_url="https://api.agentforge.ai"
)

# Send chat message
response = await client.chat.send_message(
    message="Analyze this data",
    context={
        "userId": "user123",
        "sessionId": "session456"
    },
    capabilities=["data_analysis"]
)

print(f"Response: {response.message}")
print(f"Agents deployed: {response.agents_deployed}")

# Deploy agent swarm
deployment = await client.agents.deploy(
    objective="Security analysis",
    scale="large_swarm",
    specializations=["security_analysis", "vulnerability_scanning"]
)

print(f"Deployment ID: {deployment.deployment_id}")
print(f"Agents deployed: {deployment.agents_deployed}")
```

### JavaScript/TypeScript SDK

```typescript
import { AgentForgeClient } from '@agentforge/sdk';

// Initialize client
const client = new AgentForgeClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.agentforge.ai'
});

// Send chat message
const response = await client.chat.sendMessage({
  message: 'Create a dashboard from this data',
  context: {
    userId: 'user123',
    sessionId: 'session456',
    dataSources: [{
      type: 'file',
      name: 'data.csv',
      url: 'https://example.com/data.csv'
    }]
  },
  capabilities: ['data_analysis', 'visualization']
});

console.log(`Response: ${response.message}`);
console.log(`Agents deployed: ${response.agentsDeployed}`);

// Real-time dashboard updates
const ws = client.dashboard.connect('dashboard123');

ws.on('update', (data) => {
  console.log('Dashboard update:', data);
});
```

### cURL Examples

```bash
# Chat message
curl -X POST https://api.agentforge.ai/v1/chat/message \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze network security logs",
    "context": {
      "userId": "user123",
      "sessionId": "session456"
    },
    "capabilities": ["security_analysis", "log_analysis"]
  }'

# Deploy agents
curl -X POST https://api.agentforge.ai/v1/agents/deploy \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Code quality analysis",
    "scale": "medium_swarm",
    "specializations": ["code_review", "quality_assessment"],
    "dataSources": [{
      "type": "repository",
      "url": "https://github.com/company/repo"
    }]
  }'

# System status
curl -X GET https://api.agentforge.ai/v1/system/status \
  -H "Authorization: Bearer YOUR_API_KEY"
```

This comprehensive API documentation provides detailed information about all AgentForge APIs, including request/response formats, authentication methods, error handling, and client SDK usage examples. The documentation follows OpenAPI 3.0 standards and includes practical examples for integration.
