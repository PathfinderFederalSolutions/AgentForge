#!/bin/bash

echo "Testing AgentForge System..."

echo "=== Backend Health Check ==="
curl -s http://localhost:8000/health | head -10

echo -e "\n=== Frontend Status ==="
curl -s -I http://localhost:3002 | head -1

echo -e "\n=== Upload Endpoint Test ==="
curl -s -X POST http://localhost:8000/v1/io/upload \
  -F "files=@/dev/null" | head -5

echo -e "\n=== Chat API Test ==="
curl -s -X POST http://localhost:8000/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "test",
    "context": {
      "userId": "test_user",
      "sessionId": "test_session",
      "dataSources": []
    }
  }' | jq -r '.response' | head -5

echo -e "\nSystem test complete."
