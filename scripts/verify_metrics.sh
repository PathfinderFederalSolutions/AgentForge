#!/bin/bash

# AgentForge Metrics Verification Script
# Tests that all metrics are pulling correctly from backend

echo "🔍 AgentForge Metrics Verification"
echo "=================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

test_endpoint() {
    local name="$1"
    local url="$2"
    local expected_field="$3"
    
    echo -n "Testing $name... "
    
    response=$(curl -s "$url" 2>/dev/null)
    if [ $? -eq 0 ] && echo "$response" | grep -q "$expected_field"; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        echo "  Response: $response"
        return 1
    fi
}

echo ""
echo "🔧 Backend API Endpoints:"
test_endpoint "Main Health" "http://localhost:8000/health" "status"
test_endpoint "Chat Health" "http://localhost:8000/v1/chat/health" "status"
test_endpoint "Jobs Health" "http://localhost:8000/v1/jobs/health" "status"
test_endpoint "Real-time Health" "http://localhost:8000/v1/realtime/health" "status"
test_endpoint "Enterprise Health" "http://localhost:8000/v1/enterprise/health" "status"

echo ""
echo "📊 Data Endpoints:"
test_endpoint "Chat Capabilities" "http://localhost:8000/v1/chat/capabilities" "inputFormats"
test_endpoint "Active Jobs" "http://localhost:8000/v1/jobs/active" "id"
test_endpoint "Organizations" "http://localhost:8000/v1/enterprise/organizations" "total_organizations"
test_endpoint "Connections" "http://localhost:8000/v1/enterprise/connections" "summary"

echo ""
echo "🧠 Intelligence Endpoints:"
test_endpoint "Intelligence Status" "http://localhost:8000/v1/intelligence/status" "status"
test_endpoint "Predictive Status" "http://localhost:8000/v1/predictive/status" "status"
test_endpoint "System Config" "http://localhost:8000/v1/config" "environment"

echo ""
echo "📈 Sample Metrics Data:"
echo -e "${BLUE}Organizations:${NC}"
curl -s http://localhost:8000/v1/enterprise/organizations | jq '.organizations[0]' 2>/dev/null || curl -s http://localhost:8000/v1/enterprise/organizations

echo ""
echo -e "${BLUE}Connections Summary:${NC}"
curl -s http://localhost:8000/v1/enterprise/connections | jq '.summary' 2>/dev/null || curl -s http://localhost:8000/v1/enterprise/connections

echo ""
echo -e "${BLUE}Active Jobs:${NC}"
curl -s http://localhost:8000/v1/jobs/active | jq '.[0]' 2>/dev/null || curl -s http://localhost:8000/v1/jobs/active

echo ""
echo "🎯 Verification Complete!"
echo "All metrics should now be pulling from backend instead of mock data."
