#!/bin/bash

# AgentForge System Status Checker
# Verifies all components are running properly

echo "🔍 AgentForge Multi-Tier System Status Check"
echo "============================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

check_service() {
    local name="$1"
    local url="$2"
    local port="$3"
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name${NC} - Running on port $port"
        return 0
    else
        echo -e "${RED}❌ $name${NC} - Not responding on port $port"
        return 1
    fi
}

echo ""
echo "🔧 Backend Services:"
check_service "AGI Backend API" "http://localhost:8000/health" "8000"

echo ""
echo "🌐 Frontend Interfaces:"
check_service "Admin Interface" "http://localhost:3001" "3001"
check_service "Individual Interface" "http://localhost:3002" "3002"

echo ""
echo "📊 API Endpoint Tests:"
if curl -s "http://localhost:8000/v1/chat/capabilities" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Chat API${NC} - Endpoints responding"
else
    echo -e "${YELLOW}⚠️ Chat API${NC} - May not be fully loaded"
fi

echo ""
echo "🌐 Access Points:"
echo -e "${BLUE}👥 Admin Dashboard:${NC}     http://localhost:3001"
echo -e "${BLUE}👤 Individual Chat:${NC}     http://localhost:3002" 
echo -e "${BLUE}🔧 Backend API:${NC}         http://localhost:8000"
echo -e "${BLUE}📊 API Documentation:${NC}   http://localhost:8000/docs"

echo ""
echo "🎯 System Architecture:"
echo "┌─────────────────────────────────────────────┐"
echo "│  👥 Admin (3001) ←──→ 🔧 Backend (8000)     │"
echo "│  👤 Individual (3002) ←──→ 🔧 Backend (8000) │"
echo "└─────────────────────────────────────────────┘"

echo ""
echo "💡 Next Steps:"
echo "1. Open http://localhost:3001 for admin dashboard"
echo "2. Open http://localhost:3002 for AGI chat interface"
echo "3. Test the complete AGI functionality"
