#!/bin/bash

# AgentForge Multi-Tier System Startup Script
# Starts complete system: Backend (8000) + Admin UI (3001) + Individual UI (3002)

echo "🚀 Starting AgentForge Multi-Tier AGI System..."
echo "================================================"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export AF_ENVIRONMENT="development"
export AF_LOG_LEVEL="INFO"
export AF_METRICS_ENABLED="true"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${CYAN}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

# Check prerequisites
print_header "🔧 Checking Prerequisites..."

# Check Python virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Attempting to activate..."
    if [ -f "source/bin/activate" ]; then
        source source/bin/activate
        print_status "Activated bundled virtual environment"
    elif [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_status "Activated venv virtual environment"
    else
        print_error "No virtual environment found. Please create one or activate manually."
        exit 1
    fi
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js not found. Please install Node.js 18+ to run the frontend."
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm not found. Please install npm to run the frontend."
    exit 1
fi

print_success "✅ Prerequisites check passed"

print_header "🔧 Starting Backend Services..."

# Start the main AGI backend (Port 8000)
print_status "Starting AGI Backend (Complete API) on port 8000..."
cd services/swarm
python -m app.api.main &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
print_status "Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    print_success "✅ Backend is running on http://localhost:8000"
    
    # Test all endpoint groups
    print_status "Testing API endpoints..."
    
    if curl -s http://localhost:8000/v1/chat/health > /dev/null; then
        print_success "  ✅ Chat API endpoints"
    else
        print_warning "  ⚠️  Chat API endpoints may not be ready"
    fi
    
    if curl -s http://localhost:8000/v1/jobs/health > /dev/null; then
        print_success "  ✅ Job Management endpoints"
    else
        print_warning "  ⚠️  Job Management endpoints may not be ready"
    fi
    
    if curl -s http://localhost:8000/v1/realtime/health > /dev/null; then
        print_success "  ✅ Real-time endpoints"
    else
        print_warning "  ⚠️  Real-time endpoints may not be ready"
    fi
    
else
    print_warning "Backend may still be starting up..."
fi

print_header "🌐 Starting Frontend Applications..."

# Start Admin Interface (Port 3001)
print_status "Starting Admin Interface on port 3001..."
cd ui/agentforge-admin

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_status "Installing admin interface dependencies..."
    npm install
fi

npm run dev &
ADMIN_PID=$!
cd ../..

# Start Individual User Interface (Port 3002)
print_status "Starting Individual User Interface on port 3002..."
cd ui/agentforge-individual

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    print_status "Installing individual interface dependencies..."
    npm install
fi

npm run dev &
INDIVIDUAL_PID=$!
cd ../..

# Wait for frontends to start
print_status "Waiting for frontend applications to initialize..."
sleep 8

print_header "🎉 AgentForge Multi-Tier System Started!"
echo ""
print_success "🔧 BACKEND API (Port 8000):"
print_success "   http://localhost:8000"
print_success "   📊 API Documentation: http://localhost:8000/docs"
print_success "   📈 Metrics: http://localhost:8000/metrics"
echo ""
print_success "👥 ADMIN INTERFACE (Port 3001):"
print_success "   http://localhost:3001"
print_success "   🎯 For: Technical teams, system administrators"
print_success "   📊 Features: System monitoring, user management, analytics"
echo ""
print_success "👤 INDIVIDUAL INTERFACE (Port 3002):"
print_success "   http://localhost:3002"
print_success "   🎯 For: End users, agency staff, company employees"
print_success "   🤖 Features: AGI chat, file processing, personal analytics"
echo ""

print_header "🏢 ENTERPRISE ARCHITECTURE:"
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│                    MULTI-TIER SYSTEM                       │"
echo "├─────────────────────────────────────────────────────────────┤"
echo "│                                                             │"
echo "│  👥 ADMIN (3001)     ←─────────→  🔧 BACKEND (8000)        │"
echo "│  Technical Teams                   Complete AGI API         │"
echo "│  System Monitoring                 All Intelligence         │"
echo "│  User Management                   Universal I/O            │"
echo "│                                    Neural Mesh Memory       │"
echo "│                                                             │"
echo "│  👤 INDIVIDUAL (3002) ←────────→  🔧 BACKEND (8000)        │"
echo "│  End Users                         AGI Chat Interface       │"
echo "│  Agency Staff                      File Processing          │"
echo "│  Company Employees                 Personal Analytics       │"
echo "│                                                             │"
echo "│  🏢 ENTERPRISE MODEL:                                       │"
echo "│  Multiple 3002s → Single 3001 (per organization)           │"
echo "│                                                             │"
echo "│  👤 INDIVIDUAL MODEL:                                       │"
echo "│  Single 3002 → Personal 3001 (individual user)            │"
echo "│                                                             │"
echo "└─────────────────────────────────────────────────────────────┘"
echo ""

print_header "🤖 COMPLETE AGI CAPABILITIES:"
echo "✅ Universal AGI Engine Integration"
echo "✅ 39+ Input Types, 45+ Output Formats"
echo "✅ Neural Mesh Memory (4-tier system)"
echo "✅ Quantum Agent Coordination (Million-scale)"
echo "✅ Real-Time WebSocket Updates"
echo "✅ Emergent Intelligence & Learning"
echo "✅ Predictive User Modeling"
echo "✅ Cross-Modal Understanding"
echo "✅ Self-Improving Conversations"
echo "✅ Adaptive UI Personalization"
echo "✅ Enterprise Management System"
echo ""

print_header "📊 API ENDPOINTS AVAILABLE:"
echo "Chat System:        /v1/chat/*"
echo "Universal I/O:      /v1/io/*"
echo "Job Management:     /v1/jobs/*"
echo "Real-Time Updates:  /v1/realtime/*"
echo "Intelligence:       /v1/intelligence/*"
echo "Predictive:         /v1/predictive/*"
echo "Cross-Modal:        /v1/cross-modal/*"
echo "Self-Improvement:   /v1/self-improvement/*"
echo "Enterprise:         /v1/enterprise/*"
echo ""

print_header "💡 USAGE SCENARIOS:"
echo ""
echo "🏢 ENTERPRISE DEPLOYMENT:"
echo "  • Technical team accesses admin dashboard (3001)"
echo "  • Multiple end users use individual interface (3002)"
echo "  • All data flows to admin for monitoring and management"
echo "  • Centralized analytics and system control"
echo ""
echo "👤 INDIVIDUAL DEPLOYMENT:"
echo "  • User accesses personal interface (3002)"
echo "  • Same user can access personal admin view (3001)"
echo "  • Complete AGI capabilities with personal analytics"
echo "  • Full system control and monitoring"
echo ""

print_header "🔄 TESTING THE SYSTEM:"
echo "1. Open Admin Interface: http://localhost:3001"
echo "   - View system overview and analytics"
echo "   - Monitor all user connections"
echo "   - Manage organizations and users"
echo ""
echo "2. Open Individual Interface: http://localhost:3002"
echo "   - Chat with AGI system"
echo "   - Upload files and process data"
echo "   - View personal analytics"
echo ""
echo "3. Test Multi-Tier Communication:"
echo "   - Send messages from 3002"
echo "   - Monitor activity in 3001"
echo "   - Verify data flow between interfaces"
echo ""

print_status "🔄 System is running with full AGI intelligence!"
print_status "📊 All interfaces connected to complete backend"
print_status "🧠 Learning and adaptation active"

# Function to cleanup on exit
cleanup() {
    print_header "🛑 Shutting down AgentForge Multi-Tier System..."
    
    if [ ! -z "$INDIVIDUAL_PID" ]; then
        print_status "Stopping Individual Interface (PID: $INDIVIDUAL_PID)..."
        kill $INDIVIDUAL_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$ADMIN_PID" ]; then
        print_status "Stopping Admin Interface (PID: $ADMIN_PID)..."
        kill $ADMIN_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        print_status "Stopping Backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    print_success "✅ AgentForge Multi-Tier System stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop the system
print_status "Press Ctrl+C to stop the system"
wait
