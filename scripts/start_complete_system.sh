#!/bin/bash

# AgentForge Complete System Startup Script
# Starts all components for full AGI chat integration

echo "🚀 Starting AgentForge Complete AGI System..."
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

# Check if Python virtual environment is activated
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

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js not found. Please install Node.js 18+ to run the frontend."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    print_error "npm not found. Please install npm to run the frontend."
    exit 1
fi

print_header "🔧 Starting Backend Services..."

# Start the main AGI backend (Swarm Gateway with all endpoints)
print_status "Starting AGI Backend (Swarm Gateway) on port 8000..."
cd services/swarm
python -m app.api.main &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    print_status "✅ Backend is running on http://localhost:8000"
else
    print_warning "Backend may still be starting up..."
fi

print_header "🌐 Starting Frontend Application..."

# Install frontend dependencies if needed
cd ui/agentforge-user
if [ ! -d "node_modules" ]; then
    print_status "Installing frontend dependencies..."
    npm install
fi

# Start the frontend on port 3001
print_status "Starting Frontend on port 3001..."
npm run dev &
FRONTEND_PID=$!
cd ../..

# Wait for frontend to start
sleep 5

print_header "🎉 AgentForge Complete System Started!"
echo ""
print_status "🌐 Frontend: http://localhost:3001"
print_status "🔧 Backend API: http://localhost:8000"
print_status "📊 API Documentation: http://localhost:8000/docs"
print_status "📈 Metrics: http://localhost:8000/metrics"
echo ""

print_header "🤖 Available API Endpoints:"
echo "Chat System:"
echo "  POST /v1/chat/message          - Process chat messages with full AGI"
echo "  POST /v1/chat/upload           - Upload files with Universal I/O"
echo "  GET  /v1/chat/capabilities     - Get available AGI capabilities"
echo "  WS   /v1/realtime/ws           - Real-time updates"
echo ""
echo "Universal I/O:"
echo "  POST /v1/io/upload             - Multi-file upload and processing"
echo "  POST /v1/io/generate           - Universal output generation"
echo "  GET  /v1/io/data-sources       - Data source management"
echo ""
echo "Job Management:"
echo "  POST /v1/jobs/create           - Create new jobs"
echo "  GET  /v1/jobs/active           - Get active jobs"
echo "  POST /v1/jobs/{id}/pause       - Pause/resume/archive jobs"
echo ""
echo "Phase 3 Intelligence:"
echo "  POST /v1/intelligence/analyze-interaction  - Pattern analysis"
echo "  POST /v1/predictive/update-profile        - User modeling"
echo "  POST /v1/cross-modal/analyze              - Multi-modal understanding"
echo "  POST /v1/self-improvement/analyze-quality - Quality optimization"
echo ""

print_header "🎯 System Features:"
echo "✅ Universal AGI Engine Integration"
echo "✅ 39+ Input Types, 45+ Output Formats"
echo "✅ Neural Mesh Memory (4-tier system)"
echo "✅ Quantum Agent Coordination"
echo "✅ Real-Time WebSocket Updates"
echo "✅ Emergent Intelligence & Learning"
echo "✅ Predictive User Modeling"
echo "✅ Cross-Modal Understanding"
echo "✅ Self-Improving Conversations"
echo "✅ Adaptive UI Personalization"
echo ""

print_header "💡 Usage Examples:"
echo "1. Upload any file type and ask for analysis"
echo "2. Request complex applications: 'Build me a task management app'"
echo "3. Ask for insights: 'What patterns do you see in my data?'"
echo "4. Try advanced features: 'Deploy quantum-coordinated analysis'"
echo "5. Monitor real-time: 'Set up continuous monitoring for anomalies'"
echo ""

print_status "🔄 System is learning from your interactions and will adapt over time!"
print_status "📊 View Advanced Analytics and Adaptive Features in the UI"

# Function to cleanup on exit
cleanup() {
    print_header "🛑 Shutting down AgentForge system..."
    
    if [ ! -z "$FRONTEND_PID" ]; then
        print_status "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        print_status "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    print_status "✅ AgentForge system stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop the system
print_status "Press Ctrl+C to stop the system"
wait
