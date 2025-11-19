#!/bin/bash

# AgentForge Services Startup Script
# This script starts all necessary services with proper paths

set -e

echo "🚀 Starting AgentForge Services..."

# Ensure we're in the project root
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

echo "📁 Project root: $PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Start Docker services
echo "🐳 Starting Docker services (postgres, redis, nats)..."
docker-compose up -d postgres redis nats

# Wait for services to be ready
echo "⏳ Waiting for Docker services to be ready..."
sleep 5

# Start main backend (optional)
# echo "🔧 Starting main backend..."
# python main.py &
# MAIN_PID=$!

# Start enhanced chat API from project root
echo "💬 Starting Enhanced Chat API..."
python apis/enhanced_chat_api.py &
API_PID=$!

# Give backend time to start
sleep 3

echo ""
echo "✅ Backend services started!"
echo "🌐 Backend API: http://0.0.0.0:8000"
echo "📊 Metrics: http://0.0.0.0:8000/metrics"
echo "🔍 Health: http://0.0.0.0:8000/live"
echo ""
echo "🎨 Starting Frontend..."
echo ""

# Start frontend
cd ui/agentforge-individual
npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_ROOT"

echo ""
echo "✨ All services started!"
echo ""
echo "📊 Admin Dashboard: http://localhost:3001"
echo "👤 Individual Frontend: http://localhost:3002"
echo "🌐 Backend API: http://0.0.0.0:8000"
echo ""
echo "Press Ctrl+C to stop all services..."

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    # kill $MAIN_PID 2>/dev/null || true
    kill $API_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    docker-compose down
    echo "✅ All services stopped"
    exit 0
}

trap cleanup INT TERM

# Wait for services
wait

