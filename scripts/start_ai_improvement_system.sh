#!/bin/bash

# Start Self-Evolving AGI System for AgentForge
# This script starts the complete self-improving AGI system

echo "🧠 Starting AgentForge Self-Evolving AGI System..."

# Kill any existing processes
echo "🔧 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true  
lsof -ti:3002 | xargs kill -9 2>/dev/null || true

sleep 3

# Set up environment
cd /Users/baileymahoney/AgentForge
export $(cat .env | xargs)

# Start Self-Evolving AGI Backend (Port 8000)
echo "🚀 Starting Self-Evolving AGI Backend on port 8000..."
python agi_chat_api.py &
BACKEND_PID=$!

sleep 5

# Check if backend started successfully
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Self-Evolving AGI Backend started successfully"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start Admin Interface (Port 3001)  
echo "🔧 Starting Admin Interface on port 3001..."
cd ui/agentforge-admin-dashboard
npm run dev &
ADMIN_PID=$!

sleep 3

# Start Individual User Interface (Port 3002)
echo "👤 Starting Individual User Interface on port 3002..."
cd ../agentforge-individual
npm run dev &
USER_PID=$!

sleep 5

echo ""
echo "🎉 Self-Evolving AGI System Started Successfully!"
echo ""
echo "📊 Admin Interface:      http://localhost:3001"
echo "👤 Individual Interface: http://localhost:3002" 
echo "🧠 AGI Backend:          http://localhost:8000"
echo ""
echo "🌟 BREAKTHROUGH FEATURES:"
echo "   ✅ Real self-analysis of agent weaknesses"
echo "   ✅ Live improvement implementation"
echo "   ✅ Automatic code generation for fixes"
echo "   ✅ Real-time capability enhancement"
echo "   ✅ Performance tracking and measurement"
echo ""
echo "💡 Ask the AGI: 'What are my agents missing?' to see live self-improvement!"
echo ""
echo "Process IDs:"
echo "   Backend: $BACKEND_PID"
echo "   Admin:   $ADMIN_PID" 
echo "   User:    $USER_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "🛑 Stopping all services..."; kill $BACKEND_PID $ADMIN_PID $USER_PID 2>/dev/null; exit 0' INT
wait
