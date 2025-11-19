#!/bin/bash

echo "Starting AgentForge System..."

# Kill any existing processes
pkill -f uvicorn 2>/dev/null
pkill -f "next dev" 2>/dev/null
sleep 2

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing dependencies..."
pip install fastapi uvicorn python-multipart --quiet

# Start backend
echo "Starting backend on port 8000..."
cd /Users/baileymahoney/AgentForge
python -m uvicorn apis.enhanced_chat_api:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend on port 3002..."
cd /Users/baileymahoney/AgentForge/ui/agentforge-individual
npm run dev &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "System starting..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3002"

# Wait for user input
read -p "Press Enter to stop services..."

# Clean shutdown
kill $BACKEND_PID 2>/dev/null
kill $FRONTEND_PID 2>/dev/null
echo "Services stopped."
