#!/bin/bash

# Clean restart script - stops everything and starts fresh
# Use this after fixing issues to ensure clean startup

echo "🛑 Stopping all AgentForge services..."

# Kill any running Python APIs
pkill -f "python apis/enhanced_chat_api.py" 2>/dev/null
pkill -f "python main.py" 2>/dev/null

# Kill any running frontend
pkill -f "npm run dev" 2>/dev/null
pkill -f "next dev" 2>/dev/null

# Stop Docker services
docker-compose down

echo "✅ All services stopped"
echo ""
echo "🧹 Cleaning up..."

# Clean Next.js cache if it exists
if [ -d "ui/agentforge-individual/.next" ]; then
    echo "Cleaning Next.js cache..."
    rm -rf ui/agentforge-individual/.next
fi

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "✅ Cleanup complete"
echo ""
echo "⏳ Waiting 3 seconds before restart..."
sleep 3
echo ""
echo "🚀 Starting services..."
echo ""

# Start everything fresh
./start_services.sh

