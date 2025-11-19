# AgentForge Quick Start Guide

## Issues Fixed

### 1. Frontend Build Error
- **Problem**: Merge conflict in `store.ts` file causing TypeScript syntax error
- **Solution**: Resolved merge conflict in DataSource type definition

### 2. Backend Import Errors  
- **Problem**: Running API from `apis/` directory prevented importing `core/`, `services/`, and `libs/` modules
- **Solution**: Created startup script that runs API from project root with proper PYTHONPATH

## Starting All Services

### Option 1: Use the Startup Script (Recommended)

```bash
# From the project root
./start_services.sh
```

This will:
1. Start Docker services (postgres, redis, nats)
2. Activate virtual environment
3. Start Enhanced Chat API from project root
4. Start Individual Frontend on port 3002

Press Ctrl+C to stop all services cleanly.

### Option 2: Manual Startup

```bash
# 1. Start Docker services
docker-compose up -d postgres redis nats

# 2. Activate virtual environment
source venv/bin/activate

# 3. Set PYTHONPATH and start API from project root
export PYTHONPATH="$(pwd):$PYTHONPATH"
python apis/enhanced_chat_api.py &

# 4. Start frontend
cd ui/agentforge-individual
npm run dev
```

## Access Points

- **Individual Frontend**: http://localhost:3002
- **Admin Dashboard**: http://localhost:3001 (if running)
- **Backend API**: http://0.0.0.0:8000
- **API Docs**: http://0.0.0.0:8000/docs
- **Health Check**: http://0.0.0.0:8000/live
- **Metrics**: http://0.0.0.0:8000/metrics

## Backend Services Status

The backend will attempt to load all available services:

### Core Services
- Neural Mesh Coordinator
- Enhanced Neural Mesh
- Self-Coding AGI
- AGI Evolution

### Advanced Services
- Quantum Scheduler
- Universal I/O
- Mega-Swarm Coordinator
- Self-Bootstrap
- Security Orchestrator
- Agent Lifecycle
- Advanced Fusion

### Infrastructure
- Enhanced Logging
- Database Manager
- Retry Handler
- Request Pipeline

### Libraries
- AF-Common (Types, Logging, Config, Tracing)
- AF-Schemas (Agent & Event Schemas)
- AF-Messaging (NATS Integration)

**Note**: Services will show ❌ if their modules can't be imported. The API will still work with basic LLM features even if advanced services are unavailable.

## Troubleshooting

### Frontend won't build
- Check for merge conflicts in TypeScript files
- Run `npm install` in `ui/agentforge-individual`
- Clear Next.js cache: `rm -rf .next`

### Backend import errors
- Ensure you're running from project root
- Set PYTHONPATH: `export PYTHONPATH="$(pwd):$PYTHONPATH"`
- Verify virtual environment is activated
- Check that all dependencies are installed: `pip install -r config/requirements.txt`

### Docker services won't start
- Check if ports are already in use
- Verify Docker is running: `docker ps`
- Check logs: `docker-compose logs`

### Services show as unavailable (❌)
- This is expected if modules aren't installed or have import errors
- The API will still work with basic LLM functionality
- To enable all features, ensure all Python dependencies are installed
- Check the specific module import error in terminal output

## Next Steps

1. Open http://localhost:3002 in your browser
2. Try the AI chat interface
3. Upload data sources
4. Create intelligent agent swarms
5. Monitor agent activity in real-time

## Features Available

✅ Multi-LLM Support (OpenAI, Anthropic, xAI)
✅ Real-time Agent Swarm Deployment
✅ Intelligent Job Management
✅ Data Source Integration
✅ Project Management
✅ SSE Streaming
✅ Health & Metrics Monitoring

