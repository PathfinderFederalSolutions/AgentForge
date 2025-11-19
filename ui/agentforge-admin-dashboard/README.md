# AgentForge Frontend

A cutting-edge, production-ready frontend for the AgentForge AI agent orchestration platform.

## 🚀 Features

### Core Platform
- **Modern React/Next.js Architecture** - Built with Next.js 14, TypeScript, and Tailwind CSS
- **Real-time Updates** - WebSocket integration for live data streaming
- **Responsive Design** - Optimized for desktop, tablet, and mobile devices
- **Dark/Light Themes** - Tactical blue (day) and dark red (night) themes
- **Advanced Animations** - Smooth Framer Motion animations throughout

### Complete Application Suite
1. **📊 Dashboard** - Mission control with real-time metrics and system overview
2. **📖 Instructions** - Comprehensive user guide and feature documentation
3. **🤖 Agent Management** - Deploy, monitor, and control AI agents
4. **💼 Job Management** - Create, track, and manage computational jobs
5. **👥 Swarm Control** - Coordinate distributed agent clusters
6. **🧠 Neural Mesh** - AI memory system with semantic search
7. **⚛️ Quantum Scheduler** - Million-scale agent coordination
8. **📈 Monitoring** - System health and performance monitoring
9. **📊 Analytics** - Performance insights and AI-generated recommendations
10. **🌐 Network** - Network topology and connection monitoring
11. **💻 Terminal** - Command-line interface for system management
12. **💾 Data** - Data source and pipeline management
13. **🔒 Security** - Security monitoring and access control
14. **⚙️ Settings** - System configuration and administration

### Advanced Features
- **Real-time WebSocket Integration** - Live updates across all interfaces
- **Comprehensive API Integration** - Full backend service connectivity
- **Advanced Search & Filtering** - Intelligent search across all data
- **Export/Import Capabilities** - Data export and configuration management
- **Mobile-Responsive Design** - Full functionality on all devices
- **Keyboard Navigation** - Power-user keyboard shortcuts
- **Error Boundaries** - Graceful error handling and recovery

## 🎨 Design System

### Theme Configuration
The application uses a sophisticated dual-theme system:

**Day Theme (Tactical Blue)**
- Background: `#05080D`
- Text: `#D6E2F0`
- Accent: `#00A39B`
- Grid: `#0E1622`
- Lines: `#0F2237`

**Night Theme (Dark Red)**
- Background: `#000000`
- Text: `#FF2B2B`
- Accent: `#FF2B2B`
- Grid: `#1a0000`
- Lines: `#891616`

### Component Library
- **Button** - Multiple variants with loading states and animations
- **Card** - Glass morphism cards with hover effects
- **Input** - Enhanced inputs with icons and validation
- **Badge** - Status indicators with semantic colors
- **Modal** - Animated modals with backdrop blur
- **Layout** - Responsive sidebar and header layout

## 🔧 Development

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Modern browser with WebSocket support

### Installation
```bash
cd apps/agent-swarm-frontend
npm install
```

### Development Server
```bash
npm run dev
# Server runs on http://localhost:3000
```

### Build for Production
```bash
npm run build
npm start
```

### Environment Variables
Create a `.env.local` file:
```env
NEXT_PUBLIC_WS_URL=ws://your-websocket-server:8080/ws
NEXT_PUBLIC_AGENT_API_BASE=http://your-api-server:8000
```

## 🔌 Backend Integration

### API Endpoints
The frontend integrates with multiple backend services:

- **Orchestrator API** - `/v1/*` - Job orchestration and system control
- **Swarm API** - `/api/v1/swarm/*` - Agent and job management
- **Neural Mesh API** - `/api/v1/neural-mesh/*` - Memory and knowledge management
- **Quantum API** - `/api/v1/quantum/*` - Quantum scheduler operations
- **WebSocket** - Real-time updates and live data streaming

### Health Checks
All services implement health check endpoints:
- `/health` - Basic health status
- `/ready` - Readiness for traffic
- `/startup` - Startup completion status

## 📱 User Interface Guide

### Navigation
- **Collapsible Sidebar** - Main navigation with 14 sections
- **Search Bar** - Global search across agents, jobs, and commands
- **Notifications** - Real-time system alerts and updates
- **User Menu** - Profile and system settings access

### Key Features by Page

#### Dashboard
- Real-time system metrics and KPIs
- System health monitoring with component status
- Resource usage tracking (CPU, Memory, GPU, Network)
- Recent activity feed and quick actions

#### Agent Management
- Agent fleet overview with real-time status
- Resource monitoring per agent (CPU, Memory, GPU)
- Agent deployment with configuration options
- Agent lifecycle management (start, stop, restart)

#### Job Management  
- Complete job pipeline from creation to completion
- Real-time progress tracking and resource usage
- Advanced filtering and search capabilities
- Job analytics and performance metrics

#### Neural Mesh
- Memory network visualization with 15,000+ nodes
- Semantic search and knowledge retrieval
- Memory type analysis and distribution
- Content processing and pattern recognition

#### Quantum Scheduler
- Million-scale agent coordination interface
- Quantum task creation and management
- Coherence level monitoring and optimization
- Entanglement and efficiency tracking

#### Monitoring
- Comprehensive system health dashboard
- Real-time performance metrics and trends
- Alert management and incident tracking
- Service status and dependency monitoring

#### Analytics
- Performance insights and trend analysis
- AI-generated recommendations
- Usage pattern recognition
- Cost and efficiency optimization

#### Security
- Security alert monitoring and management
- Access control and API key management
- Audit log tracking and analysis
- Threat detection and response

## 🚀 Deployment

### Docker Build
```bash
docker build -t agentforge-frontend:latest .
docker run -p 3000:3000 agentforge-frontend:latest
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentforge-frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agentforge-frontend
  template:
    metadata:
      labels:
        app: agentforge-frontend
    spec:
      containers:
      - name: frontend
        image: agentforge-frontend:latest
        ports:
        - containerPort: 3000
        env:
        - name: NEXT_PUBLIC_WS_URL
          value: "ws://agentforge-api:8080/ws"
        - name: NEXT_PUBLIC_AGENT_API_BASE
          value: "http://agentforge-api:8000"
```

## 🎯 Performance

### Optimization Features
- **Code Splitting** - Automatic route-based code splitting
- **Image Optimization** - Next.js automatic image optimization
- **Bundle Analysis** - Optimized bundle sizes (average 283kB)
- **Caching Strategy** - Intelligent API response caching
- **Lazy Loading** - Dynamic imports for better performance

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🧪 Testing

### Manual Testing Checklist
- [ ] All pages load without errors
- [ ] Navigation works correctly
- [ ] WebSocket connection handles failures gracefully
- [ ] Theme switching works properly
- [ ] Mobile responsiveness verified
- [ ] All interactive elements function correctly

### End-to-End Testing
1. **Dashboard** - Verify metrics display and real-time updates
2. **Agent Management** - Test agent deployment and monitoring
3. **Job Management** - Create, monitor, and manage jobs
4. **Neural Mesh** - Test memory search and visualization
5. **Quantum Scheduler** - Create and monitor quantum tasks
6. **Monitoring** - Verify alert system and health checks
7. **Settings** - Test configuration changes and persistence

## 📚 Additional Resources

- **User Guide** - Access the `/instructions` page for comprehensive documentation
- **API Documentation** - See backend service documentation
- **Component Library** - Reference `src/components/ui/` for reusable components
- **State Management** - See `src/lib/state.ts` for application state

## 🔧 Troubleshooting

### Common Issues

**WebSocket Connection Errors**
- Check if backend WebSocket server is running
- Verify WebSocket URL in environment variables
- Application works with mock data if WebSocket unavailable

**Build Errors**
- Ensure all dependencies are installed: `npm install`
- Clear Next.js cache: `rm -rf .next`
- Check TypeScript errors: `npm run build`

**Performance Issues**
- Enable production mode: `npm run build && npm start`
- Check browser developer tools for console errors
- Verify network connectivity to backend services

---

**Built with ❤️ for AgentForge - The Future of AI Agent Orchestration**

