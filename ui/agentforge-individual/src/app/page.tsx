'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Paperclip,
  Sparkles,
  Moon,
  Sun,
  Bot,
  User,
  HelpCircle,
  LogOut,
  Settings,
  ChevronDown,
  Database,
  Brain,
  CheckCircle,
  Clock,
  X,
  Folder,
  Target,
  Activity,
  Shield
} from 'lucide-react';
import { useSnapshot } from 'valtio';
import { store } from '@/lib/store';
import CapabilityShowcase from '@/components/CapabilityShowcase';
import RealtimeSuggestions, { CapabilitySuggestionBanner } from '@/components/RealtimeSuggestions';
import AdvancedAnalytics from '@/components/AdvancedAnalytics';
import AdaptiveInterface from '@/components/AdaptiveInterface';
import { simpleSync } from '@/lib/simpleSync';
import UploadModal from '@/components/UploadModal';
import JobsSidebar from '@/components/JobsSidebar';
import ProjectsSidebar from '@/components/ProjectsSidebar';
import COAWarGamePanel from '@/components/COAWarGamePanel';
import RealTimeIntelPanel from '@/components/RealTimeIntelPanel';
import QuickActionToolbar from '@/components/QuickActionToolbar';
import ReactMarkdown from 'react-markdown';
import EnhancedMarkdownRenderer from '../components/EnhancedMarkdownRenderer';
import IntelligenceDashboard from '@/components/IntelligenceDashboard';

export default function Home() {
  const [input, setInput] = useState('');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [showJobsSidebar, setShowJobsSidebar] = useState(false);
  const [showProjectsSidebar, setShowProjectsSidebar] = useState(false);
  const [showCOAWarGamePanel, setShowCOAWarGamePanel] = useState(false);
  const [showRealTimeIntelPanel, setShowRealTimeIntelPanel] = useState(false);
  const [showCapabilityShowcase, setShowCapabilityShowcase] = useState(false);
  const [showRealtimeSuggestions, setShowRealtimeSuggestions] = useState(false);
  const [showAdvancedAnalytics, setShowAdvancedAnalytics] = useState(false);
  const [showIntelDashboard, setShowIntelDashboard] = useState(false);
  const [includeIntelligence, setIncludeIntelligence] = useState(true);
  const [includePlanning, setIncludePlanning] = useState(false);
  const [includeCOAs, setIncludeCOAs] = useState(false);
  const [includeWargaming, setIncludeWargaming] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const snap = useSnapshot(store);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [snap.messages.length]);

  const theme = snap.theme === 'day' ? {
    bg: '#05080D',
    text: '#D6E2F0',
    textSecondary: '#B8C5D1', 
    accent: '#00A39B',
    neon: '#CCFF00',
    border: 'rgba(255,255,255,0.2)',
    cardBg: 'rgba(255,255,255,0.08)',
    grid: '#0E1622',
    lines: '#0F2237'
  } : {
    bg: '#000000',
    text: '#FF2B2B',
    textSecondary: '#891616',
    accent: '#FF2B2B',
    border: 'rgba(137,22,22,0.6)',
    cardBg: 'rgba(137,22,22,0.15)',
    grid: '#1a0000',
    lines: '#891616'
  };

  // Auto-scroll to bottom with delay for content rendering
  useEffect(() => {
    const scrollToBottom = () => {
      messagesEndRef.current?.scrollIntoView({ 
        behavior: 'smooth',
        block: 'end',
        inline: 'nearest'
      });
    };
    
    // Small delay to ensure content is rendered
    setTimeout(scrollToBottom, 100);
  }, [snap.messages]);

  // Close user menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showUserMenu) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showUserMenu]);

  // Listen for enhanced analysis results
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    const handleEnhancedAnalysis = (event: CustomEvent) => {
      const { enhancedResponse, intelligenceAmplification, collectiveConfidence, agentsInvolved } = event.detail;
      
      // Add enhanced analysis as follow-up message
      const enhancedMessage = {
        id: `enhanced_${Date.now()}`,
        role: 'assistant' as const,
        content: `🧠 **Enhanced Collective Analysis** (${intelligenceAmplification}x amplification)\n\n${enhancedResponse}`,
        timestamp: new Date(),
        metadata: {
          agentsWorking: agentsInvolved,
          confidence: collectiveConfidence,
          capabilitiesUsed: ['collective_reasoning', 'swarm_intelligence']
        }
      };
      
      store.messages.push(enhancedMessage);
      
      // Update swarm activity to show completion
      store.swarmActivity = store.swarmActivity.map(activity => ({
        ...activity,
        status: 'completed' as const,
        progress: 100
      }));
    };
    
    window.addEventListener('enhancedAnalysisComplete', handleEnhancedAnalysis as EventListener);
    
    return () => {
      window.removeEventListener('enhancedAnalysisComplete', handleEnhancedAnalysis as EventListener);
    };
  }, []);

  // Initialize WebSocket connection (client-side only)
  useEffect(() => {
    // Only initialize in browser environment
    if (typeof window === 'undefined') return;
    
    // Initialize simple sync when component mounts (client-side only)
    console.log('Initializing sync service...');
    try {
      simpleSync.initialize();
    } catch (error) {
      console.log('Sync not available:', error);
    }
    
    // Initialize WebSocket connection for real-time updates
    console.log('Connecting to real-time updates...');
    try {
      store.agiClient.connectWebSocket();
    } catch (error) {
      console.log('WebSocket connection failed:', error);
    }
    
    // Subscribe to real-time updates
    store.agiClient.subscribeToUpdates('swarm_activity');
    store.agiClient.subscribeToUpdates('job_updates');
    store.agiClient.subscribeToUpdates('system_status');

    // Load real data from backend (clear mock data)
    store.loadBackendData();
    
    // Handle real-time updates
    store.agiClient.addEventListener('update', (data: any) => {
      if (data.type === 'swarm_update') {
        // Update swarm activity in real-time
        store.activeAgents = data.agents_active || store.activeAgents;
      } else if (data.type === 'job_update') {
        // Update job status in real-time
        const job = store.activeJobs.find(j => j.id === data.job_id);
        if (job) {
          job.progress = data.progress;
          job.agentsAssigned = data.agents_active;
          job.confidence = data.confidence;
        }
      } else if (data.type === 'system_status') {
        // Update system status
        console.log('System status update:', data);
      }
    });
    
    // Sync current state periodically
    const syncInterval = setInterval(() => {
      try {
        simpleSync.syncCurrentState();
      } catch (error) {
        console.log('Periodic sync error:', error);
      }
    }, 30000); // Sync every 30 seconds

    return () => {
      clearInterval(syncInterval);
      try {
        simpleSync.disconnect();
        store.agiClient.disconnectWebSocket();
      } catch (error) {
        console.log('Cleanup error:', error);
      }
    };
  }, []);

  const handleSend = () => {
    if (!input.trim() || snap.isTyping) return;

    // Auto-detect intelligence features from user input
    const inputText = input.trim().toLowerCase();

    // Intelligence feature detection patterns
    const intelligencePatterns = [
      'threat', 'attack', 'hostile', 'adversary', 'enemy', 'surveillance',
      'reconnaissance', 'intelligence', 'sigint', 'humint', 'geoint', 'osint',
      'cyber', 'security', 'breach', 'intrusion', 'malware', 'exploit'
    ];

    const planningPatterns = [
      'plan', 'strategy', 'respond', 'counter', 'defend', 'protect',
      'mitigate', 'prevent', 'prepare', 'coordinate', 'deploy'
    ];

    const coaPatterns = [
      'options', 'courses?.*action', 'coa', 'what.*do', 'alternatives',
      'scenarios', 'choices', 'decisions', 'recommendations'
    ];

    const wargamingPatterns = [
      'simulate', 'wargame', 'outcome', 'predict', 'battle', 'engagement',
      'conflict', 'warfare', 'combat', 'exercise', 'drill'
    ];

    // Auto-detect features
    const needsIntelligence = intelligencePatterns.some(pattern =>
      new RegExp(pattern, 'i').test(inputText)
    ) || snap.dataSources.length > 0;

    const needsPlanning = planningPatterns.some(pattern =>
      new RegExp(pattern, 'i').test(inputText)
    ) || needsIntelligence;

    const needsCOAs = coaPatterns.some(pattern =>
      new RegExp(pattern, 'i').test(inputText)
    ) || (needsPlanning && snap.dataSources.length > 1);

    const needsWargaming = wargamingPatterns.some(pattern =>
      new RegExp(pattern, 'i').test(inputText)
    ) || (needsCOAs && snap.dataSources.length > 2);

    // Auto-enable detected features (but respect manual toggles if they're explicitly set)
    const finalIntelligence = includeIntelligence !== true ? needsIntelligence : includeIntelligence;
    const finalPlanning = includePlanning !== true ? needsPlanning : includePlanning;
    const finalCOAs = includeCOAs !== true ? needsCOAs : includeCOAs;
    const finalWargaming = includeWargaming !== true ? needsWargaming : includeWargaming;

    // Build context with intelligence feature flags
    const enhancedContext = {
      include_planning: finalPlanning,
      generate_coas: finalCOAs,
      run_wargaming: finalWargaming,
      objective: input.trim(),
      // Add intelligence metadata if intelligence features are enabled
      ...(finalIntelligence && {
        intelligence_analysis: true,
        intelligence_domain: snap.dataSources.length > 0 ? 'MULTI_DOMAIN' : 'GENERAL',
        data_sources_count: snap.dataSources.length
      })
    };

    // Merge with data sources
    store.sendMessage(input.trim(), enhancedContext);
    setInput('');
    setShowRealtimeSuggestions(false);
  };

  const handleInputChange = (value: string) => {
    setInput(value);
    
    // Update real-time suggestions
    store.updateRealtimeSuggestions(value);
    setShowRealtimeSuggestions(value.length >= 3);
  };

  const handleSuggestionClick = (suggestion: any) => {
    // Handle different suggestion actions
    if (suggestion.action === 'showUploadModal') {
      setShowUploadModal(true);
    } else if (suggestion.action === 'showCapabilityShowcase') {
      setShowCapabilityShowcase(true);
    } else if (suggestion.action === 'enableNeuralMeshMode') {
      setInput('Enable neural mesh analysis for deep pattern recognition');
    } else if (suggestion.action === 'enableQuantumMode') {
      setInput('Deploy quantum-coordinated agent swarms for complex analysis');
    } else if (suggestion.action === 'showAdvancedAnalytics') {
      setShowAdvancedAnalytics(true);
    } else {
      // Default: add suggestion as input
      setInput(suggestion.title);
    }
    
    setShowRealtimeSuggestions(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const suggestedPrompts = [
    "Analyze system security vulnerabilities and recommend improvements",
    "Optimize system performance and identify bottlenecks", 
    "Research best practices for AI system architecture",
    "Design a comprehensive data processing pipeline",
    "Investigate and resolve complex technical issues",
    "Create detailed analysis of system patterns and trends"
  ];

  return (
    <AdaptiveInterface>
      <div
        data-theme={snap.theme}
        className="h-screen relative overflow-hidden"
        style={{
          background: theme.bg,
          color: theme.text
        }}
      >
      {/* Quick Action Toolbar */}
      <QuickActionToolbar
        onOpenUploadModal={() => setShowUploadModal(true)}
        onOpenCapabilityShowcase={() => setShowCapabilityShowcase(true)}
        onOpenAdvancedAnalytics={() => setShowAdvancedAnalytics(true)}
        onOpenProjectsSidebar={() => setShowProjectsSidebar(true)}
        onOpenCOAWarGamePanel={() => setShowCOAWarGamePanel(true)}
        onOpenRealTimeIntelPanel={() => setShowRealTimeIntelPanel(true)}
        onOpenIntelDashboard={() => setShowIntelDashboard(true)}
      />
      {/* Background Effects */}
      <div className="absolute inset-0">
        {/* Animated scan line */}
        {snap.theme === 'day' && (
          <div 
            className="absolute inset-x-0 top-24 h-px animate-pulse scan-line"
            style={{
              background: `linear-gradient(to right, transparent, ${theme.accent}70, transparent)`,
              animation: 'scan 4s linear infinite'
            }}
          />
        )}
        
        {/* Grid overlay */}
        <div className="opacity-20 grid-pattern" style={{ height: '100%', width: '100%' }}>
          <svg className="w-full h-full">
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke={theme.border} strokeWidth="0.5"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)"/>
          </svg>
        </div>
        
        {/* Simple Background Animation */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Floating particles */}
          {[...Array(8)].map((_, i) => (
            <div 
              key={i}
              className="floating-particle"
              style={{
                left: `${10 + i * 12}%`,
                top: `${20 + (i % 4) * 20}%`,
                animationDelay: `${i * 0.5}s`,
                background: theme.accent,
                opacity: 0.6
              }}
            />
          ))}
          
          {/* Gradient orbs */}
          <div 
            className="absolute top-1/4 left-1/4 w-64 h-64 rounded-full opacity-20"
            style={{
              background: `radial-gradient(circle, ${theme.accent}40, transparent)`,
              animation: 'float 8s ease-in-out infinite',
              animationDelay: '1s'
            }}
          />
          <div 
            className="absolute bottom-1/4 right-1/4 w-48 h-48 rounded-full opacity-15"
            style={{
              background: `radial-gradient(circle, ${theme.neon || theme.accent}30, transparent)`,
              animation: 'float 10s ease-in-out infinite reverse',
              animationDelay: '2s'
            }}
          />
        </div>
      </div>

      {/* Full-Width Header */}
      <header 
        className="fixed top-0 left-0 right-0 z-40"
        style={{
          height: '70px',
          background: theme.cardBg,
          backdropFilter: 'blur(20px)',
          borderBottom: `1px solid ${theme.border}`,
          boxShadow: `0 1px 0 ${theme.border}`,
          margin: 0,
          padding: 0,
          width: '100vw'
        }}
      >
        <div className="flex items-center justify-between h-full px-6">
          {/* Left Side - Logo */}
          <div className="flex items-center gap-3">
            <div 
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{
                background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})`
              }}
            >
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold" style={{ color: theme.text }}>
                AgentForge AI
              </h1>
            </div>
          </div>

          {/* Center - Status */}
          <div className="flex items-center gap-4">
            {/* Enhanced AI Status Indicator */}
            <div className="flex items-center gap-2 px-3 py-1 rounded-lg" style={{ background: `${theme.accent}15` }}>
              <Brain className="w-4 h-4" style={{ color: theme.accent }} />
              <span className="text-sm font-medium" style={{ color: theme.text }}>
                Enhanced AI
              </span>
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" title="Enhanced AI capabilities active" />
            </div>
            
            {snap.dataSources.length > 0 && (
              <div 
                className="flex items-center gap-2 px-3 py-1 rounded-lg cursor-pointer transition-all hover:opacity-80"
                style={{ background: `${theme.accent}15` }}
                title={`${snap.dataSources.length} data sources connected. Click to manage attachments.`}
                onClick={() => {
                  // Scroll to attachment area
                  const attachmentArea = document.querySelector('[data-attachment-area]');
                  if (attachmentArea) {
                    attachmentArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
                  }
                }}
              >
                <Paperclip className="w-4 h-4" style={{ color: theme.accent }} />
                <span className="text-sm font-medium" style={{ color: theme.text }}>
                  {snap.dataSources.length}
                </span>
                <div className="flex items-center gap-1">
                  {snap.dataSources.filter(s => s.status === 'ready').length > 0 && (
                    <CheckCircle className="w-3 h-3 text-green-400" />
                  )}
                  {snap.dataSources.filter(s => s.status === 'processing').length > 0 && (
                    <Clock className="w-3 h-3 text-yellow-400 animate-spin" />
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Side - User Profile */}
          <div className="flex items-center gap-3">
            <button
              onClick={store.toggleTheme}
              className="p-2 rounded-lg transition-colors btn-secondary"
              title="Toggle theme"
            >
              {snap.theme === 'day' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>

            {/* User Profile */}
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center gap-2 px-3 py-2 rounded-lg transition-all btn-hover"
                style={{
                  background: theme.cardBg,
                  border: `1px solid ${theme.border}`,
                  color: theme.text
                }}
              >
                <div 
                  className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                  style={{ background: theme.accent, color: 'white' }}
                >
                  JD
                </div>
                <span className="text-sm font-medium">John Doe</span>
                <ChevronDown className="w-4 h-4" />
              </button>

              {/* User Menu Dropdown */}
              <AnimatePresence>
                {showUserMenu && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    className="absolute mt-2 rounded-lg overflow-hidden"
                  style={{
                    width: '200px',
                    right: '0px',
                    transform: 'translateX(-10px)',
                    background: theme.cardBg,
                    border: `1px solid ${theme.border}`,
                    backdropFilter: 'blur(20px)',
                    boxShadow: `0 8px 40px rgba(0,0,0,0.8), 0 0 0 1px ${theme.border}`,
                    zIndex: 50
                  }}
                >
                  <div className="p-3 border-b" style={{ borderColor: theme.border }}>
                    <p className="font-medium text-sm" style={{ color: theme.text }}>John Doe</p>
                    <p className="text-xs" style={{ color: theme.textSecondary, opacity: 0.8 }}>john.doe@company.com</p>
                  </div>
                  
                  <div className="p-2">
                    <button
                      className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all"
                      style={{ 
                        color: theme.text,
                        background: 'transparent'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = theme.accent + '15';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                      }}
                    >
                      <Settings className="w-4 h-4" />
                      Settings
                    </button>
                    <button
                      className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all"
                      style={{ 
                        color: '#ff6b6b',
                        background: 'transparent'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = '#ff6b6b15';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = 'transparent';
                      }}
                    >
                      <LogOut className="w-4 h-4" />
                      Sign Out
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
      </header>

      <div className="relative z-10 h-full flex" style={{ paddingTop: '70px' }}>
        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col" style={{ maxWidth: '100%', margin: '0 auto', paddingRight: '340px' }}>

          {/* Chat Area */}
          <div 
            ref={chatContainerRef}
            className="flex-1 overflow-y-auto px-6 py-4 scroll-smooth" 
            style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center',
              scrollBehavior: 'smooth',
              overscrollBehavior: 'contain',
              maxHeight: 'calc(100vh - 200px)', // Ensure proper height
              minHeight: '0', // Allows flex child to shrink
              height: '100%'
          }}>
            {/* Welcome State */}
            {snap.messages.length <= 1 && (
              <div 
                className="text-center py-16 mx-auto"
                style={{
                  maxWidth: '800px',
                  width: '100%'
                }}
              >
                <div 
                  className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-6"
                  style={{
                    background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})`
                  }}
                >
                  <Sparkles className="w-10 h-10 text-white" />
                </div>
                <h2 
                  className="text-4xl font-bold mb-4"
                  style={{ 
                    color: theme.text,
                    textShadow: `0 0 20px ${theme.accent}50`
                  }}
                >
                  How can I help you today?
                </h2>
                <p 
                  className="text-lg mb-8 leading-relaxed"
                  style={{ color: theme.textSecondary, maxWidth: '800px', margin: '0 auto 2rem' }}
                >
                  I&apos;m powered by an intelligent swarm of AI agents that can process data, 
                  solve complex problems, and provide deep insights. Upload your data or just ask me anything.
                </p>
                
                <div className="grid grid-cols-1 md-grid-cols-2 gap-4" style={{ maxWidth: '800px', margin: '0 auto' }}>
                  {suggestedPrompts.map((prompt, index) => (
                    <motion.button
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="hud-card p-5 text-left btn-hover"
                      style={{
                        color: theme.text,
                        minHeight: '80px',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                      onClick={() => setInput(prompt)}
                    >
                      <p className="text-sm leading-relaxed font-medium">{prompt}</p>
                    </motion.button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            <div 
              className="space-y-6 mx-auto"
              style={{
                maxWidth: '800px',
                width: '100%',
                paddingBottom: '200px', // Much more space at bottom for scrolling
                minHeight: 'fit-content'
              }}
            >
{/* Capability banner disabled to prevent unwanted popups */}
              <AnimatePresence>
                {snap.messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className="flex items-start gap-3" style={{ maxWidth: '80%' }}>
                      {message.role === 'assistant' && (
                        <div 
                          className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1"
                          style={{
                            background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})`
                          }}
                        >
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                      )}
                      
                      <div 
                        className="rounded-2xl px-4 py-3 break-words"
                        style={{
                          background: message.role === 'user' ? theme.accent : theme.cardBg,
                          color: message.role === 'user' ? 'white' : theme.text,
                          border: message.role === 'assistant' ? `1px solid ${theme.border}` : 'none',
                          backdropFilter: 'blur(10px)',
                          boxShadow: message.role === 'user' 
                            ? `0 4px 20px ${theme.accent}40` 
                            : `0 4px 20px ${theme.bg}40`,
                          maxWidth: '100%',
                          wordWrap: 'break-word',
                          overflowWrap: 'break-word'
                        }}
                      >
                        <EnhancedMarkdownRenderer
                          content={message.content}
                          onImplementationApprove={(id) => {
                            console.log('Implementation approved:', id);
                            // Optionally refresh or update UI state
                          }}
                        />
                      </div>

                      {message.role === 'user' && (
                        <div 
                          className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1"
                          style={{
                            background: theme.cardBg,
                            border: `1px solid ${theme.border}`
                          }}
                        >
                          <User className="w-4 h-4" style={{ color: theme.text }} />
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {/* Typing Indicator */}
              {snap.isTyping && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="flex items-start gap-3">
                    <div 
                      className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{
                        background: `linear-gradient(to right, ${theme.accent}, ${theme.accent})`
                      }}
                    >
                      <Bot className="w-4 h-4 text-white" />
                    </div>
                    <div 
                      className="rounded-2xl px-4 py-3"
                      style={{
                        background: theme.cardBg,
                        border: `1px solid ${theme.border}`,
                        color: theme.text
                      }}
                    >
                      <div className="flex items-center gap-1">
                        <div 
                          className="w-2 h-2 rounded-full animate-pulse"
                          style={{ background: theme.accent }}
                        ></div>
                        <div 
                          className="w-2 h-2 rounded-full animate-pulse"
                          style={{ background: theme.accent, animationDelay: '0.2s' }}
                        ></div>
                        <div 
                          className="w-2 h-2 rounded-full animate-pulse"
                          style={{ background: theme.accent, animationDelay: '0.4s' }}
                        ></div>
                      </div>
                      <p className="text-xs mt-2" style={{ opacity: 0.7 }}>
                        {snap.realAgentMetrics?.totalAgentsDeployed || 0} agents processing your request...
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} style={{ height: '20px', width: '100%' }} />
            </div>
          </div>
          
          {/* Scroll to bottom button (appears when not at bottom) */}
          {snap.messages.length > 2 && (
            <button
              onClick={() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })}
              className="fixed bottom-24 right-8 w-12 h-12 rounded-full shadow-lg transition-all duration-200 hover:scale-110 z-50"
              style={{
                background: theme.accent,
                color: 'white',
                border: 'none',
                cursor: 'pointer'
              }}
              title="Scroll to bottom"
            >
              <ChevronDown className="w-6 h-6 mx-auto" />
            </button>
          )}

        </div>
      </div>
      
      {/* Floating Input Area - Aligned with conversation */}
      <div 
        className="fixed bottom-6 p-6 rounded-lg"
        style={{
          left: '50%',
          transform: 'translateX(calc(-50% - 170px))', // Account for sidebar offset
          background: theme.cardBg,
          backdropFilter: 'blur(20px)',
          border: `1px solid ${theme.border}`,
          maxWidth: '800px',
          width: 'calc(100vw - 700px)',
          minWidth: '400px',
          zIndex: 50,
          boxShadow: `0 8px 40px rgba(0,0,0,0.3)`
        }}
      >
            {/* Intelligence Feature Toggles */}
            {snap.dataSources.length > 0 && (
              <div className="mb-3 flex flex-wrap gap-2">
                <label 
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all hover:opacity-80"
                  style={{
                    background: includeIntelligence ? `${theme.accent}30` : `${theme.cardBg}`,
                    border: `1px solid ${includeIntelligence ? theme.accent : theme.border}`,
                    color: includeIntelligence ? theme.accent : theme.textSecondary
                  }}
                >
                  <input 
                    type="checkbox" 
                    checked={includeIntelligence}
                    onChange={(e) => setIncludeIntelligence(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-xs font-medium">🧠 Intelligence Analysis</span>
                </label>
                
                <label 
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all hover:opacity-80"
                  style={{
                    background: includePlanning ? `${theme.accent}30` : `${theme.cardBg}`,
                    border: `1px solid ${includePlanning ? theme.accent : theme.border}`,
                    color: includePlanning ? theme.accent : theme.textSecondary
                  }}
                >
                  <input 
                    type="checkbox" 
                    checked={includePlanning}
                    onChange={(e) => setIncludePlanning(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-xs font-medium">📋 Goal Planning</span>
                </label>
                
                <label 
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all hover:opacity-80"
                  style={{
                    background: includeCOAs ? `${theme.accent}30` : `${theme.cardBg}`,
                    border: `1px solid ${includeCOAs ? theme.accent : theme.border}`,
                    color: includeCOAs ? theme.accent : theme.textSecondary
                  }}
                >
                  <input 
                    type="checkbox" 
                    checked={includeCOAs}
                    onChange={(e) => setIncludeCOAs(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-xs font-medium">⚔️ COA Generation</span>
                </label>
                
                <label 
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-all hover:opacity-80"
                  style={{
                    background: includeWargaming ? `${theme.accent}30` : `${theme.cardBg}`,
                    border: `1px solid ${includeWargaming ? theme.accent : theme.border}`,
                    color: includeWargaming ? theme.accent : theme.textSecondary
                  }}
                >
                  <input 
                    type="checkbox" 
                    checked={includeWargaming}
                    onChange={(e) => setIncludeWargaming(e.target.checked)}
                    className="w-4 h-4"
                  />
                  <span className="text-xs font-medium">🎮 Wargaming</span>
                </label>

                <button
                  onClick={() => {
                    setIncludeIntelligence(true);
                    setIncludePlanning(true);
                    setIncludeCOAs(true);
                    setIncludeWargaming(true);
                  }}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all hover:opacity-80"
                  style={{
                    background: `${theme.accent}20`,
                    border: `1px solid ${theme.accent}`,
                    color: theme.accent
                  }}
                  title="Enable all intelligence capabilities"
                >
                  ⚡ Enable All
                </button>
              </div>
            )}
            
            <div className="flex items-center gap-3">
              <button 
                className="btn-secondary rounded-lg btn-hover flex-shrink-0"
                title="Upload files or connect data streams"
                onClick={() => setShowUploadModal(true)}
                style={{
                  width: '48px',
                  height: '48px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <Paperclip className="w-5 h-5" />
              </button>

              <div className="flex-1 relative">
                {/* RealtimeSuggestions disabled to prevent unwanted popups */}
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => handleInputChange(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask me anything... I&apos;ll deploy the right agents to help you."
                  className="chat-input"
                  style={{
                    paddingRight: '60px',
                    minHeight: '48px',
                    maxHeight: '120px',
                    resize: 'none'
                  }}
                  disabled={snap.isTyping}
                  rows={1}
                />
                
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || snap.isTyping}
                  className="btn-primary absolute right-3 top-1/2 transform -translate-y-1/2 rounded-lg disabled-opacity disabled-cursor"
                  style={{
                    width: '40px',
                    height: '40px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Data Sources & Input Hints */}
            <div className="mt-3" data-attachment-area>
              {/* Compact Attachment Display */}
              {snap.dataSources.length > 0 && (
                <div className="mb-3">
                  <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
                    {snap.dataSources.slice(0, 10).map((source) => (
                      <div
                        key={source.id}
                        className="px-2 py-1 rounded text-xs flex items-center gap-1 group"
                        style={{
                          background: `${theme.accent}20`,
                          color: theme.accent,
                          border: `1px solid ${theme.accent}40`
                        }}
                      >
                        <span className="truncate max-w-20">
                          {source.name.length > 15 ? `${source.name.substring(0, 15)}...` : source.name}
                        </span>
                        <button
                          onClick={() => store.removeDataSource(source.id)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <X className="w-3 h-3 text-red-400" />
                        </button>
                      </div>
                    ))}
                    {snap.dataSources.length > 10 && (
                      <div
                        className="px-2 py-1 rounded text-xs"
                        style={{
                          background: `${theme.accent}10`,
                          color: theme.text,
                          opacity: 0.7
                        }}
                      >
                        +{snap.dataSources.length - 10} more
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* Input Hints */}
              <div className="flex items-center justify-center text-xs" style={{ opacity: 0.6 }}>
                <span>Press Enter to send, Shift+Enter for new line</span>
              </div>
            </div>
          </div>

      {/* Jobs Sidebar removed - only modal version used */}

      {/* Upload Modal */}
      <UploadModal 
        isOpen={showUploadModal} 
        onClose={() => setShowUploadModal(false)} 
      />

      {/* Help & Capabilities Buttons */}
      <div className="fixed bottom-6 left-6 z-20 flex flex-col space-y-3">
        <button
          onClick={() => setShowCapabilityShowcase(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="View Platform Capabilities"
        >
          <Bot className="w-5 h-5" />
        </button>

        <button
          onClick={() => setShowAdvancedAnalytics(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="Advanced Analytics"
        >
          <Database className="w-5 h-5" />
        </button>

        <button
          onClick={() => setShowProjectsSidebar(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="Project Management - Multi-Project Workflow"
        >
          <Folder className="w-5 h-5" />
        </button>

        <button
          onClick={() => setShowCOAWarGamePanel(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="COA & Wargaming - Courses of Action Analysis"
        >
          <Target className="w-5 h-5" />
        </button>

        <button
          onClick={() => setShowRealTimeIntelPanel(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="Real-Time Intelligence - Live Threat Feed"
        >
          <Activity className="w-5 h-5" />
        </button>

        <button
          onClick={() => setShowIntelDashboard(true)}
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="Intelligence Dashboard - Real-Time Threat Monitoring"
        >
          <Shield className="w-5 h-5" />
        </button>

        <button
          className="p-3 rounded-full shadow-lg hover:scale-105 transition-transform"
          style={{
            background: theme.accent,
            color: 'white'
          }}
          title="Help & Documentation"
        >
          <HelpCircle className="w-5 h-5" />
        </button>
      </div>

      {/* Modals */}
      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
      />
      <JobsSidebar
        isOpen={showJobsSidebar}
        onClose={() => setShowJobsSidebar(false)}
      />
      <ProjectsSidebar
        isOpen={showProjectsSidebar}
        onClose={() => setShowProjectsSidebar(false)}
      />
      <COAWarGamePanel
        isOpen={showCOAWarGamePanel}
        onClose={() => setShowCOAWarGamePanel(false)}
      />
      <RealTimeIntelPanel
        isOpen={showRealTimeIntelPanel}
        onClose={() => setShowRealTimeIntelPanel(false)}
      />
      <CapabilityShowcase
        isOpen={showCapabilityShowcase}
        onClose={() => setShowCapabilityShowcase(false)}
      />
      <AdvancedAnalytics
        isOpen={showAdvancedAnalytics}
        onClose={() => setShowAdvancedAnalytics(false)}
      />
      <IntelligenceDashboard
        isOpen={showIntelDashboard}
        onClose={() => setShowIntelDashboard(false)}
      />
      </div>
    </AdaptiveInterface>
  );
}