'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSnapshot } from 'valtio';
import { store } from '../lib/store';

interface AdaptiveInterfaceProps {
  children: React.ReactNode;
}

interface UserPreferences {
  expertiseLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  interactionStyle: 'brief' | 'detailed' | 'conversational' | 'technical';
  complexityPreference: number;
  preferredCapabilities: string[];
  uiDensity: 'compact' | 'comfortable' | 'spacious';
  showAdvancedFeatures: boolean;
}

interface AdaptiveFeature {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  confidence: number;
  reasoning: string;
}

export default function AdaptiveInterface({ children }: AdaptiveInterfaceProps) {
  const snap = useSnapshot(store);
  const [userPreferences, setUserPreferences] = useState<UserPreferences>({
    expertiseLevel: 'intermediate',
    interactionStyle: 'detailed',
    complexityPreference: 0.6,
    preferredCapabilities: [],
    uiDensity: 'comfortable',
    showAdvancedFeatures: false
  });
  const [adaptiveFeatures, setAdaptiveFeatures] = useState<AdaptiveFeature[]>([]);
  const [showAdaptivePanel, setShowAdaptivePanel] = useState(false);

  useEffect(() => {
    // Load user preferences and adaptive features
    loadUserPreferences();
    
    // Subscribe to real-time preference updates
    store.agiClient.addEventListener('user_profile_update', handleProfileUpdate);
    
    return () => {
      store.agiClient.removeEventListener('user_profile_update', handleProfileUpdate);
    };
  }, []);

  const loadUserPreferences = async () => {
    try {
      // Get user patterns and preferences from backend
      const patterns = await store.agiClient.getUserPatterns('user_001');
      const predictions = await store.agiClient.predictNextAction('user_001', {
        available_capabilities: store.getAllCapabilities().map(cap => cap.id),
        current_session_duration: Date.now() - (snap.messages[0]?.timestamp.getTime() || Date.now()),
        data_sources_count: snap.dataSources.length
      });

      // Update preferences based on patterns
      if (patterns.patterns) {
        const behavioralPattern = patterns.patterns.find((p: any) => p.pattern_type === 'behavioral');
        const preferencePattern = patterns.patterns.find((p: any) => p.pattern_type === 'preference');
        
        if (behavioralPattern) {
          const complexity = behavioralPattern.pattern_data?.complexity_preference || 0.6;
          const capabilityDiversity = behavioralPattern.pattern_data?.capability_diversity || 2;
          
          setUserPreferences(prev => ({
            ...prev,
            complexityPreference: complexity,
            expertiseLevel: complexity > 0.8 ? 'expert' : complexity > 0.6 ? 'advanced' : 'intermediate',
            showAdvancedFeatures: capabilityDiversity > 3
          }));
        }
        
        if (preferencePattern) {
          const preferredCaps = preferencePattern.pattern_data?.preferred_capabilities || [];
          const interactionStyle = preferencePattern.pattern_data?.interaction_style || 'detailed';
          
          setUserPreferences(prev => ({
            ...prev,
            preferredCapabilities: preferredCaps,
            interactionStyle: interactionStyle
          }));
        }
      }

      // Generate adaptive features based on predictions
      if (predictions.predictions) {
        const features: AdaptiveFeature[] = predictions.predictions.map((pred: any) => ({
          id: pred.insight_id,
          name: pred.prediction,
          description: pred.reasoning.join('; '),
          enabled: pred.confidence > 0.7,
          confidence: pred.confidence,
          reasoning: pred.reasoning.join(', ')
        }));
        
        setAdaptiveFeatures(features);
      }

    } catch (error) {
      console.error('Failed to load user preferences:', error);
    }
  };

  const handleProfileUpdate = (data: any) => {
    // Handle real-time profile updates
    if (data.profile) {
      setUserPreferences(prev => ({
        ...prev,
        expertiseLevel: data.profile.expertise_level,
        interactionStyle: data.profile.interaction_style,
        complexityPreference: data.profile.complexity_preference,
        preferredCapabilities: data.profile.preferred_capabilities
      }));
    }
  };

  const getAdaptiveStyles = () => {
    const styles: any = {};
    
    // Adjust UI density
    if (userPreferences.uiDensity === 'compact') {
      styles['--spacing-scale'] = '0.8';
      styles['--font-scale'] = '0.9';
    } else if (userPreferences.uiDensity === 'spacious') {
      styles['--spacing-scale'] = '1.2';
      styles['--font-scale'] = '1.1';
    }
    
    // Adjust complexity indicators
    if (userPreferences.expertiseLevel === 'expert') {
      styles['--show-technical-details'] = 'block';
      styles['--show-advanced-controls'] = 'flex';
    } else if (userPreferences.expertiseLevel === 'beginner') {
      styles['--show-technical-details'] = 'none';
      styles['--show-advanced-controls'] = 'none';
    }
    
    return styles;
  };

  const renderAdaptiveFeatures = () => (
    <motion.div
      initial={{ opacity: 0, x: 300 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 300 }}
      className="fixed right-4 top-20 w-80 bg-[#0D1421] border border-[#1E2A3E] rounded-lg shadow-xl z-40 max-h-96 overflow-y-auto"
    >
      <div className="p-4 border-b border-[#1E2A3E] bg-gradient-to-r from-[#0D1421] to-[#1E2A3E]">
        <div className="flex items-center justify-between">
          <h4 className="text-lg font-semibold text-[#D6E2F0] flex items-center">
            <span className="text-xl mr-2">🎯</span>
            Adaptive Intelligence
          </h4>
          <button
            onClick={() => setShowAdaptivePanel(false)}
            className="text-[#8FA8C4] hover:text-[#D6E2F0] transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* User Profile Summary */}
        <div className="p-3 bg-[#1E2A3E] rounded-lg">
          <h5 className="text-sm font-medium text-[#D6E2F0] mb-2">Your Profile</h5>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Expertise:</span>
              <span className="text-[#00A39B] capitalize">{userPreferences.expertiseLevel}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Style:</span>
              <span className="text-[#00A39B] capitalize">{userPreferences.interactionStyle}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8FA8C4]">Complexity:</span>
              <span className="text-[#00A39B]">{(userPreferences.complexityPreference * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>

        {/* Adaptive Features */}
        <div>
          <h5 className="text-sm font-medium text-[#D6E2F0] mb-3">Intelligent Adaptations</h5>
          <div className="space-y-2">
            {adaptiveFeatures.slice(0, 3).map((feature) => (
              <div key={feature.id} className="p-3 bg-[#0D1421] border border-[#1E2A3E] rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <h6 className="text-sm font-medium text-[#D6E2F0]">{feature.name}</h6>
                  <span className={`w-2 h-2 rounded-full ${
                    feature.enabled ? 'bg-[#00A39B]' : 'bg-[#8FA8C4]'
                  }`}></span>
                </div>
                <p className="text-xs text-[#8FA8C4] mb-2">{feature.description}</p>
                <div className="flex justify-between text-xs">
                  <span className="text-[#8FA8C4]">Confidence:</span>
                  <span className="text-[#00A39B]">{(feature.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="pt-3 border-t border-[#1E2A3E]">
          <button
            onClick={loadUserPreferences}
            className="w-full px-3 py-2 bg-[#00A39B] text-white rounded-lg hover:bg-[#008A84] transition-colors text-sm"
          >
            🔄 Refresh Adaptations
          </button>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div style={getAdaptiveStyles()}>
      {children}
      
      {/* Adaptive Features Toggle */}
      <button
        onClick={() => setShowAdaptivePanel(true)}
        className="fixed right-6 top-6 p-3 bg-[#00A39B] text-white rounded-full shadow-lg hover:scale-105 transition-transform z-30"
        title="View Adaptive Intelligence"
      >
        <span className="text-lg">🎯</span>
      </button>

      {/* Adaptive Features Panel */}
      <AnimatePresence>
        {showAdaptivePanel && renderAdaptiveFeatures()}
      </AnimatePresence>

      {/* Expertise Level Indicator */}
      {userPreferences.expertiseLevel === 'expert' && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed top-4 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-gradient-to-r from-[#00A39B] to-[#008A84] text-white rounded-full text-sm font-medium z-30"
        >
          🏆 Expert Mode Active
        </motion.div>
      )}

      {/* Advanced Features Hint */}
      {userPreferences.showAdvancedFeatures && snap.activeAgents > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="fixed bottom-20 right-6 p-4 bg-[#0D1421] border border-[#00A39B] rounded-lg shadow-xl max-w-xs z-30"
        >
          <div className="flex items-start space-x-3">
            <span className="text-xl">⚡</span>
            <div>
              <h6 className="text-sm font-medium text-[#D6E2F0] mb-1">
                Advanced Features Available
              </h6>
              <p className="text-xs text-[#8FA8C4] mb-2">
                Based on your usage patterns, quantum coordination is recommended for this request.
              </p>
              <button className="text-xs text-[#00A39B] hover:text-[#D6E2F0] transition-colors">
                Enable Now →
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
