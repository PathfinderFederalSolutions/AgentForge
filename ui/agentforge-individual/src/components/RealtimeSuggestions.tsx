'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { useSnapshot } from 'valtio';
import { store } from '../lib/store';

interface RealtimeSuggestionsProps {
  isVisible: boolean;
  onSuggestionClick: (suggestion: any) => void;
}

export default function RealtimeSuggestions({ isVisible, onSuggestionClick }: RealtimeSuggestionsProps) {
  const snap = useSnapshot(store);

  // Completely disable realtime suggestions popup
  return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: 10 }}
        className="absolute bottom-full left-0 right-0 mb-2 bg-[#0D1421] border border-[#1E2A3E] rounded-lg shadow-xl z-50 overflow-hidden"
      >
        <div className="p-3 border-b border-[#1E2A3E] bg-gradient-to-r from-[#0D1421] to-[#1E2A3E]">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-[#D6E2F0] flex items-center">
              <span className="w-2 h-2 bg-[#00A39B] rounded-full mr-2 animate-pulse"></span>
              AGI Capabilities Available
            </h4>
            <span className="text-xs text-[#8FA8C4]">
              {snap.realtimeSuggestions.length} suggestions
            </span>
          </div>
        </div>

        <div className="max-h-64 overflow-y-auto">
          {snap.realtimeSuggestions.map((suggestion, index) => (
            <motion.button
              key={suggestion.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onSuggestionClick(suggestion)}
              className="w-full p-4 text-left hover:bg-[#1E2A3E] transition-colors border-b border-[#1E2A3E] last:border-b-0 group"
            >
              <div className="flex items-start space-x-3">
                <div className="text-2xl group-hover:scale-110 transition-transform">
                  {suggestion.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <h5 className="text-sm font-medium text-[#D6E2F0] group-hover:text-[#00A39B] transition-colors">
                      {suggestion.title}
                    </h5>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        suggestion.priority === 'high' ? 'bg-[#00A39B] text-white' :
                        suggestion.priority === 'medium' ? 'bg-[#FFB800] text-black' :
                        'bg-[#1E2A3E] text-[#8FA8C4]'
                      }`}>
                        {suggestion.priority.toUpperCase()}
                      </span>
                      {suggestion.confidence && (
                        <span className="text-xs text-[#8FA8C4]">
                          {Math.round(suggestion.confidence * 100)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-[#8FA8C4] line-clamp-2 group-hover:text-[#D6E2F0] transition-colors">
                    {suggestion.description}
                  </p>
                  
                  {suggestion.examples && suggestion.examples.length > 0 && (
                    <div className="mt-2">
                      <div className="flex flex-wrap gap-1">
                        {suggestion.examples.slice(0, 3).map((example: string, i: number) => (
                          <span
                            key={i}
                            className="px-2 py-0.5 bg-[#1E2A3E] text-[#8FA8C4] text-xs rounded group-hover:bg-[#2A3B52] transition-colors"
                          >
                            {example.length > 20 ? `${example.substring(0, 20)}...` : example}
                          </span>
                        ))}
                        {suggestion.examples.length > 3 && (
                          <span className="px-2 py-0.5 text-[#00A39B] text-xs">
                            +{suggestion.examples.length - 3} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.button>
          ))}
        </div>

        {/* Action Footer */}
        <div className="p-3 bg-gradient-to-r from-[#1E2A3E] to-[#0D1421] border-t border-[#1E2A3E]">
          <div className="flex items-center justify-between text-xs">
            <span className="text-[#8FA8C4] flex items-center">
              <span className="w-1.5 h-1.5 bg-[#00A39B] rounded-full mr-2"></span>
              Click any suggestion to activate capability
            </span>
            <button
              onClick={() => store.realtimeSuggestions = []}
              className="text-[#8FA8C4] hover:text-[#D6E2F0] transition-colors"
            >
              Dismiss
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// Component for displaying capability suggestions in the main chat
export function CapabilitySuggestionBanner() {
  const snap = useSnapshot(store);

  // Completely disable capability suggestion banner
  return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-4 p-4 bg-gradient-to-r from-[#00A39B]/10 to-[#0D1421] border border-[#00A39B]/30 rounded-lg"
    >
      <div className="flex items-start justify-between mb-3">
        <h4 className="text-sm font-semibold text-[#D6E2F0] flex items-center">
          <span className="text-lg mr-2">🤖</span>
          AGI Capabilities Activated
        </h4>
        <span className="text-xs text-[#8FA8C4]">
          {snap.currentCapabilities.length} capabilities
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {snap.currentCapabilities.slice(0, 4).map((capability, index) => (
          <motion.div
            key={capability.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className="p-3 bg-[#0D1421] border border-[#1E2A3E] rounded-lg hover:border-[#00A39B] transition-colors cursor-pointer group"
          >
            <div className="flex items-center space-x-3">
              <div className="text-xl group-hover:scale-110 transition-transform">
                {capability.icon}
              </div>
              <div className="flex-1 min-w-0">
                <h5 className="text-sm font-medium text-[#D6E2F0] group-hover:text-[#00A39B] transition-colors">
                  {capability.title}
                </h5>
                <p className="text-xs text-[#8FA8C4] line-clamp-1">
                  {capability.description}
                </p>
              </div>
              <div className="flex items-center space-x-1">
                <span className={`w-2 h-2 rounded-full ${
                  capability.priority === 'high' ? 'bg-[#00A39B]' :
                  capability.priority === 'medium' ? 'bg-[#FFB800]' :
                  'bg-[#8FA8C4]'
                }`}></span>
                {capability.confidence && (
                  <span className="text-xs text-[#8FA8C4]">
                    {Math.round(capability.confidence * 100)}%
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {snap.currentCapabilities.length > 4 && (
        <div className="mt-3 text-center">
          <button className="text-xs text-[#00A39B] hover:text-[#D6E2F0] transition-colors">
            View all {snap.currentCapabilities.length} capabilities
          </button>
        </div>
      )}
    </motion.div>
  );
}
