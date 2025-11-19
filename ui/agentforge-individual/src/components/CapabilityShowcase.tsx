'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSnapshot } from 'valtio';
import { store } from '../lib/store';

interface CapabilityShowcaseProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CapabilityShowcase({ isOpen, onClose }: CapabilityShowcaseProps) {
  const snap = useSnapshot(store);
  const [selectedCategory, setSelectedCategory] = useState<'input' | 'processing' | 'output' | 'all'>('all');

  const capabilityCategories = {
    input: {
      title: "📁 Universal Input Processing",
      description: "I can process any input type with specialized AI agents",
      examples: [
        "📄 Documents (PDF, Word, Excel, PowerPoint)",
        "🖼️ Media (Images, Videos, Audio files)",
        "💾 Data (CSV, JSON, XML, Databases)",
        "🌐 Live Streams (APIs, Sensors, Social feeds)",
        "🔗 External Sources (URLs, Cloud storage, Git repos)",
        "📱 Mobile Data (Photos, Contacts, Messages)",
        "🎤 Voice Commands and Audio Transcription",
        "📊 Real-time Analytics and Dashboards"
      ]
    },
    processing: {
      title: "🧠 Intelligent Processing",
      description: "Deploy million-scale agent swarms for complex problem solving",
      examples: [
        "🔍 Neural Mesh Analysis with 4-tier memory system",
        "⚡ Quantum Agent Coordination for complex tasks",
        "🎯 Predictive Intelligence and Machine Learning",
        "🔄 Real-time Stream Processing and Monitoring",
        "🌟 Emergent Swarm Intelligence behaviors",
        "📈 Advanced Analytics and Pattern Recognition",
        "🛡️ Anomaly Detection and Security Analysis",
        "🎨 Creative Problem Solving and Innovation"
      ]
    },
    output: {
      title: "🛠️ Universal Output Generation",
      description: "Generate any output format from natural language descriptions",
      examples: [
        "💻 Complete Applications (Web, Mobile, Desktop)",
        "📊 Professional Reports and Presentations",
        "🎨 Creative Content (Images, Videos, Music)",
        "📈 Interactive Dashboards and Visualizations",
        "⚙️ Automation Scripts and Workflows",
        "🎮 Games and Interactive Experiences",
        "📚 Books, Articles, and Documentation",
        "🏗️ 3D Models and Architectural Designs"
      ]
    }
  };

  const allCapabilities = store.getAllCapabilities();
  const filteredCapabilities = selectedCategory === 'all' 
    ? allCapabilities 
    : store.getCapabilitiesByType(selectedCategory);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50"
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            className="fixed inset-4 md:inset-8 bg-[#0A0F1C] border border-[#1E2A3E] rounded-2xl z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="p-6 border-b border-[#1E2A3E] bg-gradient-to-r from-[#0A0F1C] to-[#0D1421]">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-bold text-[#D6E2F0] mb-2">
                    AgentForge Platform Capabilities
                  </h2>
                  <p className="text-[#8FA8C4]">
                    Discover the full power of our intelligent automation platform
                  </p>
                </div>
                <button
                  onClick={onClose}
                  className="p-2 text-[#8FA8C4] hover:text-[#D6E2F0] hover:bg-[#1E2A3E] rounded-lg transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="flex h-full">
              {/* Category Sidebar */}
              <div className="w-64 p-6 border-r border-[#1E2A3E] bg-[#0D1421]">
                <h3 className="text-lg font-semibold text-[#D6E2F0] mb-4">Categories</h3>
                <div className="space-y-2">
                  {(['all', 'input', 'processing', 'output'] as const).map((category) => (
                    <button
                      key={category}
                      onClick={() => setSelectedCategory(category)}
                      className={`w-full text-left p-3 rounded-lg transition-colors ${
                        selectedCategory === category
                          ? 'bg-[#00A39B] text-white'
                          : 'text-[#8FA8C4] hover:text-[#D6E2F0] hover:bg-[#1E2A3E]'
                      }`}
                    >
                      {category === 'all' ? '🌟 All Capabilities' : 
                       category === 'input' ? '📁 Input Processing' :
                       category === 'processing' ? '🧠 AI Processing' :
                       '🛠️ Output Generation'}
                    </button>
                  ))}
                </div>

                {/* Stats */}
                <div className="mt-8 p-4 bg-[#1E2A3E] rounded-lg">
                  <h4 className="text-sm font-semibold text-[#D6E2F0] mb-2">System Status</h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Active Agents</span>
                      <span className="text-[#00A39B]">{snap.activeAgents}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Data Sources</span>
                      <span className="text-[#00A39B]">{snap.dataSources.length}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8FA8C4]">Capabilities</span>
                      <span className="text-[#00A39B]">{allCapabilities.length}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Main Content */}
              <div className="flex-1 overflow-y-auto">
                {selectedCategory === 'all' ? (
                  <div className="p-6">
                    <div className="mb-8">
                      <h3 className="text-xl font-semibold text-[#D6E2F0] mb-4">
                        🌟 Complete AGI Platform Overview
                      </h3>
                      <div className="grid gap-6">
                        {Object.entries(capabilityCategories).map(([key, category]) => (
                          <motion.div
                            key={key}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg hover:border-[#00A39B] transition-colors cursor-pointer"
                            onClick={() => setSelectedCategory(key as any)}
                          >
                            <h4 className="text-lg font-semibold text-[#D6E2F0] mb-2">
                              {category.title}
                            </h4>
                            <p className="text-[#8FA8C4] mb-4">{category.description}</p>
                            <div className="grid grid-cols-2 gap-2">
                              {category.examples.slice(0, 4).map((example, index) => (
                                <div key={index} className="text-sm text-[#8FA8C4] flex items-center">
                                  <span className="w-2 h-2 bg-[#00A39B] rounded-full mr-2 flex-shrink-0"></span>
                                  {example}
                                </div>
                              ))}
                            </div>
                            {category.examples.length > 4 && (
                              <div className="text-sm text-[#00A39B] mt-2">
                                +{category.examples.length - 4} more capabilities...
                              </div>
                            )}
                          </motion.div>
                        ))}
                      </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="border-t border-[#1E2A3E] pt-6">
                      <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">🚀 Quick Actions</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <button className="p-4 bg-[#00A39B] text-white rounded-lg hover:bg-[#008A84] transition-colors">
                          📁 Upload Data & Files
                        </button>
                        <button className="p-4 bg-[#1E2A3E] text-[#D6E2F0] rounded-lg hover:bg-[#2A3B52] transition-colors">
                          🧠 Enable Neural Mesh
                        </button>
                        <button className="p-4 bg-[#1E2A3E] text-[#D6E2F0] rounded-lg hover:bg-[#2A3B52] transition-colors">
                          ⚡ Scale to Million Agents
                        </button>
                        <button className="p-4 bg-[#1E2A3E] text-[#D6E2F0] rounded-lg hover:bg-[#2A3B52] transition-colors">
                          🛠️ Generate Application
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="p-6">
                    <div className="mb-6">
                      <h3 className="text-xl font-semibold text-[#D6E2F0] mb-2">
                        {capabilityCategories[selectedCategory].title}
                      </h3>
                      <p className="text-[#8FA8C4]">
                        {capabilityCategories[selectedCategory].description}
                      </p>
                    </div>

                    {/* Detailed Capabilities */}
                    <div className="space-y-4">
                      {filteredCapabilities.map((capability, index) => (
                        <motion.div
                          key={capability.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="p-6 bg-[#0D1421] border border-[#1E2A3E] rounded-lg hover:border-[#00A39B] transition-colors"
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div>
                              <h4 className="text-lg font-semibold text-[#D6E2F0] mb-1">
                                {capability.icon} {capability.title}
                              </h4>
                              <p className="text-[#8FA8C4]">{capability.description}</p>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className={`px-2 py-1 rounded text-xs font-medium ${
                                capability.priority === 'high' ? 'bg-[#00A39B] text-white' :
                                capability.priority === 'medium' ? 'bg-[#FFB800] text-black' :
                                'bg-[#1E2A3E] text-[#8FA8C4]'
                              }`}>
                                {capability.priority.toUpperCase()}
                              </span>
                              <span className="text-xs text-[#8FA8C4]">
                                {Math.round(capability.confidence * 100)}% confidence
                              </span>
                            </div>
                          </div>

                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                              <h5 className="text-sm font-medium text-[#D6E2F0] mb-2">Examples:</h5>
                              <div className="space-y-1">
                                {capability.examples.slice(0, 3).map((example, i) => (
                                  <div key={i} className="text-sm text-[#8FA8C4] flex items-center">
                                    <span className="w-1.5 h-1.5 bg-[#00A39B] rounded-full mr-2 flex-shrink-0"></span>
                                    {example}
                                  </div>
                                ))}
                              </div>
                            </div>

                            {capability.action && (
                              <div className="flex items-end">
                                <button className="px-4 py-2 bg-[#00A39B] text-white rounded-lg hover:bg-[#008A84] transition-colors text-sm">
                                  Try This Capability
                                </button>
                              </div>
                            )}
                          </div>
                        </motion.div>
                      ))}
                    </div>

                    {/* Category Examples */}
                    <div className="mt-8 p-6 bg-gradient-to-r from-[#0D1421] to-[#1E2A3E] rounded-lg">
                      <h4 className="text-lg font-semibold text-[#D6E2F0] mb-4">
                        What I can do in this category:
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {capabilityCategories[selectedCategory].examples.map((example, index) => (
                          <div key={index} className="text-sm text-[#8FA8C4] flex items-center">
                            <span className="w-2 h-2 bg-[#00A39B] rounded-full mr-3 flex-shrink-0"></span>
                            {example}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
