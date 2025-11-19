"use client";

import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeImplementationBlockProps {
  title: string;
  filePath: string;
  status: string;
  codeLines: number;
  description: string;
  code: string;
  implementationId: string;
  onKeep?: (id: string) => void;
}

export default function CodeImplementationBlock({
  title,
  filePath,
  status,
  codeLines,
  description,
  code,
  implementationId,
  onKeep
}: CodeImplementationBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isApproving, setIsApproving] = useState(false);

  const handleKeep = async () => {
    if (isApproving) return;
    
    setIsApproving(true);
    
    try {
      // Use the chat API to approve implementation
      const API_BASE = process.env.NEXT_PUBLIC_AGENT_API_BASE || 'http://localhost:8000';
      const response = await fetch(`${API_BASE}/v1/chat/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `approve implementation ${implementationId}`,
          context: {
            conversationHistory: [],
            dataSources: [],
            userId: "user_001",
            sessionId: "session_001"
          },
          capabilities: []
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Check for successful integration with multiple success indicators
      const successIndicators = [
        "Successfully integrated",
        "Integration Complete", 
        "Implementation Approved and Integrated",
        "automatically deployed and integrated",
        "new capability is now active"
      ];
      
      const isSuccess = successIndicators.some(indicator => 
        result.response && result.response.includes(indicator)
      );
      
      if (isSuccess) {
        alert(`✅ Implementation ${implementationId} successfully integrated into your codebase!`);
        if (onKeep) onKeep(implementationId);
      } else if (result.response && result.response.includes("Implementation approval failed")) {
        alert(`❌ Integration failed: ${result.response}`);
      } else {
        // Show the actual response for debugging
        console.log('Full response:', result);
        alert(`⚠️ Integration response: ${result.response || 'Unknown response'}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    } finally {
      setIsApproving(false);
    }
  };

  return (
    <div className="implementation-block" style={{
      border: '1px solid #0F2237',
      borderRadius: '8px',
      background: 'linear-gradient(135deg, #05080D 0%, #0E1622 100%)',
      margin: '16px 0',
      overflow: 'hidden',
      boxShadow: '0 4px 20px rgba(0, 163, 155, 0.1)'
    }}>
      {/* Header */}
      <div style={{
        padding: '16px',
        borderBottom: '1px solid #0F2237',
        background: '#05080D'
      }}>
        <h3 style={{ 
          color: '#00A39B', 
          margin: '0 0 8px 0',
          fontSize: '1.1em',
          fontWeight: 'bold'
        }}>
          {title}
        </h3>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
          fontSize: '0.9em',
          color: '#D6E2F0'
        }}>
          <div>
            <span style={{ color: '#00A39B' }}>File:</span> {filePath}
          </div>
          <div>
            <span style={{ color: '#00A39B' }}>Status:</span> {status}
          </div>
          <div>
            <span style={{ color: '#00A39B' }}>Lines:</span> {codeLines}
          </div>
          <div>
            <span style={{ color: '#00A39B' }}>ID:</span> {implementationId}
          </div>
        </div>
        
        <p style={{ 
          margin: '12px 0 0 0',
          color: '#D6E2F0',
          opacity: 0.8,
          fontSize: '0.9em'
        }}>
          {description}
        </p>
      </div>

      {/* Code Block */}
      <div style={{
        position: 'relative',
        background: '#1a1a1a'
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '8px 16px',
          background: '#2d2d2d',
          borderBottom: '1px solid #404040'
        }}>
          <span style={{ 
            color: '#00A39B',
            fontSize: '0.9em',
            fontWeight: 'bold'
          }}>
            Full Production Code
          </span>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            style={{
              background: 'transparent',
              border: '1px solid #00A39B',
              color: '#00A39B',
              padding: '4px 8px',
              borderRadius: '4px',
              fontSize: '0.8em',
              cursor: 'pointer'
            }}
          >
            {isExpanded ? 'Collapse' : 'Expand'}
          </button>
        </div>
        
        <div style={{
          maxHeight: isExpanded ? 'none' : '400px',
          overflow: isExpanded ? 'visible' : 'auto',
          transition: 'max-height 0.3s ease'
        }}>
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            customStyle={{
              margin: 0,
              background: '#1a1a1a',
              fontSize: '0.9em',
              lineHeight: '1.4'
            }}
            showLineNumbers={true}
            lineNumberStyle={{
              color: '#666',
              borderRight: '1px solid #333',
              paddingRight: '8px',
              marginRight: '8px'
            }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      </div>

      {/* Action Bar */}
      <div style={{
        padding: '16px',
        background: '#05080D',
        borderTop: '1px solid #0F2237',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ 
          fontSize: '0.8em',
          color: '#D6E2F0',
          opacity: 0.7
        }}>
          Ready for integration into your codebase
        </div>
        
        <button
          onClick={handleKeep}
          disabled={isApproving}
          style={{
            background: isApproving ? '#666' : '#00A39B',
            color: 'white',
            border: 'none',
            padding: '10px 20px',
            borderRadius: '6px',
            cursor: isApproving ? 'not-allowed' : 'pointer',
            fontWeight: 'bold',
            fontSize: '0.9em',
            boxShadow: '0 2px 8px rgba(0, 163, 155, 0.3)',
            transition: 'all 0.2s ease'
          }}
          onMouseEnter={(e) => {
            if (!isApproving) {
              e.target.style.background = '#008a7b';
              e.target.style.boxShadow = '0 4px 12px rgba(0, 163, 155, 0.4)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isApproving) {
              e.target.style.background = '#00A39B';
              e.target.style.boxShadow = '0 2px 8px rgba(0, 163, 155, 0.3)';
            }
          }}
        >
          {isApproving ? 'Integrating...' : 'Keep & Integrate'}
        </button>
      </div>
    </div>
  );
}
