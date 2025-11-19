"use client";

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import CodeImplementationBlock from './CodeImplementationBlock';

interface ImplementationData {
  title: string;
  filePath: string;
  status: string;
  codeLines: number;
  description: string;
  code: string;
  implementationId: string;
}

interface EnhancedMarkdownRendererProps {
  content: string;
  onImplementationApprove?: (id: string) => void;
}

export default function EnhancedMarkdownRenderer({ 
  content, 
  onImplementationApprove 
}: EnhancedMarkdownRendererProps) {
  const [approvedImplementations, setApprovedImplementations] = useState<Set<string>>(new Set());

  // Parse implementation blocks from markdown content
  const parseImplementations = (text: string): { 
    processedContent: string, 
    implementations: ImplementationData[] 
  } => {
    const implementations: ImplementationData[] = [];
    let processedContent = text;

    // Regex to match implementation blocks
    const implementationRegex = /## Implementation (\d+): ([^\n]+)\n\n\*\*File Path:\*\* `([^`]+)`[^\n]*\n\*\*Status:\*\* ([^\n]+)[^\n]*\n\*\*Code Lines:\*\* (\d+) lines[^\n]*\n\*\*Description:\*\* ([^\n]+)\n\n### Full Production Code:\n\n```python\n([\s\S]*?)\n```\n\n\*\*Implementation ID:\*\* `([^`]+)`/g;

    let match;
    while ((match = implementationRegex.exec(text)) !== null) {
      const [fullMatch, number, title, filePath, status, codeLines, description, code, implementationId] = match;
      
      implementations.push({
        title,
        filePath,
        status,
        codeLines: parseInt(codeLines),
        description,
        code,
        implementationId
      });

      // Replace the matched block with a placeholder
      processedContent = processedContent.replace(fullMatch, `[IMPLEMENTATION_BLOCK_${implementations.length - 1}]`);
    }

    return { processedContent, implementations };
  };

  const { processedContent, implementations } = parseImplementations(content);

  const handleKeep = (implementationId: string) => {
    setApprovedImplementations(prev => new Set([...prev, implementationId]));
    if (onImplementationApprove) {
      onImplementationApprove(implementationId);
    }
  };

  // Custom renderer for ReactMarkdown
  const components = {
    h1: ({ children }: any) => (
      <h1 style={{ 
        fontSize: '1.5em', 
        fontWeight: 'bold', 
        margin: '0.5em 0',
        color: '#00A39B'
      }}>
        {children}
      </h1>
    ),
    h2: ({ children }: any) => (
      <h2 style={{ 
        fontSize: '1.3em', 
        fontWeight: 'bold', 
        margin: '0.4em 0',
        color: '#00A39B'
      }}>
        {children}
      </h2>
    ),
    h3: ({ children }: any) => (
      <h3 style={{ 
        fontSize: '1.1em', 
        fontWeight: 'bold', 
        margin: '0.3em 0',
        color: '#D6E2F0'
      }}>
        {children}
      </h3>
    ),
    strong: ({ children }: any) => (
      <strong style={{ fontWeight: 'bold', color: '#00A39B' }}>
        {children}
      </strong>
    ),
    code: ({ children, className }: any) => {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      
      if (language) {
        // Multi-line code block
        return (
          <div style={{
            margin: '16px 0',
            borderRadius: '8px',
            overflow: 'hidden',
            border: '1px solid #0F2237'
          }}>
            <div style={{
              background: '#2d2d2d',
              padding: '8px 16px',
              borderBottom: '1px solid #404040',
              color: '#00A39B',
              fontSize: '0.9em',
              fontWeight: 'bold'
            }}>
              {language.toUpperCase()} Code
            </div>
            <div style={{
              maxHeight: '400px',
              overflow: 'auto'
            }}>
              <SyntaxHighlighter
                language={language}
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
                {children}
              </SyntaxHighlighter>
            </div>
          </div>
        );
      } else {
        // Inline code
        return (
          <code style={{ 
            backgroundColor: 'rgba(0, 163, 155, 0.1)', 
            padding: '2px 6px', 
            borderRadius: '3px', 
            fontSize: '0.9em',
            color: '#00A39B',
            border: '1px solid rgba(0, 163, 155, 0.2)'
          }}>
            {children}
          </code>
        );
      }
    },
    p: ({ children }: any) => {
      // Check if this paragraph contains an implementation block placeholder
      const text = children?.toString() || '';
      const implementationMatch = text.match(/\[IMPLEMENTATION_BLOCK_(\d+)\]/);
      
      if (implementationMatch) {
        const index = parseInt(implementationMatch[1]);
        const impl = implementations[index];
        
        if (impl) {
          return (
            <CodeImplementationBlock
              title={impl.title}
              filePath={impl.filePath}
              status={impl.status}
              codeLines={impl.codeLines}
              description={impl.description}
              code={impl.code}
              implementationId={impl.implementationId}
              onKeep={handleKeep}
            />
          );
        }
      }
      
      return (
        <p style={{ 
          margin: '0.5em 0',
          color: '#D6E2F0',
          lineHeight: '1.6'
        }}>
          {children}
        </p>
      );
    },
    ul: ({ children }: any) => (
      <ul style={{ 
        margin: '0.5em 0', 
        paddingLeft: '1.5em',
        color: '#D6E2F0'
      }}>
        {children}
      </ul>
    ),
    ol: ({ children }: any) => (
      <ol style={{ 
        margin: '0.5em 0', 
        paddingLeft: '1.5em',
        color: '#D6E2F0'
      }}>
        {children}
      </ol>
    ),
    li: ({ children }: any) => (
      <li style={{ 
        margin: '0.2em 0',
        color: '#D6E2F0'
      }}>
        {children}
      </li>
    ),
    blockquote: ({ children }: any) => (
      <blockquote style={{
        borderLeft: '4px solid #00A39B',
        paddingLeft: '16px',
        margin: '16px 0',
        background: 'rgba(0, 163, 155, 0.05)',
        padding: '12px 16px',
        borderRadius: '0 4px 4px 0'
      }}>
        {children}
      </blockquote>
    )
  };

  return (
    <div style={{ 
      lineHeight: '1.6',
      margin: 0,
      maxHeight: 'none',
      overflow: 'visible',
      color: '#D6E2F0'
    }}>
      <ReactMarkdown components={components}>
        {processedContent}
      </ReactMarkdown>
      
      {/* Render implementation blocks that weren't caught by paragraph parsing */}
      {implementations.map((impl, index) => {
        // Check if this implementation was already rendered in a paragraph
        if (!processedContent.includes(`[IMPLEMENTATION_BLOCK_${index}]`)) {
          return (
            <CodeImplementationBlock
              key={impl.implementationId}
              title={impl.title}
              filePath={impl.filePath}
              status={impl.status}
              codeLines={impl.codeLines}
              description={impl.description}
              code={impl.code}
              implementationId={impl.implementationId}
              onKeep={handleKeep}
            />
          );
        }
        return null;
      })}
      
      {/* Show approval status */}
      {approvedImplementations.size > 0 && (
        <div style={{
          margin: '16px 0',
          padding: '12px',
          background: 'rgba(0, 163, 155, 0.1)',
          border: '1px solid #00A39B',
          borderRadius: '6px',
          color: '#00A39B'
        }}>
          <strong>Approved Implementations:</strong> {Array.from(approvedImplementations).join(', ')}
        </div>
      )}
    </div>
  );
}
