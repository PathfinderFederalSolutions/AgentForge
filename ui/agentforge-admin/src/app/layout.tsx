import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AgentForge AI - Intelligent Agent Swarm",
  description: "Harness the power of AI agent swarms for complex problem solving",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full" suppressHydrationWarning>
      <body 
        className={`${inter.className} h-full antialiased`} 
        style={{ 
          background: '#05080D', 
          color: '#D6E2F0',
          margin: 0,
          padding: 0
        }}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}