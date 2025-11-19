import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "AgentForge - Advanced AI Agent Orchestration",
  description: "Cutting-edge tactical agent swarm orchestration and management platform"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className="h-full" style={{ background: '#05080D', color: '#D6E2F0' }} suppressHydrationWarning>
      <body className="min-h-screen antialiased" style={{ background: '#05080D', color: '#D6E2F0' }} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
