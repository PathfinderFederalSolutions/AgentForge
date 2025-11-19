'use client';

import { useEffect } from 'react';
import { Layout } from "@/components/layout/Layout";
import { store, useSnapshot } from "@/lib/state";
import Dashboard from './dashboard/page';

export default function Home() {
  const snap = useSnapshot(store);

  // Initialize WebSocket connection and load real data
  useEffect(() => {
    // Load real data from backend
    store.loadRealData();
    
    if (!snap.connected && !snap.connectTried) {
      // Connect to real-time updates
      const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/v1/realtime/ws';
      console.log('Attempting WebSocket connection to:', wsUrl);
      store.connect(wsUrl);
    }

    // Cleanup on unmount
    return () => {
      store.disconnect();
    };
  }, [snap.connected, snap.connectTried]);

  // Handle page visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        // Page is hidden, don't disconnect but stop trying to reconnect
        console.log('Page hidden, WebSocket will remain connected');
      } else {
        // Page is visible, ensure connection
        if (!snap.connected && snap.connectTried) {
          console.log('Page visible, attempting to reconnect');
          store.connect(process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8080/ws');
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [snap.connected, snap.connectTried]);

  return (
    <Layout>
      <Dashboard />
    </Layout>
  );
}
