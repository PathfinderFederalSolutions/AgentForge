"use client";
import { useEffect, useRef, useState } from "react";

export function useWS(url: string) {
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (ev) => setMessages(m => [...m, String(ev.data)]);
    return () => ws.close();
  }, [url]);

  const send = (data: any) => wsRef.current?.send(JSON.stringify(data));
  return { connected, messages, send };
}
