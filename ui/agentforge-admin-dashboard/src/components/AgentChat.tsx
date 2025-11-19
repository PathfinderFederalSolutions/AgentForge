"use client";

import { useState } from "react";
import { useSnapshot } from "valtio";
import { store } from "@/lib/state";
import { Send, TerminalSquare } from "lucide-react";

/**
 * AgentChat
 * - Matches CommandConsole/HUD styling
 * - Uses shared valtio store to send jobs over WS
 * - Keeps a simple local transcript for UX feedback
 */
export default function AgentChat() {
  const s = useSnapshot(store);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<{ role: "USER" | "AGENT"; text: string }[]>([]);

  const run = async () => {
    const text = input.trim();
    if (!text) return;
    setBusy(true);
    setHistory((h) => [...h, { role: "USER", text }]);

    try {
      // send to the swarm via WS
      store.sendCommand(text);
      setHistory((h) => [
        ...h,
        { role: "AGENT", text: s.connected ? "Queued task with the swarm…" : "Offline — will retry when reconnected." },
      ]);
    } catch (e: any) {
      setHistory((h) => [...h, { role: "AGENT", text: `Error: ${e?.message ?? "unknown"}` }]);
    } finally {
      setInput("");
      setBusy(false);
    }
  };

  return (
    <section className="hud-card p-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TerminalSquare className="h-4 w-4 opacity-70" />
          <span className="label">AGENT CHAT</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="label">LINK</span>
          <span className="value">{s.connected ? "CONNECTED" : "DISCONNECTED"}</span>
        </div>
      </div>

      <form
        className="mt-4 flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          run();
        }}
      >
        <textarea
          rows={4}
          placeholder="Ask anything…"
          className="input min-h-[96px] font-medium"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <div className="flex items-center justify-between">
          <span className="label">Submit to Swarm</span>
          <button type="submit" disabled={busy || !s.connected} className="btn disabled:opacity-40">
            <Send className="h-4 w-4" />
            {busy ? "Submitting…" : "Submit"}
          </button>
        </div>
      </form>

      {/* Transcript */}
      <div className="mt-4 max-h-60 overflow-auto rounded-xl border border-white/10 bg-white/5 p-3 dark:bg-black/40">
        {history.length === 0 ? (
          <div className="text-sm opacity-60">No messages yet. Your prompts and system responses will appear here.</div>
        ) : (
          <ul className="space-y-2 text-sm">
            {history.map((m, i) => (
              <li key={i} className="flex gap-2">
                <span
                  className={`label shrink-0 ${
                    m.role === "USER" ? "text-day-accent dark:text-night.text" : "opacity-70"
                  }`}
                >
                  {m.role}
                </span>
                <span className="whitespace-pre-wrap">{m.text}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
