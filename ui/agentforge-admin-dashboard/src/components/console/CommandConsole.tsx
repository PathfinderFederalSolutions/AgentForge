'use client';

import { useState } from 'react';
import { store } from '@/lib/state';
import { useSnapshot } from 'valtio';
import { Send } from 'lucide-react';

export function CommandConsole() {
  const [text, setText] = useState('');
  const s = useSnapshot(store);

  return (
    <form
      className="mt-3 flex flex-col gap-3"
      onSubmit={(e) => {
        e.preventDefault();
        if (!text.trim()) return;
        store.sendCommand(text.trim());
        setText('');
      }}
    >
      <textarea
        placeholder="Ask anything…"
        className="input min-h-[84px] font-medium"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="flex items-center justify-between">
        <div className="label">AgentForge // Submit to Swarm</div>
        <button disabled={!s.connected} className="btn disabled:opacity-40">
          <Send className="h-4 w-4" />
          Submit
        </button>
      </div>
    </form>
  );
}
