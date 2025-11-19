'use client';

import { useSnapshot } from 'valtio';
import { store } from '@/lib/state';
import { motion } from 'framer-motion';
import { Activity, Cpu, Radio, Server, Webhook, Circle } from 'lucide-react';
import { CommandConsole } from './console/CommandConsole';
import { JobsTable } from './console/JobsTable';

export function HUD() {
  const s = useSnapshot(store);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* STATUS */}
      <section className="hud-card col-span-1 p-5">
        <div className="flex items-center justify-between">
          <div className="label">LINK</div>
          <div className="flex items-center gap-2">
            <Circle className={`h-2.5 w-2.5 ${s.connected ? 'text-green-400' : 'text-red-500'}`} fill="currentColor" />
            <div className="value">{s.connected ? 'CONNECTED' : 'DISCONNECTED'}</div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4">
          <Metric icon={<Server className="h-4 w-4 opacity-80" />} label="Nodes" value={s.meta.nodes ?? '—'} />
          <Metric icon={<Activity className="h-4 w-4 opacity-80" />} label="Queue Depth" value={s.meta.queueDepth ?? '—'} />
          <Metric icon={<Cpu className="h-4 w-4 opacity-80" />} label="RPS" value={s.meta.rps ?? '—'} />
          <Metric icon={<Radio className="h-4 w-4 opacity-80" />} label="WS" value={s.wsUrl.replace(/^ws:\/\//, '')} />
        </div>
      </section>

      {/* CONSOLE */}
      <section className="hud-card col-span-1 lg:col-span-2 p-5">
        <div className="flex items-center justify-between">
          <div className="label">COMMAND CONSOLE</div>
          <div className="flex items-center gap-2">
            <Webhook className="h-4 w-4 opacity-70" />
            <span className="value opacity-70">swarm.jobs</span>
          </div>
        </div>
        <CommandConsole />
      </section>

      {/* JOBS */}
      <section className="hud-card col-span-1 lg:col-span-3 p-0 overflow-hidden">
        <div className="flex items-center justify-between px-5 pt-4">
          <div className="label">TASK STREAM</div>
          <div className="label">Owner / Status / Updated</div>
        </div>
        <JobsTable />
      </section>
    </div>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: .3 }}
      className="rounded-xl border border-white/10 p-4 bg-white/5 dark:bg-black/40"
    >
      <div className="flex items-center gap-2 text-xs opacity-70">{icon}<span className="label">{label}</span></div>
      <div className="mt-1 value">{value}</div>
    </motion.div>
  );
}
