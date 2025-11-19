"use client";

import { useMemo } from "react";
import { useSnapshot } from "valtio";
import { store, Job } from "@/lib/state";
import { Activity, Cpu, Radio, Server, Table } from "lucide-react";

/**
 * JobMonitor
 * - Visual parity with new HUD cards
 * - Pulls live meta + jobs from shared store (WS-driven)
 * - Graceful empty/skeleton state
 */
export default function JobMonitor() {
  const s = useSnapshot(store);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const j of s.jobs) c[j.status] = (c[j.status] || 0) + 1;
    return c;
  }, [s.jobs]);

  return (
    <section className="hud-card p-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Table className="h-4 w-4 opacity-70" />
          <span className="label">SWARM STATUS</span>
        </div>
        <div className="flex items-center gap-6">
          <Badge icon={<Radio className="h-4 w-4 opacity-80" />} label="WS" value={s.connected ? "connected" : "disconnected"} />
          <Badge icon={<Server className="h-4 w-4 opacity-80" />} label="Nodes" value={fmt(s.meta.nodes)} />
          <Badge icon={<Activity className="h-4 w-4 opacity-80" />} label="Queue" value={fmt(s.meta.queueDepth)} />
          <Badge icon={<Cpu className="h-4 w-4 opacity-80" />} label="RPS" value={fmt(s.meta.rps)} />
        </div>
      </div>

      {/* Status chips by job state */}
      <div className="mt-4 flex flex-wrap gap-2">
        {Object.keys(counts).length === 0 ? (
          <span className="label opacity-60">No jobs yet.</span>
        ) : (
          Object.entries(counts).map(([k, v]) => (
            <span key={k} className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs dark:bg-black/40">
              {k}: <span className="ml-1 font-mono">{v}</span>
            </span>
          ))
        )}
      </div>

      {/* Jobs table */}
      <div className="mt-4 overflow-hidden rounded-xl border border-white/10">
        <table className="w-full border-collapse text-sm">
          <thead className="bg-white/5 dark:bg-black/40">
            <tr className="text-left">
              <Th>ID</Th>
              <Th>Status</Th>
              <Th>Owner</Th>
              <Th>Updated</Th>
            </tr>
          </thead>
          <tbody>
            {s.jobs.length === 0 ? (
              <SkeletonRows />
            ) : (
              s.jobs.slice(0, 100).map((j) => <Row key={j.id} job={j} />)
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-2 text-xs opacity-70">{children}</th>;
}

function Row({ job }: { job: Job }) {
  return (
    <tr className="border-t border-white/10 hover:bg-white/5 dark:hover:bg-red-900/5">
      <td className="px-4 py-2 font-mono">{job.id}</td>
      <td className="px-4 py-2">{job.status}</td>
      <td className="px-4 py-2">{job.owner ?? "—"}</td>
      <td className="px-4 py-2">
        {isFinite(job.updatedAt) ? new Date(job.updatedAt).toLocaleString() : "—"}
      </td>
    </tr>
  );
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 6 }).map((_, i) => (
        <tr key={i} className="border-t border-white/10">
          {Array.from({ length: 4 }).map((__, j) => (
            <td key={j} className="px-4 py-3">
              <div className="h-4 w-48 rounded bg-white/10 dark:bg-red-900/20 animate-pulse" />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
}

function Badge({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 dark:bg-black/40">
      {icon}
      <span className="label">{label}</span>
      <span className="value">{value}</span>
    </div>
  );
}

function fmt(v?: number) {
  return typeof v === "number" && isFinite(v) ? v : "—";
}
