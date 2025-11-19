'use client';

import { useSnapshot } from 'valtio';
import { store, Job } from '@/lib/state';
import { motion, AnimatePresence } from 'framer-motion';

export function JobsTable() {
  const s = useSnapshot(store);
  const rows = s.jobs;

  return (
    <div className="mt-2">
      <header className="grid grid-cols-12 gap-3 px-5 py-2 text-xs opacity-60">
        <div className="col-span-4">ID</div>
        <div className="col-span-2">Status</div>
        <div className="col-span-3">Owner</div>
        <div className="col-span-3">Updated</div>
      </header>

      <div className="divide-y divide-white/5">
        <AnimatePresence initial={false}>
          {rows.length === 0 ? (
            <SkeletonRows key="skeleton" />
          ) : (
            rows.map((j) => <Row key={j.id} job={j} />)
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function Row({ job }: { job: Job }) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      className="grid grid-cols-12 gap-3 px-5 py-3 hover:bg-white/5 dark:hover:bg-red-900/5 transition"
    >
      <div className="col-span-4 font-mono text-xs md:text-sm">{job.id}</div>
      <div className="col-span-2 text-xs md:text-sm">{job.status}</div>
      <div className="col-span-3 text-xs md:text-sm">{job.owner ?? '—'}</div>
      <div className="col-span-3 text-xs md:text-sm">
        {typeof window !== 'undefined' ? new Date(job.updatedAt).toLocaleTimeString() : 'Loading...'}
      </div>
    </motion.div>
  );
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="grid grid-cols-12 gap-3 px-5 py-3">
          {Array.from({ length: 4 }).map((__, j) => (
            <div key={j} className="col-span-3 md:col-span-{
              j===0?4:j===1?2:j===2?3:3
            } h-4 rounded bg-white/10 dark:bg-red-900/20 animate-pulse"></div>
          ))}
        </div>
      ))}
    </>
  );
}
