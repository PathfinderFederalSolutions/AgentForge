import React, { useState } from "react";

const mockQueue = [
  {
    id: "pkt1",
    target: "A",
    coa: "COA1",
    roe: "ROE1",
    evidence: ["ev1", "ev2"],
    approved: false,
    denied: false,
  },
];

export default function Engagement() {
  const [queue, setQueue] = useState(mockQueue);
  const [selected, setSelected] = useState(queue[0]);
  const [approval, setApproval] = useState({ first: false, second: false });
  const [denied, setDenied] = useState(false);

  function handleApprove() {
    if (!approval.first) {
      setApproval({ ...approval, first: true });
    } else if (!approval.second) {
      setApproval({ ...approval, second: true });
      setQueue((q) => q.map((pkt) => pkt.id === selected.id ? { ...pkt, approved: true } : pkt));
    }
  }
  function handleDeny() {
    setDenied(true);
    setQueue((q) => q.map((pkt) => pkt.id === selected.id ? { ...pkt, denied: true } : pkt));
  }

  return (
    <div style={{ display: "flex", gap: 32 }}>
      <div style={{ width: 240 }}>
        <h2>Engagement Queue</h2>
        <ul>
          {queue.map((pkt) => (
            <li key={pkt.id} style={{ marginBottom: 8 }}>
              <button onClick={() => setSelected(pkt)}>{pkt.target}</button>
              {pkt.approved && <span style={{ color: "green" }}>✔</span>}
              {pkt.denied && <span style={{ color: "red" }}>✖</span>}
            </li>
          ))}
        </ul>
      </div>
      <div style={{ flex: 1 }}>
        <h2>Details</h2>
        <div>Target: {selected.target}</div>
        <div>COA: {selected.coa}</div>
        <div>ROE: {selected.roe}</div>
        <div>Evidence: {selected.evidence.join(", ")}</div>
        <div>ROE Snapshot: <pre>{JSON.stringify(selected.roe, null, 2)}</pre></div>
        <div style={{ marginTop: 16 }}>
          <button disabled={selected.approved || denied} onClick={handleApprove}>
            {approval.first ? (approval.second ? "Approved" : "Second Approval (WebAuthn)") : "First Approval (YubiKey)"}
          </button>
          <button disabled={selected.approved || denied} style={{ marginLeft: 8 }} onClick={handleDeny}>Deny</button>
        </div>
        {denied && <div style={{ color: "red" }}>Denied. Second approval blocked.</div>}
        {selected.approved && <div style={{ color: "green" }}>Approved. Action authorized.</div>}
      </div>
    </div>
  );
}
