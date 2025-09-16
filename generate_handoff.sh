#!/bin/zsh

# Get timestamp
timestamp=$(date -u +"%Y%m%dT%H%M%SZ")

# Find latest system inventory file
latest_inventory=$(ls system_inventory_*.md | sort | tail -n 1)

# Get file list up to depth 4
find . -type f | awk -F/ 'NF<=5' | sort > filelist.txt

# Output file
handoff_file="handoff_compiled_${timestamp}.md"

# Write header and context (edit as needed for your style)
cat <<EOF > $handoff_file
# Handoff Bundle

## 1. Curated Context
# System Context (GPT-5 Input)

Purpose: Snapshot of current AgentForge platform (NATS JetStream + Prometheus + SLO + Exporter + K8s namespace agentforge-staging) for architecture, scaling, and production readiness.

## 2. System Inventory

# System Inventory Snapshot

- Timestamp (UTC): $timestamp
- Hostname: $(hostname)
- User: $(whoami)

## Latest System Inventory
EOF

# Append latest system inventory
cat "$latest_inventory" >> $handoff_file

# Write file list section
cat <<EOF >> $handoff_file

## 3. Repository Structure (Depth 4)
\`\`\`
EOF

cat filelist.txt >> $handoff_file

echo "\`\`\`" >> $handoff_file

echo "Generated $handoff_file with latest inventory and file list."