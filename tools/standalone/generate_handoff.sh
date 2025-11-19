#!/bin/zsh

timestamp=$(date -u +"%Y%m%dT%H%M%SZ")
handoff_file="handoff_compiled_${timestamp}.md"
latest_inventory=$(ls system_inventory_*.md | sort | tail -n 1)

cat <<EOF > $handoff_file
# Handoff Bundle

# Get file list up to depth 4
find . -type f | awk -F/ 'NF<=5' | sort > filelist.txt

## 1. Curated Context
Purpose: Snapshot of current AgentForge platform for architecture, scaling, and production readiness.

## 2. System Inventory
- Timestamp (UTC): $timestamp
- Hostname: $(hostname)
- User: $(whoami)

## Latest System Inventory
EOF

cat "$latest_inventory" >> $handoff_file

echo "\n## 3. Features & Key Components" >> $handoff_file

# Python features
echo "\n### Python Features:" >> $handoff_file
grep -E 'def |class |@app\.|@router\.' *.py | awk -F: '{print $1}' | sort | uniq | while read f; do
  echo "- $f" >> $handoff_file
done

# Node/TypeScript features
echo "\n### Node/TypeScript Features:" >> $handoff_file
find . -name "*.ts" -o -name "*.js" | grep -v node_modules | while read f; do
  echo "- $f" >> $handoff_file
done

# Frontend features
echo "\n### Frontend (React/Vite) Features:" >> $handoff_file
find . -name "App.tsx" -o -name "*.tsx" -o -name "*.html" | grep -v node_modules | while read f; do
  echo "- $f" >> $handoff_file
done

# Endpoints
echo "\n## 4. API Endpoints" >> $handoff_file
grep -r --include="*.py" -E "@app\.(get|post|put|delete|patch)|@router\." . | awk -F: '{print $2}' | sort | uniq >> $handoff_file
grep -r --include="*.ts" -E "app\.(get|post|put|delete|patch)" . | awk -F: '{print $2}' | sort | uniq >> $handoff_file

# Languages
echo "\n## 5. Languages Used" >> $handoff_file
find . -type f | awk -F. '{if (NF>1) print $NF}' | sort | uniq -c | sort -nr | awk '{print $2 ": " $1 " files"}' >> $handoff_file

# Frameworks
echo "\n## 6. Frameworks Detected" >> $handoff_file
grep -r -i -E "fastapi|flask|django|express|react|vite|node|kubernetes|prometheus|nats|pytest|docker|terraform" . | awk -F: '{print $1}' | sort | uniq >> $handoff_file

echo "\n--- End of Automated Feature & Endpoint Extraction ---\n" >> $handoff_file

echo "Generated $handoff_file with summarized features, endpoints, languages, and frameworks."