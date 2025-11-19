#!/bin/zsh

# Output markdown filename
timestamp=$(date -u +"%Y%m%dT%H%M%SZ")
outfile="agentforge_filetree_${timestamp}.md"

# Write header
echo "# AgentForge Complete File Tree" > $outfile
echo "" >> $outfile
echo "Generated: $timestamp" >> $outfile
echo "" >> $outfile
echo '```' >> $outfile

# Generate file tree (excluding .git and .venv for brevity)
find . \
  -not -path "./.git*" \
  -not -path "./.venv*" \
  -not -name "*.pyc" \
  | sort >> $outfile

echo '```' >> $outfile

echo "File tree written to $outfile"