#!/bin/bash

# Install Document Processing Dependencies
# This adds PDF and document parsing capabilities

echo "📄 Installing Document Processing Libraries..."

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

# Install PDF parsing library
echo "Installing pypdf..."
pip install "pypdf>=4.0.0"

# Install document parsing library
echo "Installing python-docx..."
pip install "python-docx>=1.1.0"

echo ""
echo "✅ Document processing libraries installed!"
echo ""
echo "📋 Installed:"
echo "  • pypdf (PDF text extraction)"
echo "  • python-docx (Word document processing)"
echo ""
echo "🔄 Now restart the backend to enable document extraction:"
echo "   ./restart_clean.sh"

