#!/bin/bash

# Start the agent swarm service

# Navigate to the project directory
cd "$(dirname "$0")/.."

# Install dependencies
npm install

# Build the TypeScript files
npm run build

# Start the application
npm start