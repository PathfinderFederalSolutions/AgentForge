#!/bin/bash

# This script verifies the application setup and configuration.

# Check if Node.js is installed
if ! command -v node &> /dev/null
then
    echo "Node.js is not installed. Please install Node.js to proceed."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null
then
    echo "npm is not installed. Please install npm to proceed."
    exit 1
fi

# Check if TypeScript is installed
if ! command -v tsc &> /dev/null
then
    echo "TypeScript is not installed. Please install TypeScript to proceed."
    exit 1
fi

# Check if all dependencies are installed
npm install --silent
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please check your package.json and try again."
    exit 1
fi

# Run TypeScript compiler to check for type errors
tsc --noEmit
if [ $? -ne 0 ]; then
    echo "TypeScript compilation failed. Please fix the errors and try again."
    exit 1
fi

echo "Verification completed successfully. The application is set up correctly."