#!/bin/bash
# Start the Next.js frontend

cd "$(dirname "$0")/../frontend"

echo "Starting Frontend..."
echo "================================"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Frontend running at: http://localhost:3000"

npm run dev
