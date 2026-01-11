#!/bin/bash
# Start the FastAPI backend server

cd "$(dirname "$0")/.."

echo "Starting Backend API..."
echo "================================"

# Create backend venv if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating backend virtual environment..."
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install -r backend/requirements.txt
else
    source backend/venv/bin/activate
fi

cd backend

echo "Backend API running at: http://127.0.0.1:8000"
echo "API docs at: http://127.0.0.1:8000/docs"

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
