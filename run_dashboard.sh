#!/bin/bash
echo "Starting Flask Backend..."
cd dashboard/backend
python3 app.py &
BACKEND_PID=$!

echo "Starting React Frontend..."
cd ../frontend
npm run dev -- --port 5002 &
FRONTEND_PID=$!

function cleanup() {
    echo "Shutting down dashboard..."
    kill -9 $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit
}
trap cleanup SIGINT

echo "Dashboard is running at http://localhost:5002!"
echo "Press Ctrl+C to stop both servers safely."
wait
