#!/bin/bash

# Stop All AIOps Lab Services
# This script stops both the server and main process

echo "=================================================="
echo "Stopping All AIOps Lab Services"
echo "=================================================="
echo ""

# Find and kill aiopslab_server processes
echo "[1/2] Stopping AIOps Lab Server..."
SERVER_PIDS=$(pgrep -f "environment.aiopslab_server")
if [ -z "$SERVER_PIDS" ]; then
    echo "      ⚠ No server process found"
else
    for PID in $SERVER_PIDS; do
        kill -9 $PID 2>/dev/null
        echo "      ✓ Killed server process: PID $PID"
    done
fi
echo ""

# Find and kill main_aiopslab processes
echo "[2/2] Stopping Main AIOps Lab..."
MAIN_PIDS=$(pgrep -f "main_aiopslab")
if [ -z "$MAIN_PIDS" ]; then
    echo "      ⚠ No main process found"
else
    for PID in $MAIN_PIDS; do
        kill -9 $PID 2>/dev/null
        echo "      ✓ Killed main process: PID $PID"
    done
fi
echo ""

# Wait for processes to terminate
sleep 2

kind delete cluster --name kind

# Check if any processes are still running
REMAINING=$(pgrep -f "aiopslab" | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "⚠ Warning: $REMAINING process(es) still running"
    echo "   Use 'pkill -9 -f aiopslab' to force kill if needed"
else
    echo "=================================================="
    echo "All services stopped successfully!"
    echo "=================================================="
fi
echo ""

