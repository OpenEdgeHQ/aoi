#!/bin/bash

# Check AIOps Lab Services Status
# This script checks the status of running AIOps Lab processes

echo "=================================================="
echo "AIOps Lab Services Status"
echo "=================================================="
echo ""

# Check for aiopslab_server
echo "AIOps Lab Server:"
SERVER_PIDS=$(pgrep -f "environment.aiopslab_server")
if [ -z "$SERVER_PIDS" ]; then
    echo "  Status: ❌ Not running"
else
    echo "  Status: ✓ Running"
    for PID in $SERVER_PIDS; do
        echo "    - PID: $PID"
        ps -p $PID -o etime= | sed 's/^/      Uptime: /'
    done
fi
echo ""

# Check for main_aiopslab
echo "Main AIOps Lab:"
MAIN_PIDS=$(pgrep -f "main_aiopslab")
if [ -z "$MAIN_PIDS" ]; then
    echo "  Status: ❌ Not running"
else
    echo "  Status: ✓ Running"
    for PID in $MAIN_PIDS; do
        echo "    - PID: $PID"
        ps -p $PID -o etime= | sed 's/^/      Uptime: /'
    done
fi
echo ""

# Check log files
echo "Log Files:"
if [ -f "./log/py_log/aiopslab_server.log" ]; then
    SERVER_LOG_SIZE=$(du -h "./log/py_log/aiopslab_server.log" | cut -f1)
    SERVER_LOG_LINES=$(wc -l < "./log/py_log/aiopslab_server.log")
    echo "  Server log: $SERVER_LOG_SIZE ($SERVER_LOG_LINES lines)"
else
    echo "  Server log: Not found"
fi

if [ -f "./log/py_log/main_aiopslab.log" ]; then
    MAIN_LOG_SIZE=$(du -h "./log/py_log/main_aiopslab.log" | cut -f1)
    MAIN_LOG_LINES=$(wc -l < "./log/py_log/main_aiopslab.log")
    echo "  Main log:   $MAIN_LOG_SIZE ($MAIN_LOG_LINES lines)"
else
    echo "  Main log:   Not found"
fi
echo ""

echo "=================================================="
echo "Commands:"
echo "  Start all:    ./start_all.sh"
echo "  Stop all:     ./stop_all.sh"
echo "  View logs:    tail -f ./log/py_log/*.log"
echo "=================================================="
echo ""

