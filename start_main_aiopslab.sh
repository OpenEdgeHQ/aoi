#!/bin/bash

# Start Main AIOps Lab
# This script starts the main AIOps Lab process in the background

echo "Starting Main AIOps Lab..."

# Ensure log directory exists
mkdir -p ./log/py_log

# Start the main process
nohup python -m main_aiopslab > "./log/py_log/main_aiopslab.log" 2>&1 &

# Get the PID
MAIN_PID=$!

echo "Main AIOps Lab started with PID: $MAIN_PID"
echo "Log file: ./log/py_log/main_aiopslab.log"
echo ""
echo "To check logs: tail -f ./log/py_log/main_aiopslab.log"
echo "To stop process: kill $MAIN_PID"
echo ""
echo "PID saved for reference: $MAIN_PID"
