#!/bin/bash

# Start AIOps Lab Server
# This script starts the AIOps Lab server in the background

echo "Starting AIOps Lab Server..."

# Ensure log directory exists
mkdir -p ./log/py_log

# Start the server
nohup python -m environment.aiopslab_server > "./log/py_log/aiopslab_server.log" 2>&1 &

# Get the PID
SERVER_PID=$!

echo "AIOps Lab Server started with PID: $SERVER_PID"
echo "Log file: ./log/py_log/aiopslab_server.log"
echo ""
echo "To check logs: tail -f ./log/py_log/aiopslab_server.log"
echo "To stop server: kill $SERVER_PID"
echo ""
echo "PID saved for reference: $SERVER_PID"


# Step 2: 等待服务器初始化并配置 Docker Hub 凭据
echo "[2/3] Waiting for server initialization..."
sleep 20
echo "      ✓ Server should be ready"
echo ""

echo "[2/3] Setting up Docker Hub credentials..."
./setup_dockerhub_secret_safe.sh "${DOCKER_USER}" "${DOCKER_PASS}" "${DOCKER_EMAIL}"
if [ $? -eq 0 ]; then
    echo "      ✓ Docker Hub credentials configured"
else
    echo "      ⚠️  Warning: Failed to configure Docker Hub credentials"
    echo "      Applications may encounter ImagePullBackOff errors"
fi
echo ""