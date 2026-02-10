#!/bin/bash

# 只启动 AIOps Lab 服务器（不启动主程序）

# 指定 Python 路径（默认使用当前环境的 python）
PYTHON="${PYTHON:-python}"

echo "=================================================="
echo "Starting AIOps Lab Server Only"
echo "=================================================="
echo "Using Python: $PYTHON"
echo ""

mkdir -p ./log/py_log

# 启动 AIOps Lab Server
echo "[1/2] Starting AIOps Lab Server..."
nohup $PYTHON -m environment.aiopslab_server > "./log/py_log/aiopslab_server.log" 2>&1 &
SERVER_PID=$!
echo "      ✓ Server started with PID: $SERVER_PID"
echo ""

# 等待服务器初始化并配置 Docker Hub 凭据
echo "[2/2] Waiting for server initialization..."
sleep 30
echo "      ✓ Server should be ready"
echo ""

echo "Setting up Docker Hub credentials..."
./setup_dockerhub_secret_safe.sh "${DOCKER_USER}" "${DOCKER_PASS}" "${DOCKER_EMAIL}"
if [ $? -eq 0 ]; then
    echo "      ✓ Docker Hub credentials configured"
else
    echo "      ⚠️  Warning: Failed to configure Docker Hub credentials"
    echo "      Applications may encounter ImagePullBackOff errors"
fi
echo ""

echo "=================================================="
echo "Server started successfully!"
echo "=================================================="
echo ""
echo "Server PID: $SERVER_PID"
echo "Log file: ./log/py_log/aiopslab_server.log"
echo ""
echo "Commands:"
echo "  Check server logs: tail -f ./log/py_log/aiopslab_server.log"
echo "  Check server status: ps aux | grep $SERVER_PID"
echo "  Stop server: kill $SERVER_PID"
echo ""
echo "Next step: Wait 30-60 seconds, then run: ./run_mitigation_test.sh"
echo ""
