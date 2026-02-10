#!/bin/bash

# 指定 Python 路径（默认使用当前环境的 python）
PYTHON="${PYTHON:-python}"

echo "=================================================="
echo "Starting All AIOps Lab Services"
echo "=================================================="
echo "Using Python: $PYTHON"
echo ""

mkdir -p ./log/py_log

# Step 1: 启动 AIOps Lab Server
echo "[1/3] Starting AIOps Lab Server..."
nohup $PYTHON -m environment.aiopslab_server > "./log/py_log/aiopslab_server.log" 2>&1 &
SERVER_PID=$!
echo "      ✓ Server started with PID: $SERVER_PID"
echo ""

# Step 2: 等待服务器初始化并配置 Docker Hub 凭据
echo "[2/3] Waiting for server initialization..."
sleep 30
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

# Step 3: 启动 Main AIOps Lab（支持多轮运行）
echo "[3/3] Starting Main AIOps Lab..."
sleep 60

# 检查是否设置了 NUM_ROUNDS 环境变量（默认1轮）
NUM_ROUNDS=${NUM_ROUNDS:-1}

if [ "$NUM_ROUNDS" -gt 1 ]; then
    echo "      Running $NUM_ROUNDS rounds of evaluation..."
    for i in $(seq 1 $NUM_ROUNDS); do
        echo ""
        echo "========== Round $i/$NUM_ROUNDS Start: $(date) =========="
        ROUND=$i $PYTHON -m main_aiopslab
        echo "========== Round $i/$NUM_ROUNDS End: $(date) =========="
    done
    echo ""
    echo "      ✓ All $NUM_ROUNDS rounds completed"
else
    nohup $PYTHON -m main_aiopslab > "./log/py_log/main_aiopslab.log" 2>&1 &
    MAIN_PID=$!
    echo "      ✓ Main process started with PID: $MAIN_PID"
fi
echo ""

echo "=================================================="
echo "All services started successfully!"
echo "=================================================="
echo ""
echo "Log files:"
echo "  - Server: ./log/py_log/aiopslab_server.log"
echo "  - Main:   ./log/py_log/main_aiopslab.log"
echo ""
echo "Commands:"
echo "  Check server logs: tail -f ./log/py_log/aiopslab_server.log"
echo "  Check main logs:   tail -f ./log/py_log/main_aiopslab.log"
echo "  Stop all:          ./stop_all.sh"
echo ""