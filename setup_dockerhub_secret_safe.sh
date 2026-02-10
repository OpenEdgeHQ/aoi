#!/usr/bin/env bash
# 为 kind 集群配置 Docker Hub 认证（安全版本）
# 解决 Docker Hub 速率限制问题
# 
# 使用方法:
#   ./setup_dockerhub_secret_safe.sh <username> <password> <email>
# 或者使用环境变量:
#   export DOCKER_USERNAME="your_username"
#   export DOCKER_PASSWORD="your_password"
#   export DOCKER_EMAIL="your_email"
#   ./setup_dockerhub_secret_safe.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 从参数或环境变量读取配置
DOCKER_USERNAME="${1:-${DOCKER_USERNAME}}"
DOCKER_PASSWORD="${2:-${DOCKER_PASSWORD}}"
DOCKER_EMAIL="${3:-${DOCKER_EMAIL}}"

# 需要配置的所有 namespace
NAMESPACES=("openebs" "observe" "test-social-network")

# 显示使用说明
usage() {
    echo -e "${BLUE}使用方法:${NC}"
    echo "  $0 <username> <password> <email>"
    echo ""
    echo -e "${BLUE}或者使用环境变量:${NC}"
    echo "  export DOCKER_USERNAME=\"your_username\""
    echo "  export DOCKER_PASSWORD=\"your_password\""
    echo "  export DOCKER_EMAIL=\"your_email\""
    echo "  $0"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo "  $0 myuser mypassword myemail@example.com"
    echo ""
    echo -e "${BLUE}将配置以下 namespaces:${NC}"
    echo "  - openebs"
    echo "  - observe"
    echo "  - test-social-network"
    exit 1
}

# 检查必需的参数
if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ] || [ -z "$DOCKER_EMAIL" ]; then
    echo -e "${RED}错误: 缺少必需的参数${NC}"
    echo ""
    usage
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}配置 Docker Hub 认证 - 解决速率限制问题${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "${BLUE}配置信息:${NC}"
echo "  用户名: $DOCKER_USERNAME"
echo "  邮箱: $DOCKER_EMAIL"
echo "  命名空间: ${NAMESPACES[@]}"
echo ""

# 步骤 1: 登录 Docker Hub
echo -e "${YELLOW}[1/4] 登录 Docker Hub...${NC}"
if echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin; then
    echo -e "${GREEN}✓ Docker Hub 登录成功${NC}"
else
    echo -e "${RED}✗ Docker Hub 登录失败${NC}"
    exit 1
fi
echo ""

# 步骤 2: 配置所有 kind 节点
echo -e "${YELLOW}[2/4] 配置 kind 节点的 Docker 凭据...${NC}"
KIND_NODES=$(docker ps --filter "name=kind" --format "{{.Names}}")

if [ -n "$KIND_NODES" ]; then
    for node in $KIND_NODES; do
        echo -e "  配置节点: ${BLUE}$node${NC}"
        docker exec $node mkdir -p /root/.docker 2>/dev/null || true
        docker cp ~/.docker/config.json $node:/root/.docker/config.json 2>/dev/null
        docker exec $node chmod 600 /root/.docker/config.json 2>/dev/null
        
        if docker exec $node test -f /root/.docker/config.json 2>/dev/null; then
            echo -e "    ${GREEN}✓ $node 配置成功${NC}"
        else
            echo -e "    ${YELLOW}⚠ $node 配置失败${NC}"
        fi
    done
else
    echo -e "${YELLOW}  未找到 kind 节点，跳过节点配置${NC}"
fi
echo ""

# 定义配置单个 namespace 的函数
configure_namespace() {
    local NAMESPACE=$1
    local NS_NUM=$2
    local TOTAL_NS=$3
    
    echo -e "${BLUE}============================================${NC}"
    echo -e "${YELLOW}[3/4] 配置 Namespace [$NS_NUM/$TOTAL_NS]: ${BLUE}$NAMESPACE${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    
    # 检查 namespace 是否存在
    echo -e "${YELLOW}  检查 namespace...${NC}"
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        echo -e "  ${GREEN}✓ Namespace '$NAMESPACE' 存在${NC}"
    else
        echo -e "  ${YELLOW}⚠ Namespace '$NAMESPACE' 不存在，将创建...${NC}"
        kubectl create namespace "$NAMESPACE"
        echo -e "  ${GREEN}✓ Namespace '$NAMESPACE' 创建成功${NC}"
    fi
    echo ""
    
    # 创建 Kubernetes secret
    echo -e "${YELLOW}  创建 Docker registry secret...${NC}"
    if kubectl get secret dockerhub-secret -n "$NAMESPACE" &>/dev/null; then
        echo -e "  ${YELLOW}  删除已存在的 secret...${NC}"
        kubectl delete secret dockerhub-secret -n "$NAMESPACE"
    fi
    
    if kubectl create secret docker-registry dockerhub-secret \
      --docker-server=https://index.docker.io/v1/ \
      --docker-username="$DOCKER_USERNAME" \
      --docker-password="$DOCKER_PASSWORD" \
      --docker-email="$DOCKER_EMAIL" \
      -n "$NAMESPACE" 2>/dev/null; then
        echo -e "  ${GREEN}✓ Secret 'dockerhub-secret' 创建成功${NC}"
    else
        echo -e "  ${RED}✗ Secret 创建失败${NC}"
        return 1
    fi
    echo ""
    
    # 将 secret 关联到 ServiceAccount
    echo -e "${YELLOW}  配置 ServiceAccount...${NC}"
    SERVICE_ACCOUNTS=$(kubectl get serviceaccounts -n "$NAMESPACE" -o name 2>/dev/null | sed 's|serviceaccount/||' || echo "default")
    
    for sa in $SERVICE_ACCOUNTS; do
        if kubectl patch serviceaccount "$sa" -n "$NAMESPACE" \
          -p '{"imagePullSecrets": [{"name": "dockerhub-secret"}]}' 2>/dev/null; then
            echo -e "    ${GREEN}✓ ServiceAccount '$sa' 配置成功${NC}"
        else
            echo -e "    ${YELLOW}⚠ ServiceAccount '$sa' 配置跳过${NC}"
        fi
    done
    echo ""
    
    # 重启失败的 pods
    echo -e "${YELLOW}  删除 ImagePullBackOff 和失败的 pods...${NC}"
    BACKOFF_PODS=$(kubectl get pods -n "$NAMESPACE" 2>/dev/null | grep -E 'ImagePullBackOff|ErrImagePull' | awk '{print $1}' || true)
    if [ -n "$BACKOFF_PODS" ]; then
        for pod in $BACKOFF_PODS; do
            kubectl delete pod -n "$NAMESPACE" "$pod" 2>/dev/null && echo -e "    ${GREEN}✓ 删除 $pod${NC}" || true
        done
    fi
    
    kubectl delete pod -n "$NAMESPACE" --field-selector=status.phase=Failed 2>/dev/null || true
    echo -e "  ${GREEN}✓ Namespace '$NAMESPACE' 配置完成${NC}"
    echo ""
}

# 步骤 3-4: 循环配置所有 namespace
TOTAL_NS=${#NAMESPACES[@]}
NS_NUM=0

for NAMESPACE in "${NAMESPACES[@]}"; do
    NS_NUM=$((NS_NUM + 1))
    configure_namespace "$NAMESPACE" "$NS_NUM" "$TOTAL_NS"
done

# 等待 pods 重启
echo -e "${BLUE}等待 5 秒让所有 pods 开始重启...${NC}"
sleep 5
echo ""

# 显示最终状态
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}所有配置完成！${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# 显示所有 namespace 的 pod 状态
for NAMESPACE in "${NAMESPACES[@]}"; do
    echo -e "${BLUE}Namespace: ${YELLOW}$NAMESPACE${NC}"
    kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  暂无 pods"
    echo ""
done

echo -e "${YELLOW}有用的命令:${NC}"
for NAMESPACE in "${NAMESPACES[@]}"; do
    echo -e "  ${BLUE}$NAMESPACE${NC}:"
    echo -e "    监控: ${BLUE}kubectl get pods -n $NAMESPACE -w${NC}"
    echo -e "    详情: ${BLUE}kubectl describe pod <pod-name> -n $NAMESPACE${NC}"
done
echo ""

# 检查所有 namespace 是否还有错误
echo -e "${YELLOW}最终检查:${NC}"
HAS_ERRORS=false
for NAMESPACE in "${NAMESPACES[@]}"; do
    ERRORS=$(kubectl get pods -n "$NAMESPACE" 2>/dev/null | grep -E 'ImagePullBackOff|ErrImagePull|Error' || true)
    if [ -n "$ERRORS" ]; then
        echo -e "  ${RED}⚠ Namespace '$NAMESPACE' 仍有错误:${NC}"
        echo "$ERRORS" | sed 's/^/    /'
        HAS_ERRORS=true
    else
        echo -e "  ${GREEN}✓ Namespace '$NAMESPACE' 所有 pods 正常${NC}"
    fi
done

if [ "$HAS_ERRORS" = true ]; then
    echo ""
    echo -e "${YELLOW}建议:${NC}"
    echo "  1. 等待几分钟让 pods 完全重启"
    echo "  2. 手动删除失败的 pods: kubectl delete pod <pod-name> -n <namespace>"
    echo "  3. 查看详细错误: kubectl describe pod <pod-name> -n <namespace>"
fi

echo ""
echo -e "${GREEN}脚本执行完成！${NC}"

exit 0

