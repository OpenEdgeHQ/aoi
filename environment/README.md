# AIOpsLab 环境服务器与客户端使用文档

## 目录
- [概述](#概述)
- [架构设计](#架构设计)
- [环境服务器 (aiopslab_server.py)](#环境服务器-aiopslab_serverpy)
- [环境客户端 (aiopslab_client.py)](#环境客户端-aiopslab_clientpy)
- [完整工作流示例](#完整工作流示例)
- [常见问题](#常见问题)

---

## 概述

AIOpsLab 环境系统由服务器和客户端两部分组成,提供了一个完整的 Kubernetes 故障诊断和修复的测试环境。

- **环境服务器** (`aiopslab_server.py`): 基于 FastAPI 和 AIOpsLab Orchestrator 的后端服务,负责管理 Kind 集群、会话和问题实例
- **环境客户端** (`aiopslab_client.py`): Python 客户端库,提供简洁的 API 与服务器交互

### 主要特性

✅ **会话管理**: 支持多个并发会话,每个会话独立管理一个问题实例  
✅ **检查点/回滚**: 支持创建检查点并回滚到之前的状态  
✅ **动作执行**: 执行 kubectl、get_metrics 等诊断和修复动作  
✅ **解决方案提交**: 提交并评估解决方案  
✅ **端口管理**: 自动管理 Prometheus 端口转发  
✅ **集群管理**: 自动创建和管理 Kind 集群  
✅ **性能追踪**: 自动记录从初始化到提交的总时长 (`total_duration_seconds`)  

---

## 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                     多智能体系统                           │
│  (main_aiopslab.py / main.py / 自定义 Agent)             │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP API 调用
                     ▼
┌──────────────────────────────────────────────────────────┐
│              EnvironmentClient (客户端)                   │
│  • init_problem()    • execute_action()                  │
│  • submit_solution() • create_checkpoint()               │
│  • rollback()        • get_session_status()              │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP 请求 (REST API)
                     ▼
┌──────────────────────────────────────────────────────────┐
│         EnvironmentServer (FastAPI 服务器)                │
│  ┌────────────────────────────────────────────────────┐  │
│  │  OrchestratorSession (会话管理)                     │  │
│  │  • Orchestrator (问题初始化和执行)                  │  │
│  │  • Checkpoint (检查点管理)                          │  │
│  │  • RemoteAgent (动作队列)                           │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────────────────┘
                     │ kubectl / API 调用
                     ▼
┌──────────────────────────────────────────────────────────┐
│              Kind Kubernetes 集群                         │
│  • Pods  • Services  • Deployments                       │
│  • Prometheus  • OpenEBS  • 应用程序                      │
└──────────────────────────────────────────────────────────┘
```

---

## 环境服务器 (aiopslab_server.py)

### 1. 功能说明

环境服务器基于 **FastAPI** 和 **AIOpsLab Orchestrator** 构建,提供 RESTful API 接口。

**核心组件:**

1. **ServerConfig**: 服务器配置管理
2. **EnvironmentServer**: 主服务器类,管理会话和集群
3. **OrchestratorSession**: 单个问题会话的封装
4. **Checkpoint**: 检查点管理,支持状态保存和恢复
5. **RemoteAgent**: 代理类,用于与 Orchestrator 交互

### 2. 启动服务器

#### 基本启动

```bash
# 使用默认配置 (host=127.0.0.1, port=8002)
python environment/aiopslab_server.py
```

#### 自定义配置

```bash
# 指定端口和主机
python environment/aiopslab_server.py --host 0.0.0.0 --port 8080

# 指定 Kind 集群名称
python environment/aiopslab_server.py --cluster-name my-cluster

# 指定 Kind 配置文件
python environment/aiopslab_server.py --kind-config /path/to/kind-config.yaml

# 服务器关闭时自动删除集群
python environment/aiopslab_server.py --auto-delete
```

#### 完整命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | str | 127.0.0.1 | 服务器监听地址 |
| `--port` | int | 8002 | 服务器端口 |
| `--cluster-name` | str | kind | Kind 集群名称 |
| `--kind-config` | str | AIOpsLab/kind/kind-config-x86.yaml | Kind 配置文件路径 |
| `--auto-delete` | flag | False | 关闭时自动删除集群 |

### 3. API 端点说明

服务器启动后,可访问:
- **API 文档**: http://127.0.0.1:8002/docs (Swagger UI)
- **备用文档**: http://127.0.0.1:8002/redoc (ReDoc)

#### 会话管理 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 获取服务器信息和活动会话列表 |
| `/health` | GET | 健康检查 |
| `/init_problem` | POST | 初始化问题实例 |
| `/execute_action` | POST | 执行动作 (kubectl, get_metrics 等) |
| `/session/{session_id}/status` | GET | 获取会话状态 |
| `/session/{session_id}/submit_format` | GET | 获取提交格式要求 |
| `/session/{session_id}` | DELETE | 清理会话 |

#### 检查点管理 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/checkpoint` | POST | 创建检查点 |
| `/rollback` | POST | 回滚到指定检查点 |

#### 提交 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/submit` | POST | 提交解决方案 |

#### 维护 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/port/cleanup` | POST | 清理所有端口转发 |
| `/port/status` | GET | 查看端口状态 |

### 4. 请求/响应模型

#### InitProblemRequest

```json
{
  "problem_id": "k8s_target_port-misconfig-detection-1",
  "reset_if_exists": false
}
```

#### ActionRequest

```json
{
  "session_id": "e3dd22b0-af93-402e-a581-06441763a80b",
  "action": "exec_shell(\"kubectl get pods\")"
}
```

#### SubmitSolutionRequest

```json
{
  "session_id": "e3dd22b0-af93-402e-a581-06441763a80b",
  "solution": {
    "problem": "Service target port mismatch",
    "fix": "Changed targetPort from 8080 to 80"
  }
}
```

#### EnvironmentResponse

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "timestamp": "2025-10-09T02:00:00"
}
```

### 5. 服务器生命周期

```python
# 启动时:
1. 清理所有端口转发
2. 检查/创建 Kind 集群
3. 导出 kubeconfig 到 ~/.kube/config
4. 启动 FastAPI 服务器

# 关闭时:
1. 清理所有活动会话
2. 清理端口转发
3. 可选: 删除 Kind 集群 (--auto-delete)
4. 清理临时 kubeconfig 文件
```

---

## 环境客户端 (aiopslab_client.py)

### 1. 类初始化

#### 基本用法

```python
from environment.aiopslab_client import EnvironmentClient

# 方式 1: 使用默认配置 (127.0.0.1:8000)
client = EnvironmentClient()

# 方式 2: 指定服务器地址
client = EnvironmentClient(host="127.0.0.1", port=8002)

# 方式 3: 使用完整 URL
client = EnvironmentClient(server_url="http://127.0.0.1:8002")

# 方式 4: 连接到已存在的会话
client = EnvironmentClient(
    host="127.0.0.1",
    port=8002,
    session_id="e3dd22b0-af93-402e-a581-06441763a80b"
)
```

#### 设置全局默认服务器

```python
# 为所有新创建的客户端设置默认服务器
EnvironmentClient.set_default_server(host="127.0.0.1", port=8002)

# 之后创建的客户端会自动使用该配置
client = EnvironmentClient()
```

### 2. 主要方法

#### 2.1 连接管理

##### `check_connection() -> bool`

检查与服务器的连接状态。

```python
if client.check_connection():
    print("✅ 服务器连接正常")
else:
    print("❌ 无法连接到服务器")
```

---

#### 2.2 问题初始化

##### `init_problem(problem_id: str, reset_if_exists: bool = False) -> Dict`

初始化一个问题实例。

**参数:**
- `problem_id`: 问题 ID (例如: `"k8s_target_port-misconfig-detection-1"`)
- `reset_if_exists`: 如果该问题已存在会话,是否重置

**返回值:**
```python
{
    "session_id": "uuid...",
    "problem_id": "...",
    "task_description": "...",
    "instructions": "...",
    "available_actions": {...},
    "checkpoints": ["initial"]
}
```

**示例:**

```python
result = client.init_problem("k8s_target_port-misconfig-detection-1")

if result:
    print(f"Session ID: {client.session_id}")
    print(f"Task: {client.task_description}")
    print(f"Available actions: {list(client.available_actions.keys())}")
```

##### `connect_session(session_id: str) -> Dict`

连接到已存在的会话。

```python
result = client.connect_session("e3dd22b0-af93-402e-a581-06441763a80b")

if result:
    print(f"✅ 已连接到会话: {client.problem_id}")
    print(f"历史记录长度: {result['history_length']}")
    print(f"检查点数量: {len(client.checkpoints)}")
```

---

#### 2.3 动作执行

##### `execute_action(action: str) -> Dict`

执行诊断或修复动作。

**常用动作格式:**

```python
# 1. 执行 shell 命令
result = client.execute_action('exec_shell("kubectl get pods")')

# 2. 获取指标数据
result = client.execute_action('get_metrics("pod_cpu_usage")')

# 3. 查看日志
result = client.execute_action('exec_shell("kubectl logs pod-name")')

# 4. 修改配置
result = client.execute_action('exec_shell("kubectl apply -f config.yaml")')

# 5. 提交解决方案 (通过 execute_action)
result = client.execute_action('submit("Fixed the targetPort mismatch")')
```

**返回值:**

```python
{
    "result": "输出内容...",
    "is_complete": False,  # 是否完成问题（仅在提交时为 True）
    "session_id": "...",
    "step_count": 5,
    "is_submission": False,  # 是否为提交动作
    "evaluation": None  # 如果是提交动作且成功，包含完整的评估结果（含 total_duration_seconds）
}
```

**示例:**

```python
# 获取 Pod 列表
response = client.execute_action('exec_shell("kubectl get pods -A")')
print(response['result'])

# 检查 Service 配置
response = client.execute_action('exec_shell("kubectl get svc -o yaml")')
print(response['result'])

# 查看 Prometheus 指标
response = client.execute_action('get_metrics("container_memory_usage")')
print(response['result'])
```

---

#### 2.4 提交解决方案

##### `submit_solution(solution: Union[str, List, Dict]) -> Dict`

提交解决方案 (推荐使用此方法而非 `execute_action`)。

**参数:**
- `solution`: 解决方案,可以是字符串、列表或字典

**返回值:**

```python
{
    "session_id": "...",
    "status": "submitted",
    "is_valid": True,  # 是否为有效提交
    "submission_result": "VALID",  # VALID / INVALID / SUBMITTED
    "solution": {...},
    "evaluation": {  # 评估结果（完整版）
        # 检测任务 (Detection Task)
        "Detection Accuracy": "Correct",  # "Correct" / "Incorrect"
        "TTD": 12.34,  # Time to Detect (秒)
        "steps": 5,    # 检测步数
        "in_tokens": 1234,   # 输入 token 数
        "out_tokens": 567,   # 输出 token 数
        "total_duration_seconds": 15.67,  # 从初始化到提交的总时长（秒）
        
        # 或定位任务 (Localization Task)
        "Localization Accuracy": 100.0,
        "TTL": 23.45,  # Time to Locate (秒)
        "success": True,
        "total_duration_seconds": 25.67,
        
        # 或修复任务 (Mitigation Task)
        "success": True,
        "TTM": 34.56,  # Time to Mitigate (秒)
        "total_duration_seconds": 40.12
    },
    "message": "...",
    "timestamp": "..."
}
```

**关键字段说明:**
- `is_valid`: 解决方案是否正确
- `evaluation`: 详细的评估结果，根据任务类型不同包含不同的字段：
  - **Detection Task**: `Detection Accuracy`, `TTD`, `steps`, `in_tokens`, `out_tokens`
  - **Localization Task**: `Localization Accuracy`, `TTL`, `success`
  - **Mitigation Task**: `success`, `TTM`
- `total_duration_seconds`: 从问题初始化成功到提交解决方案的总时长（秒），由服务器自动计算

**示例:**

```python
# 方式 1: 字符串描述
solution = "Changed the service targetPort from 8080 to 80 to match container port"
result = client.submit_solution(solution)

# 方式 2: 结构化数据
solution = {
    "problem": "Service targetPort mismatch",
    "root_cause": "Service targetPort was 8080 but container listens on port 80",
    "fix": "Updated service.yaml targetPort to 80",
    "verification": "Tested service connectivity, working now"
}
result = client.submit_solution(solution)

# 检查提交结果
if result and result['is_valid']:
    print("🎉 解决方案有效!")
    eval_results = result['evaluation']
    print(f"评估结果: {eval_results}")
    
    # 提取关键指标
    if 'Detection Accuracy' in eval_results:
        print(f"检测准确性: {eval_results['Detection Accuracy']}")
        print(f"检测时间 (TTD): {eval_results['TTD']:.2f}秒")
    
    # 显示总时长
    if 'total_duration_seconds' in eval_results:
        print(f"总时长 (初始化→提交): {eval_results['total_duration_seconds']:.2f}秒")
else:
    print("❌ 解决方案无效,需要重试")
```

##### `get_submit_format() -> Dict`

获取提交格式要求。

```python
format_info = client.get_submit_format()

print(f"提交 API: {format_info['submit_api']['api_name']}")
print(f"描述: {format_info['submit_api']['description']}")
print(f"指引: {format_info['instructions']}")
```

---

#### 2.5 检查点和回滚

##### `create_checkpoint(checkpoint_name: Optional[str] = None) -> Dict`

创建检查点以保存当前状态。

```python
# 自动命名 (checkpoint_1, checkpoint_2, ...)
result = client.create_checkpoint()

# 自定义名称
result = client.create_checkpoint("after_diagnosis")

print(f"✅ 检查点已创建: {result['checkpoint_name']}")
print(f"总检查点数: {result['total_checkpoints']}")
print(f"检查点列表: {result['checkpoints']}")
```

##### `rollback(checkpoint_name: Optional[str] = None) -> Dict`

回滚到指定检查点。

```python
# 回滚到最近的检查点
result = client.rollback()

# 回滚到指定检查点
result = client.rollback("after_diagnosis")

if result and result['success']:
    print(f"✅ 已回滚到: {result['checkpoint_name']}")
    print(f"历史长度: {result['history_length']}")
    print(f"剩余检查点: {result['remaining_checkpoints']}")
```

**注意:** 回滚会:
1. 恢复 Kubernetes 集群状态
2. 恢复会话历史
3. 删除该检查点之后创建的所有检查点
4. 重置提交状态

---

#### 2.6 会话状态查询

##### `get_session_status() -> Dict`

获取当前会话的详细状态。

```python
status = client.get_session_status()

print(f"会话 ID: {status['session_id']}")
print(f"问题 ID: {status['problem_id']}")
print(f"是否激活: {status['is_active']}")
print(f"历史长度: {status['history_length']}")
print(f"是否已提交: {status['is_submitted']}")
print(f"提交结果: {status['submission_result']}")
print(f"检查点列表: {status['checkpoints']}")
```

##### `get_history(last_n: Optional[int] = None) -> List[Dict]`

获取会话历史记录。

```python
# 获取所有历史
history = client.get_history()

# 获取最近 5 条历史
history = client.get_history(last_n=5)

for entry in history:
    print(f"{entry['role']}: {entry['content'][:100]}...")
```

##### `is_problem_solved() -> bool`

检查问题是否已解决。

```python
if client.is_problem_solved():
    print("✅ 问题已成功解决!")
else:
    print("问题尚未解决,继续诊断...")
```

---

#### 2.7 会话清理

##### `cleanup_session() -> bool`

清理当前会话,释放资源。

```python
if client.cleanup_session():
    print("✅ 会话已清理")
```

##### `reset_environment() -> bool`

重置整个环境 (清理所有会话)。

```python
if client.reset_environment():
    print("✅ 环境已重置")
```

##### `list_sessions() -> List[Dict]`

列出所有活动会话。

```python
sessions = client.list_sessions()

for session in sessions:
    print(f"Session: {session['session_id']}")
    print(f"  Problem: {session['problem_id']}")
    print(f"  Active: {session['is_active']}")
```

---

### 3. 客户端属性

客户端对象维护以下属性:

| 属性 | 类型 | 说明 |
|------|------|------|
| `server_url` | str | 服务器 URL |
| `session_id` | str | 当前会话 ID |
| `problem_id` | str | 当前问题 ID |
| `task_description` | str | 任务描述 |
| `instructions` | str | 操作指引 |
| `available_actions` | Dict | 可用动作列表 |
| `checkpoints` | List[str] | 检查点名称列表 |
| `submit_api_info` | Dict | 提交 API 信息 |
| `is_submitted` | bool | 是否已提交 |
| `submission_result` | str | 提交结果 (VALID/INVALID/SUBMITTED) |

---

## 完整工作流示例

### 示例 1: 基本问题解决流程

```python
from environment.aiopslab_client import EnvironmentClient

# 1. 设置默认服务器
EnvironmentClient.set_default_server(host="127.0.0.1", port=8002)

# 2. 创建客户端并初始化问题
client = EnvironmentClient()
result = client.init_problem("k8s_target_port-misconfig-detection-1")

if not result:
    print("❌ 问题初始化失败")
    exit(1)

print(f"✅ 问题已初始化")
print(f"📝 任务: {client.task_description}")
print(f"🎯 Session ID: {client.session_id}")

# 3. 执行诊断动作
print("\n=== 开始诊断 ===")

# 3.1 查看 Pod 状态
response = client.execute_action('exec_shell("kubectl get pods -A")')
print(f"Pods:\n{response['result']}")

# 3.2 查看 Service 配置
response = client.execute_action('exec_shell("kubectl get svc -o yaml")')
print(f"Services:\n{response['result'][:500]}...")

# 3.3 创建诊断后的检查点
client.create_checkpoint("after_diagnosis")

# 3.4 查看 Prometheus 指标
response = client.execute_action('get_metrics("pod_status")')
print(f"Metrics:\n{response['result']}")

# 4. 尝试修复
print("\n=== 尝试修复 ===")

# 假设发现了 targetPort 配置错误
fix_command = 'exec_shell("kubectl patch svc example-service -p \'{\"spec\":{\"ports\":[{\"port\":80,\"targetPort\":80}]}}\'")'
response = client.execute_action(fix_command)
print(f"修复结果: {response['result']}")

# 5. 验证修复
response = client.execute_action('exec_shell("kubectl get svc example-service -o yaml")')
print(f"验证配置: {response['result'][:300]}...")

# 6. 提交解决方案
print("\n=== 提交解决方案 ===")

solution = {
    "problem": "Service targetPort mismatch",
    "root_cause": "Service targetPort was 8080, but container listens on port 80",
    "fix": "Changed service targetPort from 8080 to 80 using kubectl patch",
    "verification": "Checked service YAML, targetPort now matches container port"
}

submit_result = client.submit_solution(solution)

if submit_result and submit_result['is_valid']:
    print("🎉 问题已成功解决!")
    eval_results = submit_result['evaluation']
    print(f"评估结果: {eval_results}")
    
    # 显示性能指标
    if 'Detection Accuracy' in eval_results:
        print(f"✅ 检测准确性: {eval_results['Detection Accuracy']}")
        print(f"⏱️  检测时间 (TTD): {eval_results['TTD']:.2f}秒")
        print(f"📊 执行步数: {eval_results['steps']}")
    
    # 显示总时长（从初始化到提交）
    if 'total_duration_seconds' in eval_results:
        duration = eval_results['total_duration_seconds']
        print(f"⏱️  总时长 (初始化→提交): {duration:.2f}秒 ({duration/60:.2f}分钟)")
else:
    print("❌ 解决方案无效")
    # 回滚到诊断后的状态重试
    client.rollback("after_diagnosis")
    print("已回滚到诊断后的状态,可以重新尝试修复")

# 7. 清理会话
client.cleanup_session()
print("✅ 会话已清理")
```

---

### 示例 2: 使用检查点进行多次尝试

```python
from environment.aiopslab_client import EnvironmentClient

client = EnvironmentClient(host="127.0.0.1", port=8002)
client.init_problem("k8s_target_port-misconfig-mitigation-1")

# 创建初始检查点
client.create_checkpoint("before_any_changes")

# 尝试方案 1
print("\n=== 尝试方案 1 ===")
client.execute_action('exec_shell("kubectl scale deployment example-app --replicas=3")')
client.create_checkpoint("after_scale")

solution_1 = "Scaled deployment to 3 replicas"
result = client.submit_solution(solution_1)

if not result['is_valid']:
    print("方案 1 失败,回滚并尝试方案 2")
    client.rollback("before_any_changes")
    
    # 尝试方案 2
    print("\n=== 尝试方案 2 ===")
    client.execute_action('exec_shell("kubectl patch svc example-service --type=json -p=\'[{\"op\":\"replace\",\"path\":\"/spec/ports/0/targetPort\",\"value\":80}]\'")')
    
    solution_2 = {
        "problem": "Service targetPort mismatch",
        "fix": "Updated targetPort to 80 using kubectl patch"
    }
    result = client.submit_solution(solution_2)
    
    if result['is_valid']:
        print("🎉 方案 2 成功!")
    else:
        print("方案 2 也失败,需要重新分析")

client.cleanup_session()
```

---

### 示例 3: 连接到现有会话

```python
from environment.aiopslab_client import EnvironmentClient

# 场景: Agent 中断后重新连接
existing_session_id = "e3dd22b0-af93-402e-a581-06441763a80b"

client = EnvironmentClient(
    host="127.0.0.1",
    port=8002,
    session_id=existing_session_id
)

# 或者
client = EnvironmentClient(host="127.0.0.1", port=8002)
client.connect_session(existing_session_id)

# 查看会话状态
status = client.get_session_status()
print(f"问题 ID: {status['problem_id']}")
print(f"历史长度: {status['history_length']}")
print(f"检查点: {status['checkpoints']}")

# 查看最近 5 条历史
history = client.get_history(last_n=5)
for entry in history:
    print(f"{entry['role']}: {entry['content'][:80]}...")

# 继续执行动作
if not client.is_submitted:
    response = client.execute_action('exec_shell("kubectl get events")')
    # ...继续诊断和修复
```

---

### 示例 4: 在多智能体系统中使用

```python
from environment.aiopslab_client import EnvironmentClient

class AIOPlatform:
    def __init__(self, problem_id: str):
        self.client = EnvironmentClient(host="127.0.0.1", port=8002)
        self.problem_id = problem_id
        
    def run(self):
        # 初始化问题
        result = self.client.init_problem(self.problem_id)
        if not result:
            raise Exception("Failed to initialize problem")
        
        # 创建初始检查点
        self.client.create_checkpoint("initial")
        
        max_iterations = 10
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # 1. Observer 观察环境
            observations = self.observe_environment()
            
            # 2. Compressor 压缩上下文
            compressed = self.compress_context(observations)
            
            # 3. Observer 分析并决策
            decision = self.analyze_and_decide(compressed)
            
            # 4. Executor 执行动作
            if decision['action_type'] == 'submit':
                result = self.client.submit_solution(decision['solution'])
                if result['is_valid']:
                    print("🎉 问题解决成功!")
                    break
            else:
                self.execute_decision(decision)
            
            # 5. 每次迭代后创建检查点
            self.client.create_checkpoint(f"iteration_{iteration + 1}")
        
        # 清理
        self.client.cleanup_session()
    
    def observe_environment(self):
        # 收集各种观察数据
        pods = self.client.execute_action('exec_shell("kubectl get pods -A -o wide")')
        services = self.client.execute_action('exec_shell("kubectl get svc -A")')
        events = self.client.execute_action('exec_shell("kubectl get events --sort-by=.lastTimestamp")')
        
        return {
            "pods": pods['result'],
            "services": services['result'],
            "events": events['result']
        }
    
    def compress_context(self, observations):
        # 压缩上下文逻辑
        return compressed_observations
    
    def analyze_and_decide(self, context):
        # 使用 LLM 分析并决策
        return decision
    
    def execute_decision(self, decision):
        # 执行决策
        for action in decision['actions']:
            self.client.execute_action(action)


# 使用
platform = AIOPlatform("k8s_target_port-misconfig-detection-1")
platform.run()
```

---

## 常见问题

### Q1: 如何检查服务器是否正在运行?

```python
client = EnvironmentClient(host="127.0.0.1", port=8002)

if client.check_connection():
    print("✅ 服务器运行正常")
else:
    print("❌ 服务器未运行,请先启动服务器:")
    print("   python environment/aiopslab_server.py --port 8002")
```

### Q2: 如何查看所有活动会话?

```python
client = EnvironmentClient(host="127.0.0.1", port=8002)
sessions = client.list_sessions()

print(f"活动会话数: {len(sessions)}")
for session in sessions:
    print(f"  - {session['session_id']}: {session['problem_id']}")
```

### Q3: 如何手动清理端口转发?

**方法 1: 使用 API**
```python
import requests

response = requests.post("http://127.0.0.1:8002/port/cleanup")
print(response.json())
```

**方法 2: 使用命令行**
```bash
curl -X POST http://127.0.0.1:8002/port/cleanup
```

**方法 3: 使用 PortManager (服务器端)**
```python
from utils.port_manager import PortManager

PortManager.cleanup_all_port_forwards()
```

### Q4: 如何查看端口状态?

```python
import requests

response = requests.get("http://127.0.0.1:8002/port/status")
port_status = response.json()

for port_info in port_status['ports']:
    status = "可用" if port_info['available'] else "占用"
    print(f"端口 {port_info['port']}: {status}")
```

### Q5: execute_action 和 submit_solution 的区别?

- **`execute_action()`**: 通用动作执行,可以执行任何动作 (包括 `submit`)
- **`submit_solution()`**: 专门用于提交解决方案,更清晰和类型安全

**推荐用法:**
```python
# 诊断和修复: 使用 execute_action
client.execute_action('exec_shell("kubectl get pods")')

# 提交解决方案: 使用 submit_solution
client.submit_solution({"problem": "...", "fix": "..."})
```

### Q6: 如何处理长时间运行的操作?

某些操作 (如 `get_metrics`) 可能需要较长时间,客户端会等待服务器响应。

```python
import time

print("正在获取指标数据...")
start = time.time()

response = client.execute_action('get_metrics("pod_cpu_usage")')

elapsed = time.time() - start
print(f"操作耗时: {elapsed:.2f} 秒")
print(f"结果: {response['result'][:200]}...")
```

### Q7: total_duration_seconds 是如何计算的?

`total_duration_seconds` 字段记录从**问题初始化成功**到**提交解决方案**的总时长（秒）。

**计算方式:**
- **开始时间**: 调用 `init_problem()` 成功后记录的 `initialized_at` 时间戳
- **结束时间**: 调用 `submit_solution()` 或 `execute_action('submit(...)')` 时记录的时间戳
- **总时长**: `结束时间 - 开始时间` (秒)

**示例:**
```python
# 初始化问题（开始计时）
client.init_problem("k8s_target_port-misconfig-detection-1")
# → initialized_at = 2025-10-09T10:00:00

# ... 执行多个诊断和修复动作 ...

# 提交解决方案（结束计时）
result = client.submit_solution("Fixed the port mismatch")
# → submission_time = 2025-10-09T10:05:30

# 评估结果中会包含
# total_duration_seconds = 330.0 (5分30秒)
```

**注意:**
- 这个时长是"墙上时间"（wall-clock time），包括所有操作的等待时间
- 与 `TTD` (Time to Detect) 不同，`TTD` 是 AIOpsLab 内部计算的任务完成时间
- `total_duration_seconds` 更适合用于评估整体性能和超时控制

### Q8: 如何在服务器重启后恢复会话?

**注意:** 当前版本不支持持久化,服务器重启后会话会丢失。

**建议:**
1. 在重要步骤后记录 `session_id` 和关键信息
2. 使用检查点保存状态
3. 在应用层实现重试逻辑

### Q9: 如何调试 API 调用?

**方法 1: 使用 Swagger UI**
```
浏览器访问: http://127.0.0.1:8002/docs
```

**方法 2: 启用详细日志**
```python
import logging

logging.basicConfig(level=logging.DEBUG)

client = EnvironmentClient(host="127.0.0.1", port=8002)
# 客户端会打印详细的请求和响应信息
```

**方法 3: 使用 curl 测试**
```bash
# 测试连接
curl http://127.0.0.1:8002/health

# 初始化问题
curl -X POST http://127.0.0.1:8002/init_problem \
  -H "Content-Type: application/json" \
  -d '{"problem_id": "k8s_target_port-misconfig-detection-1", "reset_if_exists": false}'
```

---

## 最佳实践

### 1. 始终使用检查点

在关键步骤创建检查点,便于回滚:

```python
# 诊断前
client.create_checkpoint("before_diagnosis")

# 诊断后
client.create_checkpoint("after_diagnosis")

# 修复前
client.create_checkpoint("before_fix")

# 修复后
client.create_checkpoint("after_fix")
```

### 2. 优雅地处理错误

```python
try:
    result = client.init_problem(problem_id)
    if not result:
        print("❌ 初始化失败")
        return
    
    # ... 执行操作 ...
    
except Exception as e:
    print(f"❌ 发生错误: {e}")
    
    # 尝试回滚
    if client.session_id and client.checkpoints:
        client.rollback()
        print("已回滚到上一个检查点")

finally:
    # 确保清理资源
    if client.session_id:
        client.cleanup_session()
```

### 3. 使用结构化的解决方案

```python
# 推荐: 结构化数据
solution = {
    "problem": "明确的问题描述",
    "root_cause": "根本原因分析",
    "fix": "具体的修复步骤",
    "verification": "验证方法",
    "commands_executed": [
        "kubectl patch ...",
        "kubectl get ..."
    ]
}

# 不推荐: 简单字符串
solution = "Fixed it"  # 缺乏详细信息
```

### 4. 监控会话状态

```python
# 定期检查会话状态
status = client.get_session_status()

if not status['is_active']:
    print("⚠️ 会话已失效,需要重新初始化")
    client.init_problem(problem_id)
```

---
