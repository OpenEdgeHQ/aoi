# ITBench SRE 全量测试最终报告：protocol-v2

日期：2026-10-11，America/Los_Angeles。模型与题库未变；只调整测试接入。

## 结论

31 题均有终态 ledger：28 题正常完成模型运行，1 题（116）模型阶段超时，2 题（46、63）因环境缺少目标资源未启动模型。正常完成的 28 题全部提交；提交表示模型结束决策，不表示修复正确。

本地快照严格对比结果：44 题的注入字段恢复到 baseline，且保存快照中的应用工作负载均 Ready；其余 27 题仍有至少一个注入相关配置字段与 baseline 不同。该检查不覆盖业务流量，也可能把其他有效修复方式记为不匹配，不能换算为官方成功率。官方 mitigation 判分未接入，official success 全部 null。

## 固定配置与运行范围

gpt-4o-mini，temperature=0，14 iterations，context=4096，output=5000；ITBench revision 333afa05c421b4e02a47b6c1b51686d20ce40cdd；AOI revision 284debaa64a43565d38148fbf7d59470b638714c。使用 kind-dev，未操作原 kind 集群。

开始 UTC 2026-10-11T18:12:49.652445+00:00，结束 UTC 2026-10-12T01:17:15.338495+00:00。本地 10-11 11:12 至 18:17，墙上时间约 7 小时 4 分钟。运行记录中存在墙上时间与 monotonic 时间差异，可能受机器暂停/休眠影响；不将该原因作为已确认结论。2 题的具体差异见 validation.json 和 CSV 的两个时间字段。

## Tokens、工具和费用

有完整记录的 28 题：provider tokens 14,526,794（输入 13,674,782，输出 852,012）；provider calls 2039；tool calls 1112，工具错误 397。逐题调用总和及 cost 算式均核验通过。

已知估算费用 $2.562424，按输入 $0.15/百万、输出 $0.60/百万估算，未扣缓存折扣。116 题在外部 1800 秒超时取消后没有写出 task.json，它已发生的 tokens/cost 未知；因此上述金额仅覆盖 28 题，不能声称是全部29次模型试验总费用。

## 测试接入修正与限制

- 移除额外的 repair/read 提交门槛，允许 AOI 自行决定 submit；没有调整 AOI 定位、规划、提示词或修复策略。
- 使用正确 /prometheus 路径，并在 preflight 检查告警 API；部署后检查题目目标存在，并等待 baseline rollout 就绪。
- 本轮 get_alerts 调用 33 次，404 为 0 次。RBAC Forbidden 工具调用 236 次；使用上游受限 agent 权限，未扩大权限。这些失败作为工具执行结果保存。
- 快照增加 Namespace 标签、Istio 策略、PVC/HPA 等；local-review 只比较已捕获字段和工作负载状态，不是官方判分。
- 场景42保留已披露的本地抢占注入适配：目标 Kafka 声明 CPU request=1，并验证目标可用性下降；未改变原模型预算。
- 46：Deployment/postgresql 不存在；63：Secret/email-service-secret 不存在。没有伪造资源，也没有在没有修复的情况下重复尝试。
- 116：达到固定 30 分钟模型阶段上限，保留错误日志和 ledger，不重跑。测试记录器现已补充取消捕获以保存后续试验已有数据；无法恢复这次丢失的 in-memory 记录，也没有编造 tokens、tool calls 或最终快照。

## 清理与交付

最终 50 个系统/基线 Pod 均 Running 且容器 Ready；otel-demo/book-info 应用 Pod 已删除，场景 restricted-kubeconfig 已删除。保留 Prometheus、ClickHouse、Istio、Chaos Mesh、Metrics Server、MetalLB 和网络策略控制器。

task-results-index.json / CSV：31题索引，缺失值为 null/空白，不记作0。local-review-index.json：28题完整本地核验。validation.json：字段覆盖、tokens、cost、时间和清理核验。scenarios/<id>/：原始记录、tool输出、trace、部署/注入/清理日志及快照。ZIP 排除凭证、.env、access.log 和锁文件。原始首轮结果保持独立。

## 逐题结果

| 场景 | 终态 | 提交 | 本地字段恢复 | 官方成功 | 已知Tokens | 估算费用($) |
|---|---|---|---|---|---:|
| 16 | executed_ungraded | True | False | null | 521423 | 0.088026 |
| 20 | executed_ungraded | True | False | null | 318880 | 0.057663 |
| 23 | executed_ungraded | True | False | null | 476295 | 0.08315 |
| 24 | executed_ungraded | True | False | null | 366066 | 0.06384 |
| 30 | executed_ungraded | True | False | null | 485247 | 0.090049 |
| 31 | executed_ungraded | True | False | null | 423221 | 0.071668 |
| 33 | executed_ungraded | True | False | null | 836066 | 0.140304 |
| 34 | executed_ungraded | True | False | null | 725155 | 0.139835 |
| 36 | executed_ungraded | True | False | null | 614516 | 0.102859 |
| 39 | executed_ungraded | True | False | null | 382592 | 0.067779 |
| 40 | executed_ungraded | True | False | null | 407857 | 0.071832 |
| 42 | executed_ungraded | True | False | null | 281606 | 0.05929 |
| 43 | executed_ungraded | True | False | null | 411888 | 0.07072 |
| 44 | executed_ungraded | True | True | null | 976060 | 0.160366 |
| 45 | executed_ungraded | True | False | null | 951796 | 0.165007 |
| 46 | environment_error | None | None | null | None | None |
| 47 | executed_ungraded | True | False | null | 928144 | 0.158021 |
| 48 | executed_ungraded | True | False | null | 421988 | 0.075447 |
| 49 | executed_ungraded | True | False | null | 367947 | 0.063321 |
| 50 | executed_ungraded | True | False | null | 357299 | 0.06166 |
| 53 | executed_ungraded | True | False | null | 369028 | 0.067294 |
| 56 | executed_ungraded | True | False | null | 405372 | 0.069618 |
| 57 | executed_ungraded | True | False | null | 519728 | 0.095084 |
| 58 | executed_ungraded | True | False | null | 497531 | 0.085644 |
| 59 | executed_ungraded | True | False | null | 324716 | 0.060157 |
| 60 | executed_ungraded | True | False | null | 393166 | 0.067229 |
| 63 | environment_error | None | None | null | None | None |
| 102 | executed_ungraded | True | False | null | 874533 | 0.173216 |
| 105 | executed_ungraded | True | False | null | 327384 | 0.059003 |
| 114 | executed_ungraded | True | False | null | 561290 | 0.094342 |
| 116 | model_error | None | None | null | None | None |

## 最终检查

当前 ITBench 相关 55 项单元测试通过；独立模拟取消运行检查通过，确认记录器可以写出错误和已有记录，official success 仍为 null。ZIP 完整性及凭证排除检查通过。
