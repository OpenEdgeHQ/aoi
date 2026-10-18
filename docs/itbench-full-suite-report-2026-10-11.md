# ITBench SRE 全量运行最终报告

日期：2026-10-11（America/Los_Angeles）

## 结果概览

31 个 library SRE 场景：29 题执行模型，2 题环境错误未执行，模型运行异常 0 题。没有官方判分，所有 official success 和成功率为 null。

模型 gpt-4o-mini，temperature=0，每题最多 14 iterations，context=4096，output=5000；隔离集群 kind-dev。

Provider tokens：12,619,466（输入 11,795,948，输出 823,518）；provider calls：2155；tool calls：1223，其中工具错误 504；submitted：11/29（不代表修复成功）。

估算费用 $2.263501，按输入 $0.15/百万、输出 $0.60/百万计算，未扣缓存折扣，不是实际账单。

首轮 UTC 12:51:50 至 16:32:59，约 3 小时 41 分钟。最终重试结束 UTC 17:22:31（本地 10:22:31）。含重试及间隔的总历时约 4 小时 31 分钟。

## 影响结果的限制

- 本次告警工具遗漏 Prometheus /prometheus routePrefix，get_alerts() 返回 404 共 75 次。这影响模型的观测能力，不能将本次运行当作没有该问题的标准成绩。最终只读验证：/api/v1/alerts 为 404；/prometheus/api/v1/alerts 为 200。已修正后续配置，没有重跑已执行模型以改善分数。
- 46：当前 OpenTelemetry 应用没有题目指定的 Deployment/postgresql；63：没有指定的 Secret/email-service-secret。未伪造目标资源，保留 environment_error，不计为模型失败。
- 42：原注入器等待所有高优先级 Pod Ready，但资源饱和会留下 Pending Pod；改为验证目标不可用。Kafka 原来 CPU request=0，不能被 CPU 饱和抢占；注入时增加 1 CPU request，成功触发抢占后执行一次模型试验。这是本地环境适配，跨论文比较须披露。两次注入失败 ledger 保留在 history，日志追加写入。

## 记录核验与清理

31 个 case.json、29 个 task.json 和 raw-aoi-trace.json 完整。逐题验证固定配置、tool calls 的输出与时间字段、provider_usage_calls 与总 tokens 一致、cost 计算一致。
2 题的 UTC 起止差与 monotonic duration 有明显差异。CSV 同时保留 duration_seconds 和 wall_duration_seconds；未修改原始值，差异见 validation.json。
summary.json 的 mean/cost_coverage 统计面向官方已判分样本，因此为空或为 0；不能解读为没有保存费用。全量统计见 task-results-index.json。
应用 Pod 已清理，场景 restricted-kubeconfig 已删除；50 个基线/系统 Pod 全部 Running 且容器 Ready。Prometheus、ClickHouse、Istio、Chaos Mesh、Metrics Server、MetalLB、网络策略控制器保留；未对原 kind 集群执行操作。

本轮 ITBench 相关 30 个单元测试通过，Prometheus API 路径只读验证通过。没有官方 mitigation verdict，不将 submit、attempt.success、快照或本地检查视为官方成绩。

## 逐题结果

| 场景 | 状态 | Submitted | 官方成功 | Tokens | 记录耗时(s) | 估算费用($) |
|---|---|---|---|---:|---:|---:|
| 16 | executed_ungraded | True | null | 482407 | 188.456 | 0.080619 |
| 20 | executed_ungraded | True | null | 470476 | 212.973 | 0.079739 |
| 23 | executed_ungraded | True | null | 407723 | 197.0 | 0.071107 |
| 24 | executed_ungraded | True | null | 401607 | 166.634 | 0.067914 |
| 30 | executed_ungraded | False | null | 485224 | 219.44 | 0.083493 |
| 31 | executed_ungraded | False | null | 420310 | 185.305 | 0.071642 |
| 33 | executed_ungraded | False | null | 740062 | 858.848 | 0.163524 |
| 34 | executed_ungraded | False | null | 350206 | 175.736 | 0.060668 |
| 36 | executed_ungraded | True | null | 336774 | 319.893 | 0.062987 |
| 39 | executed_ungraded | False | null | 242695 | 206.414 | 0.045443 |
| 40 | executed_ungraded | False | null | 344948 | 196.459 | 0.060716 |
| 42 | executed_ungraded | False | null | 552057 | 496.346 | 0.098023 |
| 43 | executed_ungraded | False | null | 478630 | 254.422 | 0.083622 |
| 44 | executed_ungraded | False | null | 552562 | 306.341 | 0.096983 |
| 45 | executed_ungraded | False | null | 419344 | 273.422 | 0.075721 |
| 46 | environment_error | None | null | None | None | None |
| 47 | executed_ungraded | False | null | 390950 | 304.188 | 0.071557 |
| 48 | executed_ungraded | True | null | 454065 | 266.66 | 0.07896 |
| 49 | executed_ungraded | True | null | 427127 | 252.564 | 0.075488 |
| 50 | executed_ungraded | True | null | 405962 | 255.088 | 0.070949 |
| 53 | executed_ungraded | True | null | 336598 | 289.342 | 0.062094 |
| 56 | executed_ungraded | False | null | 718941 | 366.791 | 0.12578 |
| 57 | executed_ungraded | True | null | 371822 | 386.324 | 0.064622 |
| 58 | executed_ungraded | True | null | 353409 | 190.907 | 0.060741 |
| 59 | executed_ungraded | False | null | 351752 | 288.953 | 0.065465 |
| 60 | executed_ungraded | False | null | 437617 | 361.799 | 0.079241 |
| 63 | environment_error | None | null | None | None | None |
| 102 | executed_ungraded | False | null | 293288 | 231.766 | 0.054949 |
| 105 | executed_ungraded | False | null | 471580 | 214.321 | 0.080376 |
| 114 | executed_ungraded | False | null | 360058 | 222.619 | 0.063874 |
| 116 | executed_ungraded | False | null | 561272 | 440.495 | 0.107204 |

## 交付

task-results.csv / task-results-index.json 保存逐题索引。validation.json 保存核验证据。scenarios/<id>/ 保存原始 task、ledger、tool 输出、日志、快照与 trace。retry-environment-fixes.patch 保存本地注入修复。results.zip 排除 .env、kubeconfig、access.log、lock 和归档自身。
