# ITBench SRE mitigation

AOI can run against a locally deployed ITBench SRE scenario. ITBench owns
deployment, fault injection, restricted agent access, and grading. This adapter
reuses AOI's Observer, Probe, Executor, and Compressor. It does not treat an
AOI `submit()` call as a passing ITBench verdict.

## Validated local deployment

The integration was validated against ITBench commit `333afa05` on Kubernetes
1.34.0. Reproducible environment fixes are in
`benchmarks/itbench/patches/itbench-333afa05.patch`: scenario application enablement,
failed-injection propagation, missing-object validation, pinned telemetry image
and configuration corrections, and Jaeger database initialization.

For a clean checkout at that revision:

```sh
git -C /absolute/path/to/ITBench apply --check --unidiff-zero /absolute/path/to/aoi/benchmarks/itbench/patches/itbench-333afa05.patch
git -C /absolute/path/to/ITBench apply --unidiff-zero /absolute/path/to/aoi/benchmarks/itbench/patches/itbench-333afa05.patch
```

The existing local checkout already has these edits. Other revisions require
reviewing the patch first. Ansible was run with ITBench's Python 3.13 environment
and uv 0.12.24. The isolated Kind cluster uses MetalLB for gateway IPs;
Prometheus is exposed to the host with `kubectl port-forward`.

## Prepare ITBench

Follow the upstream [ITBench SRE scenario guide](https://github.com/itbench-hub/ITBench/blob/main/documentation/getting-started/scenarios.md).
Use a separate ITBench checkout, install its own dependencies, configure its
group variables, and start one SRE scenario. For example, from that checkout:

```sh
cd scenarios/sre
SCENARIO_NUMBER=1 make start-scenario
SCENARIO_NUMBER=1 make enable-agent-access
```

Find the restricted agent kubeconfig produced by ITBench's agent-access step.
The path depends on ITBench's `storage.local.directory` setting. Use that file,
not your personal `~/.kube/config`. Obtain the incident text and optional
Prometheus URL from the running scenario. Keep the scenario running until its
external evaluation is complete.

## Run AOI

From this AOI checkout:

```sh
.venv/bin/python -m benchmarks.itbench.run \
  --itbench-dir /absolute/path/to/ITBench \
  --scenario-number 1 \
  --namespace otel-demo \
  --incident 'Investigate and resolve the active SRE incident' \
  --kubeconfig /absolute/path/to/restricted-kubeconfig \
  --prometheus-url http://prometheus.example:9090
```

`--prometheus-url` is optional. If supplied, AOI may use `get_alerts()` and
`query_prometheus('PromQL')`. Kubernetes commands use
`exec_shell('kubectl ...')`; the adapter injects the restricted kubeconfig.
The runner uses `gpt-4o-mini` by default; use `--model` only to override it.
Probe rejects state-changing kubectl verbs. The scenario can also be started
with `--start-scenario`, which calls ITBench's `start-scenario` and
`enable-agent-access` Make targets after its checkout has been configured.
Use `--stop-scenario` only when the run and external evaluation are finished.

Pass `--namespace` for the scenario's application namespace (repeat it for
multiple namespaces). The batch runner forwards this metadata from the catalog;
the adapter generates examples for those namespaces instead of assuming
`otel-demo`. Without metadata, examples explicitly require replacing
`<incident-namespace>` with a namespace from the incident. Namespace metadata
is also saved in the task configuration.

Agents should prefer structured `exec_kubectl([...])` over nested shell strings.
JSON patches are one argument in the list. For a manifest, both
`exec_kubectl(['apply', '-n', 'book-info', '-f', '-'], stdin='...')` and
`exec_kubectl(['apply', '-f', '-', '-n', 'book-info'], stdin='...')` are supported.
`--filename -` and `--filename=-` are also accepted. Exactly one stdin filename
is required: file paths, multiple filename options and kustomize input cannot
be combined with `stdin`. Arguments and manifest text pass through unchanged.
Shell operators, interactive `kubectl edit`, remote shell execution, and
connection overrides are rejected. `submit()` records the agent's decision to
finish without imposing an additional repair/read gate. It never establishes
benchmark success. Tool errors and submission decisions are preserved for
evaluation without changing AOI's reasoning or repair strategy.

Each attempt writes `res/itbench/sre-<number>-<timestamp>.json` with the
task ID, ITBench revision, model and budgets, tool calls with outputs,
Observer/Probe/Executor/Compressor token usage, and elapsed time. Pass
`--input-usd-per-million` and `--output-usd-per-million` to record an estimated
USD cost for that model. The total uses provider-reported usage when AWorld
supplies it; the record also keeps per-call usage. Without usage or prices,
`cost_usd` is `null`, never an assumed zero.

The `success` field remains `null` until an external ITBench verifier provides
a JSON object containing a boolean `success` or `passed`. Supply its path with
`--verdict-json` if it already exists. If grading finishes later, attach it
without rerunning the agent:

```sh
.venv/bin/python -m benchmarks.itbench.verdict \
  res/itbench/sre-1-<timestamp>.json /absolute/path/to/itbench-verdict.json
```

AOI's submission and local command output are never used as an official
pass/fail score. Do not compare unverified attempts with published ITBench
success rates.

For archived scenario 1, `--local-verify` additionally records whether the
homepage flood flag is disabled and `flagd` / `load-generator` are running with
at least one ready replica at the current generation. It records
`local_verification`, with `official: false`, and leaves `success` unchanged.
It does not establish full symptom recovery or an official score. Run it independently:

```sh
.venv/bin/python -m benchmarks.itbench.verify \
  --kubeconfig /absolute/path/to/restricted-kubeconfig \
  --output res/itbench/runtime/local-check.json
```

Local check JSON is rejected by the official verdict importer. The public
[ITBench-Evaluations](https://github.com/itbench-hub/ITBench-Evaluations) toolkit
scores SRE RCA outputs with an LLM judge; its RCA scores are not a live
mitigation pass/fail verdict for this archived scenario.

After grading, summarize all local attempts with:

```sh
.venv/bin/python -m benchmarks.itbench.summary res/itbench --output res/itbench/summary.json
```

The summary counts unverified attempts separately and computes success rate
only over externally graded attempts.

## Full library SRE suite

`benchmarks.itbench.batch` selects all SRE scenarios in the pinned
`library/scenarios` catalog (31 in the validated checkout). The current upstream
Make/Ansible injector uses `archive/scenarios`; the batch checks matching fault
IDs and Kubernetes targets, and saves both definitions' hashes. Fault metadata
and ground truth are not included in agent incidents.

Each scenario uses fresh application namespaces on `kind-dev`. The runner
checks the admin context, node names, and ITBench inventory before cleanup.
The default model is gpt-4o-mini; override it with `--model`. Temperature 0
and the 14-iteration budget remain fixed.
Deployment, injection, model, and cleanup logs are saved separately. Setup
failures remain in the suite ledger rather than becoming model failures.

```sh
.venv/bin/python -m benchmarks.itbench.batch \
  --itbench-dir /absolute/path/to/ITBench \
  --admin-kubeconfig res/itbench/runtime/kind-dev-admin.yaml \
  --itbench-python /absolute/path/to/ITBench/.venv/bin/python \
  --docker-config /absolute/path/to/docker-config \
  --output res/itbench/full-library-gpt-4o-mini
```

You can set `UV_PYTHON` and `DOCKER_CONFIG` instead of passing those two options.

Resume with the same command and output directory. Completed cases are skipped;
`--retry-environment-errors` retries setup failures while preserving their earlier
ledgers. `progress.json` records the active scenario/phase, while `summary.json`
and `report.md` track finished cases. Completion or a blocked batch produces
`results.zip`, excluding credentials. Each model attempt retains full task-level
configuration, tools, provider usage, timing, and estimated cost.

NetworkPolicy faults need enforcement beyond Kind's default CNI. The tested
[policy-only kube-router configuration](../benchmarks/itbench/manifests/kube-router-policy.yaml)
keeps the existing CNI and kube-proxy. Its temporary-pod integration test confirmed
allow → deny → allow. The upstream scenario-31 injector is patched to use
`ingress: []`; an ingress rule with an empty ports list allows all ports.

## Testing protocol after the first full suite

The testing adapter accepts the agent's `submit()` without an extra repair/read
requirement. AOI reasoning, task planning and repair strategy are unchanged.
The batch checks the actual Prometheus alert API, validates catalog target
existence and waits for baseline deployment rollout before fault injection.
Missing targets remain environment errors; no substitute workloads are invented.

`benchmarks.itbench.review` compares saved baseline, injected and post-model
configuration fields and records workload readiness. The batch saves this as
`local-review.json` before cleanup. This is a strict comparison to baseline,
not an official mitigation verdict: alternative valid repairs and application
traffic are not scored, uncaptured changes remain unknown. Namespace labels
and Istio policies are included in new snapshots. Official success stays null.

The public ITBench-Evaluations toolkit evaluates RCA agent outputs rather than
providing an online SRE mitigation grader. The corrected testing protocol uses
a new output directory; the original suite and its limitations are preserved.

## Model pricing, resume identity, and ungraded execution statistics

The batch runner accepts any model name supported by the configured provider,
including `--model Qwen/Qwen3-14B`. Configure the provider/API endpoint separately
as for the single-task runner; this option does not start a local model server.
The default model remains `gpt-4o-mini`. Neither runner assumes prices from the
model name. Omit both pricing flags to keep `cost_usd: null`, including for local
Qwen runs. To estimate cost, explicitly supply both `--input-usd-per-million`
and `--output-usd-per-million` using the rates for your provider and model.
Rates must be finite and nonnegative; zero is an explicit zero price, not unknown.
The batch saves those rates in its manifest and forwards them to each task.

Resuming requires identical configuration (including effective `api_source`,
`llm_provider`, and `api_base`), catalog, AOI and ITBench commit IDs,
and SHA-256 hashes of both repositories' tracked changes (`git diff --binary HEAD`,
including staged changes). A mismatch or an older manifest missing these fields
requires a new `--output` directory. Rejected resumes preserve the existing
manifest, progress, summaries, and task records. Untracked files and installed
dependencies are not fingerprinted: commit experiment code and separately pin
runtime dependencies before collecting comparable results.

`summary.json.execution_stats` describes **all saved model task records**,
including ungraded and failed model attempts, but not environment-only case
ledgers. It reports `attempts`, `mean_duration_seconds`, `mean_tool_calls`,
`mean_total_tokens`, and `mean_cost_usd`, with per-metric `*_coverage` counts.
Missing measurements are excluded from their respective means; an empty sample
has a null mean. Explicit zero measurements remain included. These statistics
are still available when every official verdict is null.

Existing top-level `success_rate`, `mean_duration_seconds`, `mean_tool_calls`,
`mean_cost_usd`, and `cost_coverage` retain their graded-only definitions.
Top-level token/cost totals and `priced_attempts` still cover all input records.
Thus execution statistics do not turn submissions or ungraded attempts into
successes or failures, and cost means disclose the number of priced attempts.

Backend identity is resolved through `llm_config.py`, using the same `.env`,
environment overrides, and defaults as the task runner. With an empty `API_BASE`,
the recorded endpoint follows AWorld's `OPENAI_ENDPOINT`/OpenAI default fallback.
Changing `API_SOURCE` or the effective endpoint requires a new output directory,
even with the same model name. The checked source and endpoint are pinned in
the task subprocess environment and also recorded in each task configuration.
API keys are never included in backend identity; rotating a key alone does not
change the manifest.

## Mitigation evaluation and interruption-safe records

Enable `--mitigation-evaluation` for protocol `itbench_alert_resolution_v1`.
It records ground-truth file hashes and uses the declared alert names/labels
only in the external evaluator. The agent receives observed incident alerts,
not fault IDs, solutions or ground-truth entities.

A valid trial requires a healthy application baseline with traffic and healthy,
fresh Prometheus alert-rule evaluations. After injection, the expected alert
condition must appear in the application namespace (pending or firing); absent
conditions are environment errors, never assumed successful. After the model
ends, collect at least three observations spaced 30 seconds apart. Passing
requires all expected conditions absent, traffic/telemetry healthy, unchanged
alert rules, and baseline workloads still present with nonzero desired replicas,
current observed generations and fully updated/ready replicas. A post-model
check may wait at most 180 seconds. Setup checks wait at most 300 seconds.

This is a versioned local research protocol implementing alert-condition
resolution, not an official upstream ITBench verdict. It intentionally includes
pending conditions, before upstream firing delays expire; disclose this choice
in the paper. Official success remains null. Report `alert_resolution_rate`
only over trials with boolean mitigation evaluations; separately disclose
unknown/model-error/excluded cases and full 31-scenario coverage.

Scenarios 16/24 have no alert criterion in this pinned ground truth and are
excluded from this protocol. Missing catalog targets (currently 46/63) are
explicitly excluded after deployment validation; do not create fake targets.
The previous protocol-v2 run remains an exploratory run, not retrospectively
graded under this protocol.

`events.jsonl` journals configuration/start time, completed provider usage and
tool-call starts/results with fsync after each event. Graceful cancellation
writes task.json; after forced termination the batch recovers a partial task
from the journal. In-flight provider usage and tool outputs are unknown, not 0,
and recovered records are marked incomplete. The old scenario116 missing usage
cannot be recovered and remains disclosed.

## Smoke validation for tool-contract fixes

Before a new full suite, use a fresh output directory and the same model/budgets
for representative scenarios 16, 48, and 53. Verify telemetry and fault injection
first, and explicitly supply `--namespace otel-demo` for 16 and
`--namespace book-info` for 48/53 when using the single-task runner.

- 16: inspect the initial Probe command; it should no longer contain `--show-all`.
- 48: confirm the generated examples use `book-info`, not `otel-demo`. Restricted
  RBAC errors on Istio resources may still occur; this change does not grant permissions.
- 53: verify manifest calls with namespace options after `-f -` reach kubectl.
  Invalid manifests, immutable PVC fields, and unavailable storage classes remain
  separate failures and must be recorded rather than counted as successful repair.

Save task records and obtain an independent repair verdict before claiming a
success-rate change. Offline command replays only verify adapter acceptance.

## Access isolation and permission preflight

Reapply the updated `itbench-333afa05.patch` to a clean checkout of the pinned
ITBench revision. The revoke role now discovers RoleBindings and Roles across
all namespaces using **both** ITBench agent ownership labels, deletes those
objects before deleting the ServiceAccount, and leaves unrelated RBAC objects
alone. Batch cleanup also invokes `disable-agent-access` before removing apps.
Deleting a ServiceAccount alone is insufficient: bindings reference its name,
so a new account with the same name would inherit old namespace access.

Both runners check known repair operations with `kubectl auth can-i` using the
**restricted** kubeconfig before starting the model. Batch grants access and
checks it before fault injection; a denial produces an `environment_error` case
ledger without a model task record. Single-task runs save `<output>.preflight.json`
and stop before model configuration on denial; if the scenario was already
injected (including `--start-scenario`), the caller still owns its cleanup.
Batch saves `permission-preflight.json` in the case directory. No permissions
are automatically granted or expanded.

Checks currently cover namespace-scoped pod/deployment inspection and the known
repair operations for environment-variable changes (16), Namespace ambient
labels (48), and missing PVCs (53). They are not an exhaustive repair plan:
`unchecked_fault_ids` and `coverage: partial` disclose unsupported fault types.
In the default restricted setup, 48 is blocked because Namespace patch is absent.
Fault metadata is used only by the harness and is not added to model prompts.

Actions must be Python call **strings**. A list/object action now returns a
logged, retryable format error instead of raising an unrecorded `.strip()`
exception. For multiline `stdin`, use escaped newlines in a single-quoted Python
literal (e.g. `stdin='kind: ConfigMap\n'`) or a triple-quoted literal. Malformed
syntax is rejected with this guidance; the adapter does not guess how to repair
an ambiguous command. Rejected dispatches remain in `tool_calls` and tool-error
statistics. Existing read-only Probe and connection-override restrictions apply.

### Local validation (2026-10-11 PDT)

The [sanitized validation record](itbench-access-fixes-validation-2026-10-11.json)
contains 76 passing offline tests and the local Kubernetes checks. Granting
scenario 16, revoking twice, then granting 53 removes old namespace access while
preserving RBAC objects without both ownership labels. Scenario 48's real CLI
stops before model initialization and does not create a model task record.
Malformed command replays return logged errors; valid multiline retries apply
and read back a ConfigMap, while Probe writes remain blocked.

One fresh scenario 53 run with `gpt-4o-mini`, temperature 0 and 14 iterations
completed in 392.163 seconds (542,665 provider-reported tokens). The model
recreated the PVC with `standard`, completed the Deployment rollout, and passed
independent direct-v3 and productpage requests before harness baseline cleanup.
Its manifest call reached kubectl successfully. There were still 16 tool errors
across 46 dispatches, including immutable PVC patch and restricted diagnostic
queries; this is not a claim of error-free execution. Official success and cost
remain null. No full suite or success-rate comparison was performed.

After updating to upstream `43d5a59` (including merged PR #2 and the live-alert
protocol/durable journal), 87 offline tests pass. Additional checks cover
permission denial before injection in alert-evaluation mode and recovery of
malformed list/object actions from the journal. Recovery uses a canonical JSON
identity instead of hashing raw actions. A local adapter smoke verified logged
format rejection, stdin retry/readback, scenario 48 blocking, and preservation
of an unfinished object-valued dispatch during simulated interruption. No new
model inference or full-suite experiment was performed for this merge update;
the scenario 53 model result above retains its original code revision.

A subsequent **fresh real-API run at `a2bb64b`** validated the merged code with
`gpt-4o-mini`, temperature 0, and a 14-iteration limit. It completed in 384.532
seconds using 728,563 provider-reported tokens. All 67 provider usage entries
and 37 tool dispatches (8 errors) match the final task record exactly. The
journal contains trial start/end events and no unfinished tool dispatches;
replaying recovery preserves usage and tool results. The live process was not
interrupted, so this checks recovery replay rather than a real cancellation.

The model deleted the invalid PVC, removed its injected volume/mount, and
increased reviews-v3 to two replicas. Complete rollout and direct-v3/productpage
requests passed before cleanup. The harness then restored the original specs
and one replica. No stdin manifest or non-string command occurred in this fresh
model run; those paths retain their separate adapter/offline test evidence.
Official success and estimated cost remain null; no full suite or batch
live-alert evaluation was run.
