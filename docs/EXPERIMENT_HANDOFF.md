# AccelPact experiment and network-safety handoff

Last updated: 2026-08-29 (Asia/Shanghai)

## 1. Purpose and authorization

AccelPact is an authorized academic systems project. It tests stateful
stream/event/graph/allocator/collective protocols on accelerator servers owned
or administered by the research group. The current phase uses one NVIDIA A100
server and one Ascend 910B server. The user has explicitly confirmed research
authorization for these machines and asked the experiment to continue.

The project does **not** scan third-party networks, probe public services,
exploit software, bypass authentication, establish persistence, or move private
datasets. Its payloads are fixed integer generations, event dependencies,
small graph operations, seeds, logs, and JSONL results.

## 2. Why a Codex safeguard may have appeared

The exact trigger is not visible to this project and must not be guessed as a
fact. A plausible explanation is the combination of dual-use terms and tools:

- state-machine fuzzing, failure injection, abort, timeout, and deadlock;
- repeated SSH/SFTP/SCP commands across two accelerator hosts;
- graph-capture failure and communicator-recovery experiments;
- discussion of automatically minimizing and repairing failures.

Those patterns can resemble cybersecurity work even though this instance is a
bounded runtime-correctness experiment. Official OpenAI documentation states
that real-time safeguards can occasionally intervene in legitimate dual-use
work such as code review, debugging, defensive testing, and vulnerability
research:

<https://developers.openai.com/api/docs/guides/latest-model#safeguards>

Do not try to evade or disable a safeguard. If it appears again, pause new
network actions, present this handoff and the exact authorized scope, and resume
only actions allowed below.

## 3. Allowed network boundary

| Path | Purpose | Data allowed |
| --- | --- | --- |
| Local workstation → GitHub over HTTPS | Versioned public source/protocol/results | Repository files with no credentials or private topology |
| Local workstation → 910B gateway over authenticated SFTP/SSH | Deploy frozen source archives and retrieve results | Git archive, SHA-256, JSONL, stdout/stderr |
| 910B host → A100 host over the laboratory private 1 GbE link | Copy the same frozen archive and run the matched backend | Same archive and experiment artifacts |
| Local workstation → official documentation sites | Verify runtime semantics and prior work | Public search terms only |

Only existing authenticated connections may be used. Do not place passwords,
tokens, private keys, temporary bridge credentials, personal filesystem paths,
or exact private-network topology in this public repository. The GitHub push
used the workstation credential manager; no credential from chat text was
written to a command, file, or log.

## 4. Prohibited actions

- No port scanning, credential testing, exploit validation, or access to hosts
  outside the two authorized accelerator servers.
- No disabling product, OS, firewall, or network safeguards.
- No destructive filesystem cleanup, broad process kills, or device resets.
- No arbitrary internet access from either accelerator server.
- No model, user dataset, checkpoint, or large tensor transfer in AP-G0Q.
- No intentional deadlock without an outer timeout and an exact process scope.
- No publication claim based on an intentionally invalid negative control.

## 5. Repository and frozen revisions

- Public repository: <https://github.com/fanyules/accelpact>
- `62f1d18`: AP-G0Q protocol frozen before accelerator execution.
- `76c60c4`: initial backend-neutral oracle and TP1 runner.
- `850f222`: corrected CANN event lowering: normal multi-waiter Event and a
  fresh retained event pair per buffer generation.
- `bb25b98`: corrected cross-stream graph capture to use graph-internal Event;
  NPU `ExternalEvent` is reserved for host-side update/dispatch semantics.

All scientific continuation must use `bb25b98` or a later committed construct
fix that leaves the abstract invariants and thresholds unchanged.

## 6. Current environment boundary

| Platform | Qualified stack | AP-G0Q device scope |
| --- | --- | --- |
| A100 | PyTorch 2.11.0+cu128, CUDA 12.8 | TP1, one A100 PCIe 40 GB |
| 910B | PyTorch 2.10.0, torch-npu 2.10.0.post4, CANN 9.1.0 | TP1, one 910B4-1 64 GB |

Both use the native PyTorch caching allocator. AP-G0Q does not establish
`cudaMallocAsync` behavior. TP2 collectives belong to the next independently
frozen gate.

The deployed source directory for the current revision is:

```text
/data/AccelPact-bb25b98
```

Before any run, verify that the selected device has no compute process. After a
timeout or failure, verify the same condition again. Never reuse an old project
working tree that contains unrelated results.

## 7. Experiments already run

### Construct-only attempts, not scientific failures

Two initial 910B attempts exposed runner-contract mistakes:

1. `ExternalEvent` was incorrectly used for one record with two waiters; the
   current torch-npu contract permits only one wait per ExternalEvent record.
2. External rather than graph-internal events were used for cross-stream graph
   join, causing a timeout.

Both were corrected without changing the abstract protocol. The corrected
isolated tests completed 128/128 generations. These attempts remain engineering
notes and must not be reported as CANN protocol violations.

### Discovery at `bb25b98`

**A100**

- five valid event/buffer/graph paths passed;
- both negative controls were detected;
- `capture_abort_eager_recovery` failed: after deterministic capture
  invalidation, eager RNG raised `Offset increment outside graph capture
  encountered unexpectedly`; `graph.reset()` did not repair it in-process.

This matches the public failure class in PyTorch issue #171263 and is therefore
a regression seed, not a novel AccelPact discovery by itself.

**910B**

- all six valid paths passed, including capture-abort → eager recovery;
- missing cross-stream join was correctly rejected;
- the subsequent rebound-input negative control hit a harness error because the
  intentionally invalid missing-join capture left allocator capture bookkeeping
  active in that process.

The rebound control must be rerun alone in a fresh process. The preceding
missing-join poison is not yet classified as a runtime violation because the
test did not retain and explicitly reset the failed graph object.

## 8. Exact next actions

1. Run `rebound_input` alone on 910B in one fresh process. It must be detected
   without a harness error to finish oracle qualification.
2. Run `capture_abort_eager_recovery` in five additional sequential fresh
   processes on A100. Confirmation requires at least four of five failures.
3. Run the same litmus in five matched fresh 910B processes as a clean-control
   distribution. Do not reinterpret a known PyTorch regression as novelty.
4. Preserve every JSONL, log, timeout, exit code, stack version, and SHA-256.
5. Only after this adjudication, freeze a separate TP2 communicator-generation
   protocol. Collective abort destroys a communicator; recovery must create a
   new generation rather than reuse the aborted object.

Representative commands after entering the already authorized host:

```bash
# 910B container, isolated negative control
ASCEND_RT_VISIBLE_DEVICES=0 PYTHONPATH=/data/AccelPact-bb25b98/src \
  python3 /data/AccelPact-bb25b98/scripts/run_ap_g0q.py \
  --backend npu --iterations 128 --seed 20260829 \
  --litmus rebound_input --output RESULT.jsonl

# A100, isolated confirmation cell
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data/AccelPact-bb25b98/src \
  "$PYTHON" /data/AccelPact-bb25b98/scripts/run_ap_g0q.py \
  --backend cuda --iterations 128 --seed 20260829 \
  --litmus capture_abort_eager_recovery --output RESULT.jsonl
```

Wrap every command in a finite external timeout and use a new output filename;
the runner refuses to overwrite results.

## 9. Scientific decision boundary

The current evidence only qualifies the abstract oracle and reproduces one
known CUDA failure class. AccelPact earns a systems implementation only if later
experiments find at least two current-stack violations with different roots,
including one independent of vLLM and one causing a silent stale read, deadlock,
or poisoned fallback. At least one violation must arise from a pre-frozen
protocol-generated sequence rather than copying a public issue reproducer.

If all new findings are documented invalid usage or already-known issue
reproductions, stop at a regression/conformance suite; do not present it as an
ASPLOS system.

