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

## 8. Completed adjudication

Run ID: `ap_g0q_adjudication_20260829T192202CST`.

Every cell used a separate Python process, device 0, seed `20260829`, a
120-second outer timeout, and a unique output filename. `--iterations 128` was
supplied for contract consistency; both adjudication litmus tests are
non-iterative and do not loop over that value. Preflight and postflight checks
found no accelerator compute process on either platform.

**910B isolated negative control**

- `rebound_input` was detected in a fresh process with exit code 0,
  `negative_control_detected`, `replayable`, distinct captured and rebound
  addresses, and `stale_generation_detected=true`.
- The preceding discovery-run harness error is therefore resolved as process
  contamination from the earlier invalid missing-join capture, not an oracle
  miss and not a CANN runtime failure.

**Matched capture-abort recovery trials**

- A100: 5/5 trials returned exit code 2 and
  `protocol_violation -> poisoned`. Eager RNG recovery failed with the same
  capture-state error in every process, and diagnostic `graph.reset()` did not
  restore RNG operation. The device-synchronized litmus region measured
  27.883 +/- 2.845 ms (mean +/- sample SD, n=5).
- 910B: 5/5 trials returned exit code 0 and
  `valid_pass -> clean_fallback`. Eager RNG, allocator probe, new stream, and
  graph cleanup all succeeded. The same device-synchronized litmus region
  measured 351.991 +/- 10.950 ms (mean +/- sample SD, n=5).
- These timings document the execution boundary only; they are not a
  cross-platform performance comparison.

The two deployed trees have identical core files. After normalizing CRLF to LF,
their Git blob IDs exactly match commit `bb25b98`.

Evidence remains unchanged on both servers at:

```text
/data/AccelPact-bb25b98/results/ap_g0q_adjudication_20260829T192202CST
```

The retrieved local raw-work copy is ignored by Git at:

```text
results/raw_work/ap_g0q_adjudication_20260829T192202CST
```

Adjudication archive SHA-256:

- A100: `e165944cc6826b5d2aff1f1e6c93aef5fff3bec73cd9984c4d205fee30b4b94b`
- 910B: `78bc067516717412ec589d4b9747f34eaf955f6ffcdf6578cf98427dcbaab551`

All 54 file-level checksum entries in the retrieved evidence validated.

The original discovery JSONL and logs were also retrieved without modification:

- A100 archive: `370f922df8fca5c38275ec0feb8ff004715ce6f3c3e40f1c7ad0eec573d7a3df`
- 910B archive: `d99f7ffa82e85f0ce0f611ce3bd469cba09b66dd540e1b83f6b8a3f425b38847`

## 9. AP-G0C TP2 adjudication

Final run ID: `ap_g0c_tp2_20260829T214339CST`.

The frozen AP-G0C protocol ran from source revision
`bf81c36ac89785cef48f32de046e65061160d4ea`. Each platform used devices 0 and 1,
one rank per device, exact `int64[4]` payloads, and a separate torchrun job for
every fresh process pair.

| Litmus | A100/NCCL | 910B/HCCL |
| --- | --- | --- |
| stale generation dispatch | detected | detected |
| 128 matched same-generation epochs | pass | pass |
| clean destroy/recreate | 5/5 `capability_pass` | 5/5 `capability_pass` |
| partial epoch then recreate | 5/5 `expected_timeout_recovered` | 5/5 `reinitialization_capability_failure` |

For every 910B partial-epoch pair, both ranks published `fault_ready`,
`fault_observed`, and `ready_destroy`. HCCL's watchdog reported the designated
rank-0 collective timeout and announced process termination. Rank 1 returned
from group destruction and published `destroyed`; rank 0 remained in the
teardown path, rank 1's bounded out-of-band wait expired, and torchrun then
terminated rank 0. The five pairs produced the same structured rank exits
(`rank 0=-15`, `rank 1=4`) and the same last valid phase.

This is an in-process communicator-reinitialization capability limitation, not
an AP-G0C protocol violation. The incomplete collective is the intentional
stimulus, and the public process-group reinitialization path is not a qualified
contract in this gate. A100's successful recovery is a matched capability
control, not evidence that HCCL violated a promised API.

Preflight and postflight snapshots found no accelerator compute process on
either platform. The retrieved evidence contains 24 run manifests and 452
manifested artifacts; every size and SHA-256 entry validated.

Evidence locations:

```text
/data/AccelPact-bf81c36/results/ap_g0c_tp2_20260829T214339CST
results/raw_work/ap_g0c_tp2_20260829T214339CST
```

Archive SHA-256:

- A100: `79f1955fdbca6f9ffa7ef25875eb4bee7f11b1e71d86d33830ea836c28cc39f0`
- 910B: `ca0e478613d7e64f0677fcdb1412c95911f1c7662451e9818cafd6999314aec6`

The canonical compact result is
[`results/ap_g0c_tp2_summary.json`](../results/ap_g0c_tp2_summary.json).

Commissioning attempts remain immutable but are excluded from the scientific
matrix. They exposed three harness issues: an NPU preflight incorrectly queried
NCCL metadata, the adjudicator command was constructed before marker creation,
and backend-fatal worker evidence was initially left at a conservative harness
classification. Each issue received a focused regression test before the final
uniform rerun.

## 10. Exact next actions

1. Keep AP-G0Q and AP-G0C raw evidence immutable.
2. Freeze a separate repair gate before execution. Its 910B path should replace
   the failed in-process communicator recreation with a fresh worker-process
   pair, then require 128 exact epochs on a new communicator generation.
3. Measure process-replacement recovery latency separately from protocol
   correctness; do not compare raw A100 and 910B timings as performance results.
4. Continue the allocator/event/graph sequence families to search for a genuine
   current-contract violation independent of the known PyTorch regression seed.
5. Gate generic repair synthesis and workload integration on the stricter
   multi-root threshold below.

## 11. Scientific decision boundary

The evidence qualifies both AP-G0Q and AP-G0C oracle sensitivity. It confirms a
known PyTorch CUDA-path failure class and a separate, vLLM-independent HCCL
reinitialization capability limitation. The latter does not count as a protocol
violation under the frozen claim boundary.

AccelPact earns a full systems implementation only if later experiments find at
least two current-stack violations with different roots, including one
independent of vLLM and one causing a silent stale read, deadlock, or poisoned
fallback. At least one violation must arise from a pre-frozen
protocol-generated sequence rather than copying a public issue reproducer.

If all new findings are documented capability boundaries, invalid usage, or
already-known issue reproductions, stop at a regression/conformance suite; do
not present it as an ASPLOS system.
