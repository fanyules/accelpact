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

## 9. Exact next actions

1. Keep the AP-G0Q raw evidence immutable and retain the A100 result as a known
   regression seed, not an AccelPact novelty claim.
2. Freeze a separate TP2 communicator-generation protocol before implementation.
   An aborted communicator must be destroyed; recovery creates a new generation
   rather than reusing the aborted object.
3. Give TP2 its own process timeout, result manifest, device/rank ownership, and
   fresh-process confirmation rule before accelerator execution.
4. The confirmed A100 non-negative-control result meets the frozen 4/5 threshold
   for scoped diagnosis and local repair experiments, while remaining labeled as
   a reproduction of a known failure class.
5. Gate generic automatic repair and full-system investment on the stricter
   multi-root project threshold below.

## 10. Scientific decision boundary

The adjudicated evidence qualifies oracle sensitivity and confirms one known
PyTorch CUDA-path failure class. AccelPact earns a systems implementation only if
later experiments find at least two current-stack violations with different roots,
including one independent of vLLM and one causing a silent stale read, deadlock,
or poisoned fallback. At least one violation must arise from a pre-frozen
protocol-generated sequence rather than copying a public issue reproducer.

If all new findings are documented invalid usage or already-known issue
reproductions, stop at a regression/conformance suite; do not present it as an
ASPLOS system.
