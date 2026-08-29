# AP-G0Q: stateful protocol oracle qualification

Status: **FROZEN BEFORE ACCELERATOR EXECUTION**

AP-G0Q is an engineering/scientific qualification slice, not the full AccelPact
Gate 0. It tests whether the current A100/CUDA and 910B/CANN stacks support a
small, auditable protocol oracle and whether any valid sequence leaves poisoned,
stale, or prematurely reused state.

## Frozen environment boundary

- A100: one A100 PCIe 40 GB, PyTorch 2.11.0+cu128, CUDA 12.8.
- 910B: one Ascend 910B4-1, PyTorch 2.10.0, torch-npu 2.10.0.post4,
  CANN 9.1.0.
- TP1 only; no model, vLLM, NCCL, or HCCL in AP-G0Q.
- Exact integer payloads; cross-platform floating-point equality is irrelevant.
- Both current PyTorch stacks use their native caching allocator. AP-G0Q makes
  no claim about `cudaMallocAsync` or other allocator backends.
- Seed `20260829`, 128 generations per iterative litmus.
- One fresh-process discovery run per platform. Any abnormal supported cell is
  rerun in five fresh processes.

## Abstract invariants

1. A consumer may observe generation `g` only after the publication edge for
   `g`.
2. A buffer may become reclaimable only after every registered consumer of its
   generation completes.
3. Re-recording an event must not make an earlier registered waiter observe a
   later generation.
4. A committed graph replays using the live contents of retained external
   addresses, not a rebound Python object.
5. Every captured auxiliary stream must rejoin the capture origin before commit
   on both CUDA and CANN.
6. An aborted capture must either permit clean eager fallback in the same
   process or expose an explicit unsupported/poisoned state. Silent partial
   recovery is a violation.

## Valid litmus

| ID | Resource | Required observation |
| --- | --- | --- |
| `event_rerecord_single_waiter` | event/buffer | 128/128 exact generations after re-record |
| `event_multi_waiter` | event/buffer | both waiters observe each generation before legal reset/reuse |
| `buffer_reuse_with_ready_and_consumed_events` | buffer | no reuse before both ready and consumed edges |
| `graph_static_input_generations` | graph/buffer | one retained address accepts 128 content generations |
| `graph_cross_stream_join` | graph/event | valid fork/join captures and replays exactly |
| `capture_abort_eager_recovery` | graph/RNG/allocator | expected capture error followed by working eager RNG, allocation, and stream |

An API combination that the backend explicitly reports as unsupported is
recorded as `unsupported`, not as a protocol violation. A timeout, silent stale
read, wrong generation, or fallback poison is retained as an abnormal result.

## Negative controls

- `graph_missing_cross_stream_join` intentionally omits the join edge and should
  be rejected or otherwise identified as invalid.
- `graph_rebound_external_input` intentionally rebinds the Python tensor while
  leaving the captured address unchanged; the oracle must identify the old
  address semantics.

Negative controls qualify oracle sensitivity. They never count as discovered
runtime violations.

## Timing and failure handling

- Device initialization is outside the per-litmus duration but inside process
  wall time.
- Every observation synchronizes only the streams required by the protocol.
- Each process has a 120-second external timeout.
- Raw stdout, stderr, exit code, JSONL, hardware/software versions, and timeout
  status are preserved. A crash or timeout is never deleted from the run set.

## Decision

The AP-G0Q oracle is qualified only if:

1. every supported valid control passes on both platforms;
2. both negative controls are detected where their APIs are supported.

Once the oracle qualifies, run the separately frozen TP2 collective
qualification even if AP-G0Q is clean, because communicator generations add a
different state domain. A non-negative-control abnormality earns repair work
only after it reproduces in at least four of five fresh processes.

If the harness itself is invalid on either platform, repair the qualification
construct once without changing the abstract invariants. If all valid cells pass
and no state poison appears, AP-G0Q contains no TP1 violation; AccelPact still
does not earn an automatic-repair implementation. Known issue reproductions may
be retained as controls but do not alone establish novelty.

The full project threshold remains stricter: at least two current-stack protocol
violations with different roots, one independent of vLLM, one causing silent
stale state/deadlock/poisoned fallback, and a local repair below roughly 3%
overhead in real workloads.
