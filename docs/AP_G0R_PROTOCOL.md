# AP-G0R: TP1 allocator-retirement qualification

Status: **FROZEN BEFORE ACCELERATOR EXECUTION**

AP-G0R tests the lifetime boundary between non-default stream work and the
framework-native caching allocator. It asks whether `Tensor.record_stream`
prevents a released tensor's storage from being reassigned until every
registered consumer has completed the work that was already queued at release
time.

This gate is separate from AP-G0Q. AP-G0Q kept its buffer storage alive and
reused the same tensor object; AP-G0R releases the final Python owner, requests
same-size allocations, and observes the allocator generation before and after
consumer completion.

## Contract basis

PyTorch 2.10 and 2.11 state that `Tensor.record_stream(stream)` marks a tensor
as used by the stream and prevents its memory from being reused until the work
queued on that stream at deallocation has completed. The same documentation
allows an explicit alternative: synchronize every non-creation stream back to
the creation stream before deallocation.

- <https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.record_stream.html>
- <https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.record_stream.html>
- <https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-streams>
- <https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-streams>

TorchNPU 26.1 directs native-API use to the PyTorch 2.10 contract subject to its
listed NPU restrictions. In that exact table, `Tensor.record_stream` is
supported on Atlas A2/A3 with a `float32` tensor restriction, while
`torch_npu.npu.Stream.wait_stream` is supported on A2/A3. The official allocator
documentation further states that `record_stream` marks a pool allocation to
prevent early return and that the pool queries device events before safe
release. AP-G0R therefore uses only a `float32` payload on 910B.

- <https://www.hiascend.com/document/detail/zh/Pytorch/2610/apiref/nativeapi/docs/zh/native_apis/pytorch_2-10-0/overview.md>
- <https://www.hiascend.com/document/detail/zh/Pytorch/2610/apiref/nativeapi/docs/zh/native_apis/pytorch_2-10-0/torch-Tensor.md>
- <https://www.hiascend.com/document/detail/zh/Pytorch/2610/apiref/nativeapi/docs/zh/native_apis/pytorch_2-10-0/torch-cuda.md>
- <https://www.hiascend.com/document/detail/zh/Pytorch/2610/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-erase_stream.md>

The version map binds torch-npu `2.10.0.post4` to branch
`v2.10.0-26.1.0`, PyTorch 2.10.0, and CANN 9.1.0. The exact branch exposes
`record_stream`; its native caching allocator records usage streams, inserts
per-stream events during free, and returns the block to its free pool only after
those events complete. These source links explain the mechanism; the public
documents above define the claim boundary.

- <https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md>
- <https://github.com/Ascend/pytorch/blob/v2.10.0-26.1.0/torch_npu/csrc/aten/npu_native_functions.yaml>
- <https://github.com/Ascend/pytorch/blob/v2.10.0-26.1.0/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp>

The public-API preflight confirmed that the deployed CUDA and NPU tensors both
accept their native stream objects in `record_stream`. Both stacks report the
native allocator backend. This preflight is capability evidence, not a
scientific run.

## Frozen scope

| Platform | Qualified stack | Device | Allocator |
| --- | --- | --- | --- |
| A100 | PyTorch 2.11.0+cu128, CUDA 12.8 | 0 | native caching allocator |
| 910B | PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0 | 0 | native caching allocator |

- TP1, eager execution, one fresh process per litmus.
- Exact `torch.float32` copy-only payloads with `2,097,152` elements (8 MiB).
  Values are integers in `[1, 128]`, exactly representable in `float32`; no
  arithmetic is performed on the payload.
- Seed `20260829`; 128 generations for every valid litmus.
- No graph, pinned-host transfer, collective, model, custom allocator, or
  `cudaMallocAsync` claim.
- A float32 matrix workload creates a scheduling window only. Its values are
  not evidence and are excluded from cross-platform comparisons.
- Every process has a 180-second external deadline and a new result directory.

## Abstract state and ownership

Each generation uses the existing Buffer oracle:

```text
owned -> write-pending -> ready -> published -> consumed -> reclaimable
```

These are logical ownership states, not device-completion states. `consumed`
means that every declared consumer read has been enqueued and its retirement
obligation is represented in the ledger. `reclaimable` means that the final
owner may hand the storage to the caching allocator because safe stream order
or allocator retirement responsibility has been established. It does **not**
mean that a pending block is already physically reusable on an arbitrary
stream. Event completion and physical reuse eligibility remain separate fields
in the retirement ledger.

The lowering attaches a retirement ledger:

```text
actual_consumers
registered_consumers
handed_back_consumers
release_requested
consumer_completion
allocator_reuse_before_completion
allocator_reuse_after_completion
```

A release is admissible when one of the following holds:

1. every consumer is the creation stream;
2. every non-creation consumer is present in `registered_consumers`; or
3. the creation stream has an explicit wait edge from every non-creation
   consumer before release.

The host guard rejects a release that satisfies none of these conditions. A
negative-control rejection occurs before any invalid deallocation or allocator
reuse request reaches the backend.

The symbolic `creation` role is the tensor's real allocation stream, not merely
the stream that later writes it. Source allocation and generation fill occur
inside that stream context. Every churn tensor and post-completion probe is also
allocated and poison-written inside the same `creation` stream context. Each
generation records the source-allocation, source-write, churn-allocation,
churn-write, probe-allocation, and probe-write stream roles; a role mismatch is
a harness error.

## Frozen scheduling window

The runner preallocates two `2048 x 2048` float32 matrices and one output matrix
per consumer. It warms the exact matrix path before measurement. For every
generation, an actual consumer stream performs 16 matrix multiplications before
copying the generation tensor into a retained output. A fresh completion event
is recorded after the copy.

Before any scientific cell, an excluded eight-generation construct calibration
runs the same 16-multiply path while retaining the source. All eight completion
events must be incomplete at the host query, a stricter coarse screen than the
120-of-128 scientific threshold. Calibration may only accept or reject the
frozen workload; it cannot tune the multiplier count after observing a
platform. Failure stops the campaign and requires a new protocol/config
revision.

Calibration is one standalone fresh-process run per platform. It writes its own
JSONL, adjudication summary, launcher evidence, and SHA-256 manifest, bound to
platform, source revision, full config digest, and run ID. Every later
scientific launcher must verify that artifact and pass only its run ID and
adjudication-summary SHA-256 into the fresh runner. Scientific rows reference
those two values and do not copy the eight observations.

Immediately before releasing the source, the runner queries every completion
event. A generation is pending-qualified only when the relevant event still
reports incomplete. At least 120 of 128 valid generations must be
pending-qualified; otherwise the scheduling window is a harness error.

Before each measured generation, the runner synchronizes all gate streams,
drops generation-local references, and calls the platform allocator's public
`empty_cache` operation exactly once (`torch.cuda.empty_cache` or
`torch.npu.empty_cache`). It then allocates and fills the source in the real
`creation` stream context. After source release, that same stream requests and
poison-writes eight retained same-size churn tensors. After consumer completion,
it requests and poison-writes up to eight additional same-size probes while the
churn tensors remain live. For generation `g`, every churn/probe poison value is
the exact `float32` value `-(g + 1)`, which cannot equal the positive source
generation. This isolates the source block from other same-size free blocks
without relying on raw allocator interfaces. The per-generation ledger records
the empty-cache call ordinal and cumulative call count.

A generation is reuse-qualified when either:

- the source address is reassigned to a churn tensor in a same-stream or
  explicit-handback schedule and an immediate event query after pointer equality
  still reports at least one relevant consumer pending; or
- the same generation was pending-qualified at release in a `record_stream`
  schedule and its source address is either reassigned to a churn tensor whose
  pointer-equality query shows every consumer already complete, or reassigned to
  a post-completion probe.

The release-time query is not reused to label pointer equality. On every source
pointer equality, the runner queries each relevant completion event again before
enqueueing the poison write and records the per-consumer bitmap. Only an event
still incomplete at that second query can support
`allocator_reuse_before_completion`. For two consumers, both release-time and
pointer-equality bitmaps retain consumer identity. This prevents natural
completion between the release query and allocation from becoming a false
premature-reuse report.

Raw addresses are not persisted. Evidence records only generation-local pointer
equality and allocation ordinal. The predeclared 32-of-128 floor requires at
least 25% direct recycling exposure; it is not fitted from calibration or
results. A valid cell below that floor is `inconclusive`, never a pass.

## Valid litmus

### `allocator_same_stream_recycle_baseline`

The creation stream writes generation `g`, performs the blocker work, copies the
source to the retained output, and records completion. The final Python owner is
released while completion remains pending. Same-size churn allocations occur on
the same stream.

The oracle enters `write-pending` when fill is enqueued, `ready` when the ready
event is recorded, `published` when same-stream consumer work is enqueued,
`consumed` when the copy and completion event are enqueued, and `reclaimable`
when the guard proves that allocation and consumer roles are both `creation`.

The allocator may immediately return the same address because all old and new
uses are serialized on the creation stream. A reuse-qualified generation must
still produce exactly `g`; stale or mixed content is a protocol violation. This
baseline establishes that the tested size class is actually recycled.

### `allocator_cross_stream_record_stream`

The creation stream writes `g` and publishes a ready event. One side stream
waits for ready, performs blocker work, copies the source, and records
completion. The source records that side stream before its final Python owner is
released.

The oracle enters `published` after the side stream has waited on ready and its
consumer work is enqueued, `consumed` after that work and its completion event
are represented in the ledger, and `reclaimable` only after the actual consumer
identity appears in `registered_consumers`.

While completion is pending, no churn allocation may receive the source
address. The retained output must contain exactly `g`. After completion, a
same-size probe must eventually demonstrate that the source block remains
recyclable rather than leaked.

### `allocator_cross_stream_manual_join`

The schedule is identical to the single-consumer cross-stream case except that
it does not call `record_stream`. Before releasing the source, the creation
stream calls `wait_stream(consumer)` after the consumer completion event has
been enqueued. Churn allocations are then issued on the creation stream after
that wait edge; the event remains the host-query witness.

The source address may be returned immediately, but the new writes must remain
ordered after the consumer. A reuse-qualified generation must produce exactly
`g`.

Here `consumed` follows consumer-work enqueue. `reclaimable` follows only after
`wait_stream` has been enqueued back onto the real allocation stream and the
consumer appears in `handed_back_consumers`; physical completion may still be
pending at the host query.

### `allocator_two_consumer_record_stream`

Two side streams independently wait for the ready event, perform their own
blocker work, copy the same source into separate retained outputs, and record
separate completion events. Both streams are registered with the source before
release.

The source address may not be reassigned while either registered consumer is
pending. Both outputs must contain exactly `g`. Post-completion probes establish
that the block becomes recyclable after both consumers complete.

The oracle enters `published` after both waits and consumer paths are enqueued,
`consumed` after both completion events are represented in the ledger, and
`reclaimable` only after both exact consumer identities have been registered.

## Negative controls

### `allocator_missing_consumer_registration`

The retirement ledger contains one actual side-stream consumer, no registered
consumer, and no handback edge. The host guard must reject release, leave the
invalid backend reuse-dispatch count at zero, and then complete an eight-
generation legal `record_stream` follow-up. The follow-up must have all eight
generations pending-qualified, at least two reuse-qualified, no premature reuse,
and exact outputs.

### `allocator_wrong_stream_registration`

The ledger contains one actual consumer but registers a distinct decoy stream.
The host guard must reject release by stream identity, leave the invalid backend
reuse-dispatch count at zero, and then complete the same legal follow-up.

Negative controls qualify generator and oracle sensitivity. They are never
runtime violations. Their real consumer work and decoy registration may be
dispatched, but the invalid deallocation/reuse step is never dispatched.

## Result contract

Each fresh process writes one JSONL row and one run summary. A valid row records:

- protocol, run, source revision, config digest, platform, backend, allocator,
  device, litmus, repetition, seed, and measurement boundary;
- exact tensor dtype/size, blocker dimensions/count, churn/probe counts, and
  process deadline;
- actual, registered, and handed-back symbolic stream roles;
- a `generations` ledger with one entry for every configured generation; each
  entry contains `oracle_initial_state`, ordered `oracle_transition_targets`,
  `oracle_final_state`, `pending_at_release`, a consumer-keyed
  `pending_bitmap_at_pointer_reuse`, release decision, record count, completion
  count, pre-completion reuse ordinal, post-completion reuse ordinal,
  `reuse_qualified`, mismatch count, and exact logical payload digest;
- per-generation source-allocation, source-write, churn-allocation, churn-write,
  probe-allocation, and probe-write stream roles, plus the empty-cache call
  ordinal and cumulative count;
- pending-qualified count, reuse-qualified count, stale-generation count,
  premature-reuse count, and timeout/error details;
- verified construct-calibration run ID and adjudication-summary SHA-256. The
  standalone calibration row, not each scientific row, holds its eight pending
  observations.

The output check uses exact equality over every element with no tolerance. Each
expected integer generation is exactly representable in `float32`. The logical
payload digest covers generation, element count, expected IEEE-754 value,
observed minimum and maximum, and mismatch count. Delay-work values never enter
the result.

The launcher preserves the command, frozen environment overrides, runtime and
allocator versions, device inventory, stdout/stderr, process exit, deadline
status, JSONL, summary, and SHA-256 manifest. It never overwrites an existing run
directory.

## Classification

| Classification | Meaning | Exit |
| --- | --- | --- |
| `construct_calibration_pass` | excluded 8/8 pending-window calibration passed | 0 |
| `valid_pass` | valid schedule, sufficient pending/reuse exposure, all exact | 0 |
| `negative_control_detected` | guard rejected invalid retirement; legal follow-up passed | 0 |
| `protocol_violation` | valid supported schedule reused too early, read stale/mixed content, or violated same-stream/handback order | 2 |
| `negative_control_missed` | guard admitted an invalid retirement request | 3 |
| `harness_error` | failed calibration, corrupt/mismatched evidence, insufficient pending window, launch error, or unlocatable phase | 4 |
| `inconclusive` | valid schedule ran but fewer than 32 generations exposed actual block recycling | 5 |
| `unsupported` | the selected native stack does not expose the required public API | 5 |

Only supported valid litmus may produce `protocol_violation`. Pointer reuse or
wrong content from an intentionally unregistered schedule cannot produce a
runtime claim.

## Adjudication order

1. Run the excluded pending-window calibration once per platform. Both must
   reach eight pending observations out of eight; otherwise stop before
   any scientific cell.
2. Run both negative controls once per platform. Both guards must detect the
   invalid schedule and their legal follow-ups must pass.
3. Run the same-stream recycle baseline once per platform. It must reach
   `valid_pass` before any cross-stream claim is considered.
4. Run the three cross-stream valid litmus once per platform.
5. A supported valid abnormal cell is rerun in five fresh processes. At least
   four complete matching outcomes are required for a candidate violation.
6. A negative-control miss, source/config mismatch, invalid manifest, or
   postflight process residue stops the platform.

## Claim boundary

AP-G0R can establish a current-stack host-runtime protocol violation only when a
valid supported lowering has a pending-qualified generation and then either:

- reassigns the recorded source block before the registered consumer set
  completes;
- exposes a stale or mixed generation; or
- violates the safe ordering of the same-stream or explicit-handback baseline.

No-reuse cells are `inconclusive`, not passes. Negative controls, unsupported
custom allocators, premature host-buffer mutation, graph resource release, and
collective recovery are outside this gate.

Premature pointer equality while a registered consumer is pending directly
supports an allocator-retirement attribution. Stale or mixed output without
that equality still supports a valid-lowering host-runtime ordering failure,
but its allocator root remains unassigned until trace evidence identifies the
reuse path.
