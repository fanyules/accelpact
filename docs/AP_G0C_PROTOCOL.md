# AP-G0C: TP2 communicator-generation qualification

Status: **FROZEN BEFORE ACCELERATOR EXECUTION**

AP-G0C is the collective qualification slice that follows AP-G0Q. It asks
whether a two-rank accelerator process group preserves exact epoch generations,
rejects stale generation dispatch before backend submission, and exposes a
bounded destroy/recreate capability after clean completion or an incomplete
collective epoch.

This gate measures host-runtime behavior. It does not exercise graph capture,
TP4, models, large tensors, cross-host collectives, or performance scaling.

## Claim and support boundary

1. A successfully completed communicator generation may be reused for another
   matched epoch.
2. An outcome-unknown generation may never return to
   `reusable_same_generation`.
3. A request carrying a stale generation must be rejected before NCCL/HCCL
   dispatch.
4. Recovery, when the public process-group path supports it, destroys the old
   group on every rank, uses non-distributed out-of-band synchronization, and
   creates a distinct generation before any new collective.

PyTorch 2.10 and 2.11 document runtime process-group reinitialization as
unsupported/untested even when processes synchronize out of band. Therefore
the entire clean or post-timeout recreate path is a capability probe. Creation
failure, post-creation wrong data, or post-creation incomplete work remains a
capability failure unless a separate gate first freezes an explicit backend
contract. Only the normal matched-collective baseline in this gate can produce a
`protocol_violation` classification.

Official API basis:

- <https://docs.pytorch.org/docs/2.10/distributed.html#shutdown>
- <https://docs.pytorch.org/docs/2.11/distributed.html#shutdown>
- <https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.new_group>
- <https://docs.pytorch.org/docs/2.11/distributed.html#torch.distributed.new_group>

## Frozen environment boundary

| Platform | Device group | Qualified stack | Device IDs |
| --- | --- | --- | --- |
| A100 | NCCL | PyTorch 2.11.0+cu128, CUDA 12.8, NCCL 2.28.9 | 0, 1 |
| 910B | HCCL | PyTorch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0 | 0, 1 |

- `world_size=2`, one rank per visible device, one host per platform.
- The incomplete-epoch claim is intentionally limited to designated fault rank
  0; rank-role symmetry is outside this gate.
- Gloo is the default control process group; NCCL/HCCL is a separate device
  group created with `torch.distributed.new_group`.
- After device-group destruction and before recreation, synchronization uses
  only atomic files in the fresh run directory. No `torch.distributed` primitive
  is called in that interval.
- Each litmus uses a fresh torchrun job with automatic restarts disabled.
- Seed `20260829`; default stream; `torch.int64`; tensor shape `[4]`.
- Generation `g`, epoch `e`, rank `r` contributes `[g, e, r, 1]`. SUM must
  produce `[2g, 2e, 1, 2]` on both ranks.
- `control_group_timeout_seconds=60`,
  `device_group_timeout_seconds=15`,
  `work_wait_timeout_seconds=15`,
  `oob_phase_timeout_seconds=60`, and
  `outer_process_timeout_seconds=180`.
- Gloo initialization uses a 60-second timeout. Every NCCL/HCCL `new_group`
  receives `timeout=timedelta(seconds=15)`. Every device all-reduce uses
  `async_op=True`, followed by `Work.wait(timedelta(seconds=15))`.
- A100 launches with `TORCH_NCCL_BLOCKING_WAIT=1`; 910B launches with
  `HCCL_BLOCKING_WAIT=1`. If either stack terminates a worker instead of
  returning a Python error, the external adjudicator uses the last valid marker,
  launcher status, and outer timeout to classify the capability outcome.
- Every output path is new. Rank 0 and rank 1 write separate JSONL and logs.

## Abstract state

One `LifecycleOracle` represents one communicator generation. AP-G0C extends
the collective transition table with a successful-epoch loop and a clean-retire
branch:

```text
epoch_open -> enqueued -> completed -> reusable_same_generation
reusable_same_generation -> epoch_open
reusable_same_generation -> retiring -> destroyed -> recreated
epoch_open -> enqueued -> failed_unknown -> aborting -> destroyed -> recreated
```

Each epoch advances through the success path; every epoch except the final one
loops back to `epoch_open`. The clean branch retires a completed generation. The
failure branch retires an outcome-unknown generation. `recreated` records that a
new process-group object was returned; the replacement generation receives a
new oracle beginning at `epoch_open`.

Forbidden transitions:

- `failed_unknown -> reusable_same_generation`;
- dispatch with a generation other than the current generation;
- `reusable_same_generation -> failed_unknown` without opening and enqueueing a
  new epoch;
- recreation before both rank-specific destroyed markers are visible;
- any distributed barrier between device-group destruction and recreation;
- reuse of the old Python process-group object after destruction.

## Frozen litmus

### `stale_generation_dispatch`

This is an oracle-sensitivity control. The host ledger first records generation
0 as retired and generation 1 as current. Each rank then submits a request
labeled generation 0 to the host guard. The guard must reject it before backend
dispatch, leave the stale-request dispatch count at zero, and then allow one
legal generation-1 all-reduce to complete exactly.

Required classification: `negative_control_detected`.

### `collective_same_generation_reuse`

Create device group generation 0 and run 128 matched all-reduce epochs. Every
epoch must return the exact expected integer vector on both ranks. There may be
no stale generation, duplicate completion, timeout, or rank disagreement.

Required classification: `valid_pass`.

### `collective_clean_destroy_recreate`

On generation 0, complete 8 matched warm-up epochs. Destroy the device group on
both ranks, publish per-rank destroyed markers, wait for both markers using only
the filesystem, and call `new_group` in the same rank order to create generation
1. Run 128 exact epochs on generation 1.

Possible classifications:

- `capability_pass`: generation 1 is distinct and all 128 epochs pass;
- `capability_unavailable`: the runtime explicitly rejects or does not
  provide the reinitialization path;
- `reinitialization_capability_failure`: destroy, recreation, or subsequent
  generation-1 use fails after the capability path is attempted;
- `capability_timeout`: a complete, valid schedule reaches group creation,
  destruction, or generation-1 use but exceeds the outer deadline;
- `harness_error`: evidence, rank, or out-of-band coordination is invalid.

This litmus cannot produce `protocol_violation` solely because recreation is
unsupported or because any post-recreate operation fails.

### `collective_partial_epoch_timeout_recreate`

Create generation 0 and complete one matched warm-up epoch. At fault epoch 1,
rank 0 enqueues all-reduce and waits with the frozen deadline while rank 1 does
not call the device collective. Both ranks coordinate the fault phase through
atomic files. After the incomplete epoch is observed, both ranks destroy
generation 0, synchronize using destroyed markers, create generation 1, and run
128 exact recovery epochs.

The partial epoch is an intentional failure stimulus and never counts as a
runtime discovery. Possible classifications are:

- `expected_timeout_recovered`: the incomplete epoch is detected, generation 0
  is retired, and generation 1 completes all 128 epochs;
- `capability_unavailable`, `reinitialization_capability_failure`, or
  `capability_timeout`: same meaning as the clean recreate probe;
- `harness_error`: incomplete evidence, wrong schedule, or control failure.

## Out-of-band coordination

The run directory is fresh. Each rank writes and `fsync`s a unique temporary
file, then publishes it with a same-filesystem hard link to the final marker.
Hard-link creation must fail if the final name already exists; the runner then
reports a harness error instead of overwriting. A phase is complete only when
both final markers exist and contain the current run ID, litmus ID, rank, and
schedule digest.

The clean recreate litmus uses:

```text
ready_destroy -> destroyed -> recreate_start
```

Required phases for the timeout litmus:

```text
fault_ready -> fault_observed -> ready_destroy -> destroyed -> recreate_start
```

For rank 0, `fault_observed` means `Work.wait` returned an error or exceeded its
deadline. Rank 1 never submits the fault collective; its `fault_observed` marker
acknowledges the validated rank-0 marker and is not a local timeout claim.

Marker timeout, stale marker content, duplicate publication, or missing rank is
a harness error. The runner never deletes or rewrites markers.

## Result contract

Each rank writes only its own JSONL. Epoch ranges are aggregate records: the
oracle transition history contains every epoch loop, while observations retain
the per-epoch completion ledger and digests.

Per-rank cardinality follows resources that actually existed:

- initial device-group creation failure: 0 rows; the run manifest records the
  attempted generation and external classification;
- `stale_generation_dispatch` and `collective_same_generation_reuse`: 1 row
  after generation creation;
- clean or partial recreate when generation 0 exists but recreation has not
  returned a group: at most 1 generation-0 row;
- clean or partial recreate after `new_group` returns generation 1: exactly 2
  rows, generation 0 followed by generation 1.

No generation-1 `LitmusResult` may be created before a replacement process-group
object exists. A worker termination or outer timeout may leave fewer rows; the
adjudicator must not fabricate missing rank records and instead uses the run
manifest and last valid phase marker.

Each result records:

- run ID, protocol ID, source revision, backend, rank, local rank, world size,
  and logical device;
- litmus, repetition, generation, epoch range, seed, dtype, shape, reduce op,
  and shared schedule digest;
- local enqueue role, planned submit ranks, backend dispatch count, work-handle
  presence, completion count, and exact expected/observed payload digests;
- enqueue, completion, timeout, destroy, and recreate timestamps when applicable;
- old-object reuse status, distinct replacement-object status, destroy marker
  bitmap, and current generation;
- classification, normalized error type, raw rank-log reference, final state,
  and transition history.

The collective oracle is a replicated global-state view on every rank. In the
partial epoch, both rank records include `epoch_open -> enqueued ->
failed_unknown` because rank 0 enqueued the group operation; observations still
record `local_enqueued=false` for rank 1.

### Phase-to-row mapping

| Litmus and reached phase | Rows per surviving rank | Row classification |
| --- | --- | --- |
| Any initial `new_group` explicitly unavailable | 0 | run-level `capability_unavailable` |
| Any other initial-group creation failure | 0 | run-level `harness_error` |
| Stale guard misses before vendor dispatch | 1 g1 | `negative_control_missed` |
| Stale guard detects; legal initial g1 collective passes | 1 g1 | `negative_control_detected` |
| Stale guard detects; legal initial g1 collective fails | 1 g1 | `protocol_violation` |
| Same-generation 128-epoch baseline passes/fails | 1 g0 | `valid_pass` / `protocol_violation` |
| Clean/partial pre-destroy warm-up fails | 1 g0 | `protocol_violation` |
| Destroy returns an error before `destroyed` | 1 g0 | `reinitialization_capability_failure` |
| Recreate is explicitly unavailable | 1 g0 ending `destroyed` | `capability_unavailable` |
| Recreate returns an error without g1 | 1 g0 ending `destroyed` | `reinitialization_capability_failure` |
| Clean recreate and legal g1 use pass | 2, g0 then g1 | `capability_pass`, `capability_pass` |
| Timeout recreate and legal g1 use pass | 2, g0 then g1 | `expected_timeout_recovered`, `capability_pass` |
| Recreate returns g1, but legal g1 use fails | 2, g0 then g1 | both rows `reinitialization_capability_failure` |
| Complete schedule exceeds an API or outer deadline | phase-dependent | run-level `capability_timeout` |

For the partial litmus, failure to observe the designated rank-0 stimulus is
`inconclusive`, not a backend violation. The run stops before destroy/recreate.

The external run manifest preserves command, timeout status, launcher and rank
exit codes, stack versions, device inventory, stdout/stderr, JSONL, marker files,
and SHA-256 for every artifact. Object addresses, credentials, and private
topology are not recorded.

## Classification and exit contract

| Classification | `passed` | `error` | Adjudicator exit |
| --- | --- | --- | --- |
| `valid_pass` | true | null | 0 |
| `negative_control_detected` | true | null | 0 |
| `capability_pass` | true | null | 0 |
| `expected_timeout_recovered` | true | null | 0 |
| `capability_unavailable` | true | null | 5 |
| `negative_control_missed` | false | required | 3 |
| `reinitialization_capability_failure` | false | required | 5 |
| `capability_timeout` | no fabricated rank row | run-level evidence | 5 |
| `protocol_violation` | false | required | 2 |
| `harness_error` | false | required | 4 |
| `inconclusive` | false | required | 5 |

`protocol_violation` is permitted only for matched collectives executed before
any destroy/reinitialize attempt: the normal baseline, the stale-control
follow-up, or clean/partial warm-up. Any failure after a recreate attempt remains
a capability result in AP-G0C. The external adjudicator is authoritative because
torchrun does not preserve arbitrary per-worker semantic exit codes. Aggregation
priority is incomplete/harness evidence, then protocol violation,
negative-control miss, capability result, and pass.

## Adjudication

1. Run `stale_generation_dispatch` and `collective_same_generation_reuse` once
   per platform. Both must pass before any recreate probe.
2. Any abnormal normal-baseline cell is rerun in five fresh process pairs. At
   least `4/5` complete reproductions are required for a candidate violation.
3. Run `collective_clean_destroy_recreate` in five fresh process pairs. The
   timeout probe may run only if clean recreate is `5/5 capability_pass`.
4. Run `collective_partial_epoch_timeout_recreate` in five matched fresh process
   pairs with the same devices, payloads, deadlines, and generation-1 epoch
   count as the clean recreate control.
5. `5/5` recovery is a supported capability result. At least `4/5`
   `reinitialization_capability_failure` or `capability_timeout` outcomes are a
   reproducible capability limitation. `1-3/5` is inconclusive.
6. A complete planned schedule that reaches device work, destroy, or recreate
   and then exceeds the outer deadline is `capability_timeout` and remains in
   the capability denominator. Missing ranks, mismatched schedules, incomplete
   preconditions, corrupt evidence, or an unlocatable final phase are
   `harness_error` and are excluded.
7. `capability_unavailable` is reported as a boundary and is not a pass. It may
   stop the remaining recreate cells without additional repetitions when the
   public API explicitly rejects the operation.

## Stop conditions

Stop the platform and retain all evidence if:

- two idle devices or the frozen stack cannot be confirmed;
- source revision or config differs between ranks or platforms;
- the stale-generation control is missed;
- the normal same-generation baseline fails;
- schedule digests or rank sets disagree;
- any rank output, log, exit code, marker set, or checksum is missing;
- a run exceeds its outer timeout or leaves a worker behind; classify it as
  `capability_timeout` only when the complete schedule and final phase are
  unambiguous, otherwise as `harness_error`;
- clean recreation is unsupported or fails;
- a replacement group is created but a legal exact collective is stale, wrong,
  duplicated, or incomplete; retain it as `reinitialization_capability_failure`
  under this gate's unsupported/untested reinitialization boundary.

Completion of AP-G0C does not authorize graph collectives, TP4, model workloads,
automatic repair, or performance claims.
