# AccelPact idea assessment

## Optimized problem boundary

The defensible problem is not whether CUDA and CANN expose identical APIs or
produce identical concrete traces. Their documented event and graph semantics
already differ. The research question is:

> Can one abstract ownership, publication, capture, and collective protocol be
> lowered into backend-specific legal traces, then checked for stale reads,
> premature reuse, non-atomic abort, duplicate publication, and deadlock?

The detector must therefore use a common invariant oracle plus explicit
backend-specific lowering rules. A concrete CUDA/CANN difference is evidence
only when one lowering violates the shared abstract contract; a documented API
difference is not a bug.

## Contribution that could survive prior work

The strongest potential contribution is a runtime-protocol conformance system:

1. a small protocol language over resource generations and happens-before
   edges;
2. backend-specific lowering into streams, events, graphs, allocator reuse, and
   collective epochs;
3. state-preserving metamorphic variants and a protocol-aware failure oracle;
4. event-DAG reduction and, only after confirmed violations, local repair.

This must stay distinct from kernel-pipeline barrier verification, numerical
divergence localization, generic CUDA program fuzzing, and API coverage fuzzing.

## Closest overlap risks

- [**AccelSync**](https://arxiv.org/abs/2605.07881) verifies synchronization
  coverage inside accelerator pipeline programs. AccelPact remains at the host
  runtime layer.
- [**OpGuard**](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)
  fingerprints tensors to locate numerical divergence in LLM training.
  AccelPact checks lifecycle and recovery invariants without requiring
  cross-platform numerical equality.
- [**cuFuzz**](https://research.nvidia.com/publication/2026-03_hunting-cuda-bugs-scale-cufuzz)
  and [**CuFuzz**](https://doi.org/10.1145/3808170) are the strongest overlap
  risk. Recent systems fuzz whole
  CUDA programs or generate context-dependent CUDA library API sequences.
  Merely adding CANN or naming API order as a protocol is insufficient.
  AccelPact needs a cross-runtime abstract trace oracle, stateful metamorphic
  relations, and protocol-level reduction/repair that those coverage-oriented
  systems do not provide.
- [GPU Progress Litmus](https://arxiv.org/abs/2109.06132) and classical GPU
  memory-model litmus tests are important methodological precedents for abstract
  model to backend lowering. They do not cover the same host-runtime resource
  generations and recovery semantics.

Collective recovery is generation-based rather than transactional. Returning
from `ncclGroupEnd` is not device completion, and an asynchronous error leaves
completion unknown. Abort destroys the communicator; recovery creates a new
generation. The collective protocol is therefore:

```text
epoch-open(g) -> enqueued -> completed -> reusable-same-generation
                         \-> failed-unknown -> aborting -> destroyed(g)
                                                    \-> recreated(g+1)
```

See the official [NCCL group-call](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
and [communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html)
semantics.

Novelty remains provisional until a broader current search covers API-state
fuzzing, distributed collective recovery, and graph-capture testing.

## Causal chain

Backend-specific legal lowering of one abstract protocol
→ generation-aware trace observation
→ detect a violated ownership or recovery invariant
→ reduce the event DAG while preserving the violation
→ insert the smallest backend-valid ordering/lifetime action
→ remove the failure without global synchronization.

The final two arrows are not implemented until Gate 0 finds a stable violation.

## Main reviewer risks

| Risk | Class | Required resolution |
| --- | --- | --- |
| Only rediscovers documented API differences | design-fixable | Judge abstract invariants, not concrete trace equality |
| Becomes a collection of vendor bugs | requires-new-result | Show one protocol abstraction explains multiple independent roots |
| Overlaps API fuzzers | evidence-fixable | Compare sequence validity, state coverage, reduction, and unique bugs |
| Automatic repair is heuristic or global synchronization | requires-new-result | Prove local repair preserves legal traces and measure overhead |
| Only vLLM reproductions | requires-new-result | Include at least one independent runtime reproducer and three application classes |
| Intentional invalid API calls manufacture failures | design-fixable | Keep negative controls separate from scientific violations |
