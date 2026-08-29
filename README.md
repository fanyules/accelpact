# AccelPact

AccelPact studies stateful host-runtime protocols across disjoint accelerator
stacks. Its scope is stream/event ordering, buffer ownership, graph
capture/replay, allocator reuse, collective epochs, and failure recovery. It
does not compare cross-platform floating-point values and does not inspect
barriers inside accelerator kernels.

The repository has completed the **AP-G0Q** adjudication slice. Oracle
sensitivity is qualified: negative controls are detected on both platforms,
including the isolated 910B rebound-input adjudication, and five matched
fresh-process trials distinguish a reproducible A100 capture-abort recovery
failure from clean 910B fallback. The A100 result matches an already-public
PyTorch failure class, so it is retained as a regression seed and not claimed as
an AccelPact discovery.

The AP-G0Q protocol is frozen in
[`docs/AP_G0Q_PROTOCOL.md`](docs/AP_G0Q_PROTOCOL.md).
No automatic repair, model workload, or large fuzzing framework is implemented.
The TP2 communicator-generation protocol is frozen in
[`docs/AP_G0C_PROTOCOL.md`](docs/AP_G0C_PROTOCOL.md) and awaits implementation.

## Local checks

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```
